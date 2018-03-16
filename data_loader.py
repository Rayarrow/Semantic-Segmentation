import os
import re
from os.path import join

import numpy as np
from skimage import io, transform

from config import *
from palette_conversion import label_2_colormap

# This files provide the following functions:
# 1. Load images and labels.
# 2. preprocess.

def _load_dataset_VOC(data_home, dataset_path, label_path, data_id, label_ignored):
    # Load the training images, labels and weights. The weights are used to specify the labels to be ignored during
    # the training process.
    X = list()
    y = list()
    y_weights = list()
    ids = list()
    for idx, each_id in enumerate(data_id):
        if idx % 100 == 0:
            logger.info('loadding data {}/{}'.format(idx, len(data_id)))
        X.append(io.imread(join(data_home, dataset_path, each_id + '.jpg')))
        cur_label = io.imread(join(data_home, label_path, each_id + '.png'))
        y.append(cur_label)
        y_weights.append((cur_label != label_ignored).astype(np.float32))
        ids.append(each_id)
    return X, y, y_weights, ids


def load_VOC(data_home, label_ignored=21, resize_shape=None):
    # Define the absolute path of the sub directories.
    dataset_path = join(data_home, 'JPEGImages')
    label_path = join(data_home, 'SegmentationClassLabelImages')
    train_idx_path = r'ImageSets/Segmentation/train.txt'
    val_idx_path = r'ImageSets/Segmentation/val.txt'

    # load the ids of validation images.
    with open(join(data_home, train_idx_path)) as f:
        train_ids = f.read().split()

    with open(join(data_home, val_idx_path)) as f:
        val_ids = f.read().split()

    X_train, y_train, y_train_mask, id_train = _load_dataset_VOC(data_home, dataset_path, label_path, train_ids,
                                                                 label_ignored)
    X_val, y_val, y_val_mask, id_val = _load_dataset_VOC(data_home, dataset_path, label_path, val_ids, label_ignored)

    # Resize the images to the specified shape.
    if resize_shape:
        logger.info('Resizing images...')
        X_train = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in X_train], dtype=np.uint8)
        y_train = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in y_train], dtype=np.uint8)
        y_train_mask = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in y_train_mask], dtype=np.uint8)
        X_val = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in X_val], dtype=np.uint8)
        y_val = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in y_val], dtype=np.uint8)
        y_val_mask = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in y_val_mask], dtype=np.uint8)

    return list(map(np.array, [X_train, y_train, y_train_mask, id_train, X_val, y_val, y_val_mask, id_val]))


def _minhou_cropping_helper(image, label, mask, id, patch_row, patch_col):
    """
    A helper function to crop a images into small patches.
    """
    X_patches = list()
    y_patches = list()
    mask_patches = list()
    id_patches = list()
    # Calculate the number of rows and columns of the image grid to be cropped.
    patch_height, patch_width = image.shape[0] / patch_row, image.shape[1] / patch_col
    for i in range(patch_row):
        for j in range(patch_col):
            top = int(patch_height * i)
            bottom = int(patch_height * (i + 1))
            left = int(patch_width * j)
            right = int(patch_width * (j + 1))

            X_patches.append(image[top: bottom, left: right])
            y_patches.append(label[top: bottom, left: right])
            mask_patches.append(mask[top: bottom, left: right])
            id_patches.append('{}_{}_{}'.format(id, i, j))
    return X_patches, y_patches, mask_patches, id_patches


def random_bunch_sampler(images, label, mask, id, nr, sampling_size):
    """
    Randomly get `nr` sample patches measuring `sampling_size` from each of the images.
    """
    X_patches = []
    y_patches = []
    mask_patches = []
    id_patches = []
    for each_image, each_label, each_mask, each_id in zip(images, label, mask, id):
        cur_X_patches, cur_y_patches, cur_mask_patches, cur_id_patches = random_sampler(each_image, each_label,
                                                                                        each_mask, each_id, nr,
                                                                                        sampling_size)
        X_patches += cur_X_patches
        y_patches += cur_y_patches
        mask_patches += cur_mask_patches
        id_patches += cur_id_patches
    return list(map(np.array, [X_patches, y_patches, mask_patches, id_patches]))


def random_sampler(image, label, mask, id, nr, sampling_size):
    """
    Randomly get `nr` sample patches from the given image.
    """
    X_patches = list()
    y_patches = list()
    mask_patches = list()
    id_patches = list()
    height, width = image.shape[:2]
    tops = np.random.randint(0, height - sampling_size[0], nr)
    lefts = np.random.randint(0, width - sampling_size[1], nr)
    for idx, (each_top, each_left) in enumerate(zip(tops, lefts)):
        if idx % (nr // 10) == 0:
            logger.info('random sampling image {}...  {}/{}'.format(id, idx + 1, nr))
        X_patches.append(image[each_top: each_top + sampling_size[0], each_left:each_left + sampling_size[1]])
        y_patches.append(label[each_top: each_top + sampling_size[0], each_left:each_left + sampling_size[1]])
        mask_patches.append(mask[each_top: each_top + sampling_size[0], each_left:each_left + sampling_size[1]])
        id_patches.append('random_patch_{}_{}_{}_{}'.format(each_top, each_left, each_top + sampling_size[0],
                                                            each_left + sampling_size[1]))
    return X_patches, y_patches, mask_patches, id_patches


def load_minhou(data_home, nr_random_sampling, sampling_size=None, patch_row=5, patch_col=5, label_ignored=0):
    """
    :param nr_random_sampling:
        None: load images without cropping.
        >0: the number of patches to be randomly sampled from each image. (`sampling_size` must be specified)
        <=0: divide each of the images into small patches. (`patch_row` and `patch_col` must be specified)
    :return:
    """
    X = list()
    y = list()
    mask = list()
    id = list()
    data_dirs = [each_file for each_file in os.listdir(data_home) if
                 os.path.isdir(join(data_home, each_file)) and each_file.startswith('minhou')]
    for idx, each_minhou_dir in enumerate(data_dirs):
        logger.info('Loading {} ... {}/{}'.format(each_minhou_dir, idx + 1, len(data_dirs)))
        cur_id = re.search(r'(minhou_patch_\d+)', each_minhou_dir).group(1)

        # read the image from disk.
        cur_image = io.imread(join(data_home, each_minhou_dir, cur_id + '.tif'))[1:-1, 1:-1]

        # read the label.
        for each_file in os.listdir(join(data_home, each_minhou_dir)):
            if each_file.startswith(cur_id + '_') and each_file.endswith('.tif'):
                cur_label = io.imread(join(data_home, each_minhou_dir, each_file))

        # get the mask
        cur_mask = (cur_label != label_ignored).astype(float)

        if not nr_random_sampling:
            X.append(cur_image)
            y.append(cur_label)
            mask.append(cur_mask)
            id.append(cur_id)

        else:
            if nr_random_sampling > 0:
                X_patches, y_patches, mask_patches, id_patches = random_sampler(cur_image, cur_label, cur_mask,
                                                                                cur_id, nr_random_sampling,
                                                                                sampling_size)
            else:
                X_patches, y_patches, mask_patches, id_patches = _minhou_cropping_helper(cur_image, cur_label, cur_mask,
                                                                                         cur_id, patch_row, patch_col)
            X.extend(X_patches)
            y.extend(y_patches)
            mask.extend(mask_patches)
            id.extend(id_patches)

    return list(map(np.array, [X, y, mask, id]))


def save_images(palette, labels, ids, output_home):
    # Convert labels to colormaps.
    if not os.path.exists(output_home):
        logger.info('{} is not existing and created.'.format(output_home))
        os.mkdir(output_home)

    colormaps = label_2_colormap(palette, labels)

    for each_colormap, each_colormap_id in zip(colormaps, ids):
        io.imsave(join(output_home, each_colormap_id + '.png'), each_colormap)
