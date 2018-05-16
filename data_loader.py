import tensorflow as tf
import os
import re
from os.path import join
from collections import Iterable
import random

import numpy as np
from skimage import io, transform
from PIL import Image

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
        y_weights.append((cur_label != label_ignored))
        ids.append(each_id)
    return X, y, y_weights, ids


def load_VOC(data_home, label_ignored=255, resize_shape=None, load_train=True, load_val=True, data_set=True,
             num_train=None, num_val=None):
    # Define the absolute path of the sub directories.
    resized = False
    dataset_path = join(data_home, 'JPEGImages')
    label_path = join(data_home, 'SegmentationClassLabelImages')
    if resize_shape:
        dataset_path += '_{}'.format(473)
        label_path += '_{}'.format(473)
        resized = True
    train_idx_path = r'ImageSets/Segmentation/train.txt'
    val_idx_path = r'ImageSets/Segmentation/val.txt'

    # load the ids of validation images.
    with open(join(data_home, train_idx_path)) as f:
        train_ids = f.read().split()[:num_train]

    with open(join(data_home, val_idx_path)) as f:
        val_ids = f.read().split()[:num_val]

    X_train = y_train = y_train_mask = id_train = X_val = y_val = y_val_mask = id_val = []
    if load_train:
        logger.info('Loading training data...')
        X_train, y_train, y_train_mask, id_train = _load_dataset_VOC(data_home, dataset_path, label_path, train_ids,
                                                                     label_ignored)
    if load_val:
        logger.info('Loading validation data...')
        X_val, y_val, y_val_mask, id_val = _load_dataset_VOC(data_home, dataset_path, label_path, val_ids,
                                                             label_ignored)

    # Resize the images to the specified shape.
    if not resized and resize_shape:
        if isinstance(resize_shape, int):
            resize_shape = (resize_shape, resize_shape)
        logger.info('cached resized images of the shape you providied do not exist.')
        if load_train:
            X_train = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in X_train],
                               dtype=np.uint8)
            y_train = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in y_train],
                               dtype=np.uint8)
            y_train_mask = np.array(
                [transform.resize(image, resize_shape, preserve_range=True) for image in y_train_mask], dtype=np.uint8)

        if load_val:
            X_val = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in X_val],
                             dtype=np.uint8)
            y_val = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in y_val],
                             dtype=np.uint8)
            y_val_mask = np.array([transform.resize(image, resize_shape, preserve_range=True) for image in y_val_mask],
                                  dtype=np.uint8)

    if data_set:
        logger.info('Constructing dataset...')
        return tf.data.Dataset.from_tensor_slices(tuple(map(np.array, [X_train, y_train, y_train_mask, train_ids]))), \
               tf.data.Dataset.from_tensor_slices(tuple(map(np.array, [X_val, y_val, y_val_mask, val_ids])))
    else:
        return list(map(np.array, [X_train, y_train, y_train_mask, train_ids])), \
               list(map(np.array, [X_val, y_val, y_val_mask, val_ids]))


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
        cur_mask = (cur_label != label_ignored)

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

def load_isprs_train(data_home, nr_random_sampling = 1000, sampling_size = [473, 473], patch_row=10, patch_col=10, label_ignored=0):
    '''load isprs train data
    '''
    X = list()
    y = list()
    mask = list()
    id = list()
    data_dirs = [each_file for each_file in os.listdir(data_home) if
                 os.path.isdir(join(data_home, each_file)) and each_file.startswith('minhou')]
    dataset_path = join(data_home, 'top')
    label_path = join(data_home, 'gt_caffe')
    train_idx_path = r'/home/zxw/lex/FCNforISPRS/list_trainval_path.txt'

    # load the ids of validation images.
    with open(train_idx_path) as f:
        train_ids = f.read().split()

    for idx, each_id in enumerate(train_ids):
        if idx % 1 == 0:
            logger.info('loadding data {}/{}'.format(idx, len(train_ids)))
        cur_data = io.imread(join(data_home, dataset_path, each_id+ '.tif' ))
        cur_label = io.imread(join(data_home, label_path, each_id + '.png'))
        cur_id = each_id
        

        # get the mask
        cur_mask = (cur_label != label_ignored).astype(float)

        if not nr_random_sampling:
            X.append(cur_data)
            y.append(cur_label)
            mask.append(cur_mask)
            id.append(cur_id)

        else:
            if nr_random_sampling > 0:
                X_patches, y_patches, mask_patches, id_patches = random_sampler(cur_data, cur_label, cur_mask,
                                                                                cur_id, nr_random_sampling,
                                                                                sampling_size)
            else:
                X_patches, y_patches, mask_patches, id_patches = _minhou_cropping_helper(cur_data, cur_label, cur_mask,
                                                                                         cur_id, patch_row, patch_col)
            X.extend(X_patches)
            y.extend(y_patches)
            mask.extend(mask_patches)
            id.extend(id_patches)

    return list(map(np.array, [X, y, mask, id]))

def load_isprs_val(data_home, nr_random_sampling = 1000, sampling_size = [473, 473], patch_row=10, patch_col=10, label_ignored=0):
    '''load isprs val data
    '''
    X = list()
    y = list()
    mask = list()
    id = list()
    data_dirs = [each_file for each_file in os.listdir(data_home) if
                 os.path.isdir(join(data_home, each_file)) and each_file.startswith('minhou')]
    dataset_path = join(data_home, 'top')
    label_path = join(data_home, 'gt_caffe')

    val_idx_path = r'/home/zxw/lex/FCNforISPRS/list_val_path.txt'

    # load the ids of validation images.
    with open(val_idx_path) as f:
        val_ids = f.read().split()
    for idx, each_id in enumerate(val_ids):
        if idx % 1 == 0:
            logger.info('loadding data {}/{}'.format(idx, len(val_ids)))
        cur_data = io.imread(join(data_home, dataset_path, each_id+ '.tif' ))
        cur_label = io.imread(join(data_home, label_path, each_id + '.png'))
        cur_id = each_id
        

        # get the mask
        cur_mask = (cur_label != label_ignored).astype(float)

        if not nr_random_sampling:
            X.append(cur_data)
            y.append(cur_label)
            mask.append(cur_mask)
            id.append(cur_id)

        else:
            if nr_random_sampling > 0:
                X_patches, y_patches, mask_patches, id_patches = random_sampler(cur_data, cur_label, cur_mask,
                                                                                cur_id, nr_random_sampling,
                                                                                sampling_size)
            else:
                X_patches, y_patches, mask_patches, id_patches = _minhou_cropping_helper(cur_data, cur_label, cur_mask,
                                                                                         cur_id, patch_row, patch_col)
            X.extend(X_patches)
            y.extend(y_patches)
            mask.extend(mask_patches)
            id.extend(id_patches)

    return list(map(np.array, [X, y, mask, id]))



def save_pred_results(palette, labels, ids, output_home):
    # Convert labels to colormaps.
    if not os.path.exists(output_home):
        logger.info('{} is not existing and created.'.format(output_home))
        os.mkdir(output_home)

    colormaps = label_2_colormap(palette, labels)

    for each_colormap, each_colormap_id in zip(colormaps, ids):
        io.imsave(join(output_home, each_colormap_id + '.png'), each_colormap)


def tf_da(img, label, resize_shape, scale = True, flip= True, rotate= True):
    if scale:
        img, label = tf_random_scale(img, label)

    if flip:
        img, label = tf_random_flip(img, label)

    #if rotate:
        #img, label = tf_random_rotate(img, label)

    img, label = random_pad_and_crop(img, label, resize_shape)


    return img, label

def tf_random_scale(img, label):
    scale = tf.random_uniform([1], minval=0.5, maxval=2, dtype=tf.float32, seed=None)
    h_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[1]), scale))
    w_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(img)[2]), scale))
    new_shape = tf.squeeze(tf.stack([h_new, w_new]), squeeze_dims=[1])
    img = tf.image.resize_images(img, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 3), new_shape)

    return img, label

def tf_random_flip(img, label):
    distort_left_right_random = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)
    mirror = tf.less(tf.stack([1.0, distort_left_right_random[0], distort_left_right_random[1]]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    img = tf.reverse(img, mirror)
    label = tf.reverse(label, mirror)
    return img, label

#def tf_random_rotate(img, label):
    #return img, label

def random_pad_and_crop(img, label, resize_shape):
    """
    Randomly crop and pads the input images.

    Args:
      image: Training image to crop/ pad.
      label: Segmentation mask to crop/ pad.
      crop_h: Height of cropped segment.
      crop_w: Width of cropped segment.
      ignore_label: Label to ignore during the training.
    """

    label = tf.cast(label, dtype=tf.float32)
    #if image_shape is smaller than resize_shape, do padding
    crop_h = resize_shape[0]
    crop_w = resize_shape[1]
    combined = tf.concat(axis=3, values=[img, label])
    image_shape = tf.shape(img)
    if image_shape[1] != crop_h:
        h = tf.maximum(crop_h, image_shape[1])
        w = tf.maximum(crop_w, image_shape[2])
        combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, h, w)
    else:
        combined_pad = combined

    #crop images and labels by resize_shape
    batch_num = []
    for i in range(image_shape[0]):
        batch_num.append(1)
    combined_split = tf.split(combined_pad, batch_num, name = 'combined_split')
    split = []
    for i in range(len(combined_split)):
        split.append(tf.squeeze(combined_split[i], 0))
    for i in range(len(split)):
        split[i] = tf.random_crop(split[i], [crop_h, crop_w, 4])
    for i in range(len(split)):
        split[i] = tf.expand_dims(split[i], 0, name = 'expand_dims')
    if len(split)>1:
        for i in range(1, len(split)):
            combined_crop = tf.concat([split[0], split[i]], 0, name = 'concate')
            split[0] = combined_crop
    else:
        combined_crop = split[0]

    img_crop = combined_crop[:, :, :, :3]
    label_crop = combined_crop[:, :, :, 3:]
    label_crop = tf.cast(label_crop, dtype=tf.uint8)
    label_crop = tf.squeeze(label_crop, 3)


    return img_crop, label_crop