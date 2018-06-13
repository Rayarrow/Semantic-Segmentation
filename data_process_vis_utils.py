import os
import re

import numpy as np
import skimage.io as io
from os.path import join

from config import *


def load_VOC_pattern_image(data_home, datalist, image_path='JPEGImages', datalist_path='datalist', ext='jpg'):
    with open(join(data_home, datalist_path, datalist)) as f:
        ids = f.read().split()
    images = []
    for idx, each_id in enumerate(ids):
        logger.info(f'loading image {each_id} from {join(data_home, image_path)}... {idx}/{len(ids)}')
        images.append(io.imread(join(data_home, image_path, '.'.join([each_id, ext]))))
    return images, ids


def maybe_create_dir(path):
    if not os.path.exists(path):
        logger.info(f'{path} does not exist and is being created...')
        os.makedirs(path)


def get_masked_images(images, colormaps, alpha=0.5):
    assert len(images) == len(colormaps)
    masked = []
    for idx, (each_image, each_label) in enumerate(zip(images, colormaps)):
        logger.info(f'masking {idx+1}/{len(images)}...')
        this_masked = (each_image * (1 - alpha) + each_label * alpha).astype('uint8')
        masked.append(this_masked)
    return masked


def save_images(output_dir, images, ids, ext='jpg'):
    maybe_create_dir(output_dir)
    for each_id, each_image in tqdm(zip(ids, images)):
        logger.info(f'saving {each_id} to {output_dir}...')
        io.imsave(join(output_dir, '.'.join([each_id, ext])), each_image)


def get_comparison(image_list):
    logger.info('Getting comparison images...')
    comparison = []
    for each_row in tqdm(zip(*image_list)):
        this_comparison = np.concatenate(each_row, axis=1)
        comparison.append(this_comparison)
    return comparison


def get_comparison_from_files(data_home, datalist, image_paths, exts):
    image_list = []
    for each_image_path, each_ext in tqdm(zip(image_paths, exts)):
        cur_list, _ = load_VOC_pattern_image(data_home, datalist, each_image_path, ext=each_ext)
        image_list.append(cur_list)
    return get_comparison(image_list)


def init_nest_list(n):
    res = []
    for i in range(n):
        res.append([])
    return res


def grid_cropping_helper(images, id, patch_row, patch_col):
    """
    A helper function to crop a images into small patches.
    """
    res = init_nest_list(len(images))
    id_patch = []

    # Calculate the number of rows and columns of the image grid to be cropped.
    image_example = images[0]
    print(image_example.shape)
    patch_height, patch_width = image_example.shape[0] / patch_row, image_example.shape[1] / patch_col
    for i in range(patch_row):
        for j in range(patch_col):
            top = int(patch_height * i)
            bottom = int(patch_height * (i + 1))
            left = int(patch_width * j)
            right = int(patch_width * (j + 1))

            for idx, each_image in enumerate(images):
                res[idx].append(each_image[top:bottom, left:right])
            id_patch.append('{}_{}_{}'.format(id, i, j))

    return res, id_patch


def stitch_patches(data_home, datalist, rows, cols, image_path='JPEGImages', datalist_path='datalist', ext='png'):
    image_home = os.path.join(data_home, image_path)
    if isinstance(datalist, list):
        ids = datalist
    else:
        with open(join(data_home, datalist_path, datalist)) as f:
            ids = f.read().split()

    # strip patch index suffix.
    ids = sorted(list(set(sorted([re.search(r'(.*?)_\d+_\d+', each_id).group(1) for each_id in ids]))))

    complete_images = []
    for idx, each_id in enumerate(ids):
        row_patches = []
        for each_row in range(rows):
            logger.info(f'stitching {each_id}_{each_row}/{rows}... {idx}/{len(ids)}')
            row_patches.append(np.concatenate(
                [io.imread(join(image_home, f'{each_id}_{each_row}_{each_col}.{ext}')) for each_col in range(cols)],
                axis=1))
        complete_image = np.concatenate(row_patches)
        complete_images.append(complete_image)
    return complete_images, ids


def grid_cropping_bunch_sampler(image_nlist, ids, patch_row, patch_col):
    """
    Randomly get `nr` sample patches measuring `sampling_size` from each of the images.
    """
    res = init_nest_list(len(image_nlist))
    id_patch = []

    logger.info('random bunch sampling...')
    for *each_list, each_id in zip(*image_nlist, ids):
        cur_patch_nlist, cur_id_list = grid_cropping_helper(each_list, each_id, patch_row, patch_col)
        for idx, each_patch_list in enumerate(cur_patch_nlist):
            res[idx].extend(each_patch_list)
        id_patch.extend(cur_id_list)

    return res, id_patch


def random_bunch_sampler(image_nlist, ids, nr, sampling_size):
    """
    Randomly get `nr` sample patches measuring `sampling_size` from each of the images.
    """
    if isinstance(sampling_size, int):
        sampling_size = (sampling_size, sampling_size)
    res = init_nest_list(len(image_nlist))
    id_patch = []

    logger.info('random bunch sampling...')
    for *each_list, each_id in zip(*image_nlist, ids):
        cur_patch_nlist, cur_id_list = random_sampler(each_list, each_id, nr, sampling_size)
        for idx, each_patch_list in enumerate(cur_patch_nlist):
            res[idx].extend(each_patch_list)
        id_patch.extend(cur_id_list)

    return res, id_patch


def random_sampler(images, id, nr, sampling_size):
    """
    Randomly get `nr` sample patches from the given image.
    """
    if isinstance(sampling_size, int):
        sampling_size = (sampling_size, sampling_size)

    res = init_nest_list(len(images))
    id_patches = []

    example_image = images[0]

    height, width = example_image.shape[:2]
    tops = np.random.randint(0, height - sampling_size[0], nr)
    lefts = np.random.randint(0, width - sampling_size[1], nr)
    for idx, (each_top, each_left) in enumerate(zip(tops, lefts)):
        if idx % (nr // 10) == 0:
            logger.info('random sampling image {}...  {}/{}'.format(id, idx + 1, nr))

        for idx, each_image in enumerate(images):
            res[idx].append(each_image[each_top: each_top + sampling_size[0], each_left:each_left + sampling_size[1]])
        id_patches.append('{}_random_{}_{}'.format(id, each_top, each_left))
    return res, id_patches


