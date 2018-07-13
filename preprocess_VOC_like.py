from pathlib import Path
from argparse import ArgumentParser

import numpy as np
from palette_conversion import *

from data_loader import *
from skimage import io
import os
from config import *
from collections import Counter

np.random.seed(2333)


# arg_parser = ArgumentParser()
# arg_parser.add_argument('--input')
# arg_parser.add_argument('--output')
# args = arg_parser.parse_args()


def verify_pixels(data_home, image_dir='colormaps', datalist='trval.txt', datalist_path='datalist'):
    """
    Count the number of possible pixel values in a single colormap (or image).
    Mainly to detect if the 'png' is compressed in a lossy way in which noise pixels are introduced (bilinear).
    """
    label_path = join(data_home, image_dir)
    datalist_path = join(data_home, datalist_path, datalist)
    with open(datalist_path) as f:
        ids = f.read().split()

    for each_id in ids:
        pixels = []
        cur_colormap = io.imread(join(label_path, each_id) + '.png')

        for i in range(cur_colormap.shape[0]):
            for j in range(cur_colormap.shape[1]):
                pixels.append(tuple(cur_colormap[i, j, :]))

        cur_counter = Counter(pixels)
        logger.info(f'{each_id} --- number of categories: {len(cur_counter)}, {cur_counter}')


def crop_and_save(data_home, image_dir, label_dir, datalist_dir, datalist, image_ext='jpg', label_ext='png'):
    image_path = join(data_home, image_dir)
    label_path = join(data_home, label_dir)
    datalist_path = join(data_home, datalist_dir, datalist)
    images, ids = load_VOC_pattern_image(image_path, datalist_path, image_ext)
    labels, _ = load_VOC_pattern_image(label_path, datalist_path, label_ext)
    (image_patches, label_patches), id_patches = random_bunch_sampler([images, labels], ids, 50, 600)
    save_images(join(data_home, 'image_patches'), image_patches, id_patches)
    save_images(join(data_home, 'label_patches'), label_patches, id_patches, label_ext)

    with open(join(data_home, datalist_dir, f'{datalist.split(".")[0]}_patches.txt'), 'w', encoding='utf8') as f:
        f.write('\n'.join(id_patches) + '\n')


crop_and_save(fujian_home, 'images', 'labels', 'datalist', 'train.txt')


def save_colormaps_to_labels(datahome):
    colormap_path = join(datahome, 'colormaps')
    datalist_path = join(datahome, 'datalist', 'trval.txt')
    colormaps, ids = load_VOC_pattern_image(colormap_path, datalist_path, ext='png')
    labels = colormap_2_label(fujian_palette, colormaps, ids)
    save_images(join(datahome, 'labels'), labels, ids, ext='png')

# save_colormaps_to_labels(fujian_home)
