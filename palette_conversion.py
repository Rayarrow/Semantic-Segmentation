import os
from os.path import join

import numpy as np
from skimage import io

from config import *
from tqdm import tqdm

VOC_palette = [[0, 0, 0],
               [128, 0, 0],
               [0, 128, 0],
               [128, 128, 0],
               [0, 0, 128],
               [128, 0, 128],
               [0, 128, 128],
               [128, 128, 128],
               [64, 0, 0],
               [192, 0, 0],
               [64, 128, 0],
               [192, 128, 0],
               [64, 0, 128],
               [192, 0, 128],
               [64, 128, 128],
               [192, 128, 128],
               [0, 64, 0],
               [128, 64, 0],
               [0, 192, 0],
               [128, 192, 0],
               [0, 64, 128],
               [224, 224, 192]]

VOC_labels = ['background',
              'aero plane',
              'bicycle',
              'bird',
              'boat',
              'bottle',
              'bus',
              'car',
              'cat',
              'chair',
              'cow',
              'dining-table',
              'dog',
              'horse',
              'motorbike',
              'person',
              'potted-plant',
              'sheep',
              'sofa',
              'train',
              'tv/monitor',
              'void']

minhou_palette = [[0, 0, 0],
                  [100, 160, 0],
                  [100, 120, 60],
                  [120, 200, 100],
                  [255, 255, 255],
                  [178, 178, 178],
                  [130, 130, 130],
                  [245, 162, 122],
                  [245, 185, 122],
                  [150, 220, 240],
                  [255, 255, 190]]

VOC_data_home = r'G:\Documents\Exp Data\PASCAL VOC\VOCdevkit\VOC2012'
VOC_colormap_home = join(VOC_data_home, 'SegmentationClass')
VOC_label_home = join(VOC_data_home, 'SegmentationClassLabelImages')


# This function converts from segmentation class colormap images to labeled images.
def colormap_2_label(palette, colormap_home):
    label_home = colormap_home + 'Labels'
    if not os.path.exists(label_home):
        os.mkdir(label_home)
    colormap_images = list()
    # load all colormap
    for each_colormap_image_file in os.listdir(colormap_home):
        colormap_images.append((each_colormap_image_file, io.imread(join(colormap_home, each_colormap_image_file))))

    pixel2label = dict(zip(map(tuple, palette), range(len(palette))))

    # Convert raw image to label image.
    for (each_colormap_image_name, each_colormap_image_file) in tqdm(colormap_images):
        # if colormap_image_idx % 100 == 0:
        #     logger.info('converting to label image {}/{}'.format(colormap_image_idx, len(colormap_images)))
        cur_label_image = np.zeros(each_colormap_image_file.shape[:2], dtype=np.int32)
        for i in range(each_colormap_image_file.shape[0]):
            for j in range(each_colormap_image_file.shape[1]):
                cur_label_image[i, j] = pixel2label[tuple(each_colormap_image_file[i, j])]
        io.imsave(join(label_home, each_colormap_image_name), cur_label_image)


def label_2_colormap(palette, labels):
    colormaps = []
    for idx, each_label in enumerate(labels):
        if idx % 100 == 0:
            logger.info('label to colormap {} / {}'.format(idx + 1, len(labels)))

        this_colormap = np.zeros((each_label.shape[0], each_label.shape[1], 3))
        for i in range(each_label.shape[0]):
            for j in range(each_label.shape[1]):
                this_colormap[i, j] = palette[each_label[i, j]]

        colormaps.append(this_colormap.astype(int))

    return colormaps


