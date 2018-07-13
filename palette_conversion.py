import os

import numpy as np
from skimage import io
from tqdm import tqdm
from config import *

VOC_colormap_label = [
                         [[0, 0, 0], 'background'],  # 0
                         [[128, 0, 0], 'aero plane'],  # 1
                         [[0, 128, 0], 'bicycle'],  # 2
                         [[128, 128, 0], 'bird'],  # 3
                         [[0, 0, 128], 'boat'],  # 4
                         [[128, 0, 128], 'bottle'],  # 5
                         [[0, 128, 128], 'bus'],  # 6
                         [[128, 128, 128], 'car'],  # 7
                         [[64, 0, 0], 'cat'],  # 8
                         [[192, 0, 0], 'chair'],  # 9
                         [[64, 128, 0], 'cow'],  # 10
                         [[192, 128, 0], 'dining-table'],  # 11
                         [[64, 0, 128], 'dog'],  # 12
                         [[192, 0, 128], 'horse'],  # 13
                         [[64, 128, 128], 'motorbike'],  # 14
                         [[192, 128, 128], 'person'],  # 15
                         [[0, 64, 0], 'potted-plant'],  # 16
                         [[128, 64, 0], 'sheep'],  # 17
                         [[0, 192, 0], 'sofa'],  # 18
                         [[128, 192, 0], 'train'],  # 19
                         [[0, 64, 128], 'tv/monitor']  # 20
                     ] + [[0, 'none']] * (256 - 22) + [[[224, 224, 192], 'void']]

VOC_palette, VOC_label = zip(*VOC_colormap_label)

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

ISPRS_palette = [[255, 255, 255],
                 [0, 0, 255],
                 [0, 255, 255],
                 [0, 255, 0],
                 [255, 255, 0],
                 [255, 0, 0]]

fujian_palette = [[0, 0, 0],
                  [140, 255, 90],
                  [150, 255, 100],
                  [160, 255, 110],
                  [255, 255, 200],
                  [200, 50, 255],
                  [92, 92, 92],
                  [200, 200, 200],
                  [229, 229, 229],
                  [180, 180, 180],
                  [100, 200, 250],
                  ]


# This function converts from segmentation class colormap images to labeled images.
def colormap_2_label(palette, colormaps, ids):
    labels = []
    pixel2label = dict(zip(map(tuple, palette), range(len(palette))))

    # Convert raw image to label image.
    for (each_colormap, each_id) in zip(colormaps, ids):
        logger.info(f'colormap to label: {each_id}')
        cur_label = np.zeros(each_colormap.shape[:2], dtype=np.int32)
        for i in range(each_colormap.shape[0]):
            for j in range(each_colormap.shape[1]):
                cur_label[i, j] = pixel2label[tuple(each_colormap[i, j])]
        labels.append(cur_label)

    return labels


def label_2_colormap(palette, labels):
    colormaps = []
    for idx, each_label in enumerate(labels):
        logger.info('label to colormap {} / {}'.format(idx + 1, len(labels)))
        this_colormap = np.zeros((each_label.shape[0], each_label.shape[1], 3))
        for i in range(each_label.shape[0]):
            for j in range(each_label.shape[1]):
                this_colormap[i, j] = palette[each_label[i, j]]

        colormaps.append(this_colormap.astype('uint8'))

    return colormaps
