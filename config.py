import logging
import platform
import sys
from os.path import join

import numpy as np
import tensorflow as tf

# =========================== DEFINE PLATFORM INDEPENDENT CONSTANT  ===========================

deeplab_batch_norm_decay = 0.9997
darknet_batch_norm_decay = 0.9

RGB_MEAN_1 = np.array([[[[123.68, 116.78, 103.94]]]], float)

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
]

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

Vinhingen_palette = [[255, 255, 255],
                     [0, 0, 255],
                     [0, 255, 255],
                     [0, 255, 0],
                     [255, 255, 0],
                     [255, 0, 0]]

fujian_palette = [
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

TSUNAMI_palette = [[255, 255, 255],
                   [0, 0, 0]]

# --------------------------- DEFINE PLATFORM INDEPENDENT CONSTANT  ---------------------------


# Some platform-specified parameters are specified in this file.

# remove tensorflow default handler.

tf.logging.set_verbosity(logging.INFO)
logger = logging.getLogger('tensorflow')
logger.removeHandler(logger.handlers[-1])

print('Setting logger handler and format...')
logger.setLevel(logging.INFO)
common_formater = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')

tensorflow_stream_handler = logging.StreamHandler(sys.stdout)
tensorflow_stream_handler.setFormatter(common_formater)
logger.addHandler(tensorflow_stream_handler)
logger.info('logger gotten.')

# On my PC
if platform.node() == 'Rayarrow-S-PC':
    dataset_home = r'D:\Datasets'
    summary_home = r'D:\tmp'

# On my lab-PC
elif platform.node() == 'Rayarrow-Lab-PC':
    dataset_home = 'D:\Datasets'
    summary_home = r'D:\tmp'

# On the server
elif platform.node() == 'bigdata':
    dataset_home = '/media/mass/dataset'
    summary_home = '/media/mass/dataset/zxw_temp/SS'

else:
    raise Exception('invalid platform.node().')

CV_dataset_home = join(dataset_home, 'CV')

semantic_segmentation_home = join(CV_dataset_home, 'semantic_segmentation')
VOC_home = join(semantic_segmentation_home, 'VOC2012')
Vaihingen_home = join(semantic_segmentation_home, 'Vaihingen')
fujian_home = join(semantic_segmentation_home, 'fujian20180924')

change_detection_home = join(CV_dataset_home, 'change_detection')
TSUNAMI_home = join(change_detection_home, 'TSUNAMI')
GSV_home = join(change_detection_home, 'GSV')

pretrained_dir = join(dataset_home, 'models')

res50_npy_path = join(pretrained_dir, 'xx.npy')
vgg16_npy_path = join(pretrained_dir, 'vgg16.npy')
res50_ckpt_path = join(pretrained_dir, 'resnet_v2_50_2017_04_14/resnet_v2_50.ckpt')
res101_ckpt_path = join(pretrained_dir, 'resnet_v2_101_2017_04_14/resnet_v2_101.ckpt')
vgg16_ckpt_path = join(pretrained_dir, 'vgg_16.ckpt')
vgg19_ckpt_path = None

# ========================= Only need to change the wrapped around code snippet ====================

data_home_bundle = {
    'VOC': [VOC_home, 'images', 'labels', 'datalist', VOC_palette],
    'Vaihingen': [Vaihingen_home, 'images', 'labels', 'datalist', Vinhingen_palette],
    'fujian': [fujian_home, 'images', 'labels', 'datalist', fujian_palette],
    'TSUNAMI': [TSUNAMI_home, 'images', 'labels', 'datalist', TSUNAMI_palette],
    'GSV': [GSV_home, 'images', 'labels', 'datalist', TSUNAMI_palette],
}

# ========================= Only need to change the wrapped around code snippet ====================

