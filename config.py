import logging
import platform
import sys
import tensorflow as tf
from palette_conversion import *

from os.path import join

# Some platform-specified parameters are specified in this file.

# remove tensorflow default handler.
tf.logging.set_verbosity(logging.INFO)
logger = logging.getLogger('tensorflow')
logger.removeHandler(logger.handlers[-1])

print('Setting logger handler and format...')
logger.setLevel(logging.INFO)
tensorflow_stream_handler = logging.StreamHandler(sys.stdout)
tensorflow_stream_handler.setFormatter(
    logging.Formatter('%(asctime)s : %(name)-12s - %(levelname)s : %(message)s'))
logger.addHandler(tensorflow_stream_handler)
logger.info('logger gotten.')

# On my PC
if platform.system() == 'Windows':
    vgg16_npy_path = r'G:\Documents\Exp Data\Models\Tensorflow\vgg16.npy'
    res50_npy_path = r'G:\Documents\Exp Data\Models\Tensorflow\Resnet50.npy'
    VOC_home = r'G:\Documents\Exp Data\PASCAL VOC\VOCdevkit\VOC2012'
    ISPRS_home = r'G:\Documents\Exp Data\data-li\segmentation-remote sensing\ISPRS\cropped'
    minhou_data_home = r'G:\Documents\Exp Data\data-li\segmentation-remote sensing\minhou_patch_十幅_171124'
    summary_home = r'G:\tmp'

# On my Mac
elif platform.system() == 'Darwin':
    pretrained_dir = r'/Volumes/Transcend/Dataset/Models'
    vgg16_npy_path = r'/Volumes/Transcend/Dataset/Models/vgg16.npy'
    res50_npy_path = r'/Volumes/Transcend/Dataset/Models/Resnet50.npy'

    ISPRS_home = '/Volumes/Transcend/Dataset/ISPRS/cropped'
    VOC_home = r'/Volumes/Transcend/Dataset/segmentation_dataset/VOC2012'
    fujian_home = '/Volumes/Transcend/Dataset/segmentation_dataset/fujian'
    summary_home = r'/Volumes/Transcend/summary/SS'

# On the server
elif platform.system() == 'Linux':
    pretrained_dir = r'/media/mass/dataset/models'
    vgg16_npy_path = r'/media/mass/dataset/models/vgg16.npy'
    res50_npy_path = r'/media/mass/dataset/models/Resnet50.npy'

    VOC_home = r'/media/mass/dataset/VOC2012'
    fujian_home = r'/media/mass/dataset/fujian'
    ISPRS_home = '/media/mass/dataset/zisprs/cropped'
    summary_home = r'/home/zxw/summary/SS'

else:
    raise Exception('npy and VOC path not specified.')

# ========================= Only need to change the wrapped around code snippet ====================
data = 'VOC'
if data == 'VOC':
    data_home = VOC_home
    datalist_home = join(VOC_home, 'ImageSets/Segmentation')
    image_home = join(VOC_home, 'JPEGImages')
    label_home = join(VOC_home, 'SegmentationClass')
    palette = VOC_palette

elif data == 'ISPRS':
    data_home = ISPRS_home
    datalist_home = join(ISPRS_home, 'datalist')
    image_home = join(ISPRS_home, 'images')
    label_home = join(ISPRS_home, 'labels')
    palette = ISPRS_palette

elif data == 'fujian':
    data_home = fujian_home
    datalist_home = join(fujian_home, 'datalist')
    image_home = join(fujian_home, 'images')
    label_home = join(fujian_home, 'labels')
    palette = fujian_palette

else:
    raise Exception('Invalid data name.')

colormap_home = join(data_home, 'colormaps')

image_ext = 'jpg'
label_ext = 'png'
datalist = 'val.txt'
# ========================= Only need to change the wrapped around code snippet ====================

VOC_image_home = join(VOC_home, 'JPEGImages')
VOC_label_home = join(VOC_home, 'SegmentationClass')
VOC_datalist_home =join(VOC_home, 'ImageSets/Segmentation')
res50_ckpt_path = join(pretrained_dir, 'resnet_v2_50_2017_04_14/resnet_v2_50.ckpt')
res101_ckpt_path = join(pretrained_dir, 'resnet_v2_101_2017_04_14/resnet_v2_101.ckpt')
vgg16_ckpt_path = join(pretrained_dir, 'vgg_16.ckpt')
vgg19_ckpt_path = None

batch_norm_decay = 0.9997
