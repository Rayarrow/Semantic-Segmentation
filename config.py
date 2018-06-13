import logging
import platform
import sys
import tensorflow as tf
from tqdm import tqdm

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
    vgg16_npy_path = r'/Volumes/Transcend/Dataset/Models/vgg16.npy'
    res50_npy_path = r'/Volumes/Transcend/Dataset/Models/Resnet50.npy'
    ISPRS_home = r'/Volumes/Transcend/Dataset/ISPRS/ISPRS'
    VOC_home = r'/Volumes/Transcend/Dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    summary_home = r'/Volumes/Transcend/summary/SS'

# On the server
elif platform.system() == 'Linux':
    vgg16_npy_path = r'/media/mass/dataset/models/vgg16.npy'
    res50_npy_path = r'/media/mass/dataset/models/Resnet50.npy'
    VOC_home = r'/media/mass/dataset/VOC2012'
    summary_home = r'/home/zxw/summary/SS'

else:
    raise Exception('npy and VOC path not specified.')

VOC_image_home = join(VOC_home, 'JPEGImages')
VOC_colormap_home = join(VOC_home, 'SegmentationClass')
VOC_label_home = join(VOC_home, 'SegmentationClassLabelImages')
