import logging
import sys
import platform

# Some platform-specified parameters are specified in this file.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger('VGGFCN32')

# On my PC
if platform.system() == 'Windows':
    vgg16_npy_path = r'G:\Documents\Exp Data\Models\Tensorflow\vgg16.npy'
    res50_npy_path = r'G:\Documents\Exp Data\Models\Tensorflow\Resnet50.npy'
    VOC_home = r'G:\Documents\Exp Data\PASCAL VOC\VOCdevkit\VOC2012'
    minhou_data_home = r'G:\Documents\Exp Data\data-li\segmentation-remote sensing\minhou_patch_十幅_171124'

# On my Mac
elif platform.system() == 'Darwin':
    vgg16_npy_path = r'/Volumes/Transcend/Dataset/Models/vgg16.npy'
    res50_npy_path = r'/Volumes/Transcend/Dataset/Models/Resnet50.npy'
    VOC_home = r'/Volumes/Transcend/Dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'

# On the server
elif platform.system() == 'Linux':

    vgg16_npy_path = r'/media/mass/dataset/models/vgg16.npy'
    res50_npy_path = r'/media/mass/dataset/models/Resnet50.npy'
    VOC_home = r'/media/mass/dataset/VOC2012'

else:
    raise Exception('npy and VOC path not specified.')


