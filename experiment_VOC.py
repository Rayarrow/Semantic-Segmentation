import os
from argparse import ArgumentParser

from data_loader import *
from models import *
from palette_conversion import VOC_palette
from training import *

seg_parser = ArgumentParser('semantic Segmentation')
seg_parser.add_argument('--model_name', type=str, default='unknown', help='Taken as ')
seg_parser.add_argument('--dump_root', type=str, default='/tmp/')
seg_parser.add_argument('--num_classes', type=int, default=22)
seg_parser.add_argument('--learning_rate', type=float, default=1e-4)
seg_parser.add_argument('--momentum', type=float, default=0.9)
seg_parser.add_argument('--nr_epoch', type=int, default=3000)
seg_parser.add_argument('--max_nr_iter', type=int, default=999999999)
seg_parser.add_argument('--val_size', type=int, default=999999999)
seg_parser.add_argument('--batch_size', type=int, default=1)
seg_parser.add_argument('--is_train', action='store_true')
seg_parser.add_argument('--is_predict', action='store_true')
seg_parser.add_argument('--FCN_stride', type=int, default=32)
seg_parser.add_argument('--resize_shape', default=[224, 224])
seg_parser.add_argument('--get_FCN', type=int, default=1)
seg_parser.add_argument('--devices', type=str, default='0')
seg_parser.add_argument('--val_interval', type=int, default=50)
seg_parser.add_argument('--epoch_checkpoint', type=int, default=100)
args = seg_parser.parse_args()

print(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

# some hyper parameters.
dump_root = args.dump_root
model_name = args.model_name
num_classes = args.num_classes
learning_rate = args.learning_rate
momentum = args.momentum
nr_epoch = args.nr_epoch
max_nr_iter = args.max_nr_iter
val_size = args.val_size
batch_size = args.batch_size
is_train = args.is_train
is_predict = args.is_predict
resize_shape = args.resize_shape
FCN_stride = args.FCN_stride
get_FCN = args.get_FCN
val_interval = args.val_interval
epoch_checkpoint = args.epoch_checkpoint

# The path where the VOC dataset is located.

# The place to which the Tensorflow write summaries.
# VOC_dump_home = r'G:\tmp\FCN{}s_VOC_full_summary'.format(FCN_stride)
VOC_dump_home = join(dump_root, model_name)

if not os.path.exists(VOC_dump_home):
    os.makedirs(VOC_dump_home)

with open(join(VOC_dump_home, 'parameters.txt'), 'w') as f:
    f.write('\n'.join(args.__str__()[10:-1].split(', ')))

# model = FCN(FRONT_VGG16, FCN_stride, num_classes, get_FCN=get_FCN)
# X_train, y_train, y_train_mask, id_train, X_val, y_val, y_val_mask, id_val = load_VOC(VOC_home, 21)

model = PSPNet(FRONT_RES50, num_classes, 473, 473, get_FCN=get_FCN)
X_train, y_train, y_train_mask, id_train, X_val, y_val, y_val_mask, id_val = load_VOC(VOC_home, 21, (473, 473))

# if is_train:
commission_training_task(model, VOC_dump_home, X_train, y_train, y_train_mask, X_val, y_val, y_val_mask, learning_rate,
                         momentum, nr_epoch, batch_size, max_nr_iter=max_nr_iter, val_size=val_size,
                         val_interval=val_interval, epoch_checkpoint=epoch_checkpoint)

if is_predict:
    y_pred = commission_predict(model, None, VOC_dump_home, X_val)
    save_images(VOC_palette, y_pred, id_val, join(VOC_dump_home, 'output'))


