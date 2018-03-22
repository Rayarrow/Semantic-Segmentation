from argparse import ArgumentParser
from pprint import pprint

from data_loader import *
from models import *
from palette_conversion import VOC_palette
from training import *

seg_parser = ArgumentParser('semantic Segmentation')
# IO
seg_parser.add_argument('--model_name', type=str, default='unknown', help='Taken as ')
seg_parser.add_argument('--dump_root', type=str, default='/tmp/')

# Model
seg_parser.add_argument('--num_classes', type=int, default=21)
seg_parser.add_argument('--get_FCN', type=int, default=1)
seg_parser.add_argument('--FCN_stride', type=int, default=32)
seg_parser.add_argument('--resize_shape', default=224)

# Learning control
seg_parser.add_argument('--learning_rate', type=float, default=1e-4)
seg_parser.add_argument('--lr_decay', type=str, default='poly')
seg_parser.add_argument('--momentum', type=float, default=0.9)
seg_parser.add_argument('--weight_decay', type=float, default=1e-4)
seg_parser.add_argument('--nr_iter', type=int, default=130000)
seg_parser.add_argument('--batch_size', type=int, default=4)
seg_parser.add_argument('--is_train', action='store_true')
seg_parser.add_argument('--is_predict', action='store_true')
seg_parser.add_argument('--devices', type=str, default='0')
seg_parser.add_argument('--report_interval', type=int, default=10)
seg_parser.add_argument('--val_interval', type=int, default=1000)
seg_parser.add_argument('--iter_ckpt_interval', type=int, default=2000)
args = seg_parser.parse_args()

pprint(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

# some hyper parameters.
dump_root = args.dump_root
model_name = args.model_name
num_classes = args.num_classes
learning_rate = args.learning_rate
momentum = args.momentum
nr_iter = args.nr_iter
batch_size = args.batch_size
is_train = args.is_train
is_predict = args.is_predict
resize_shape = args.resize_shape
FCN_stride = args.FCN_stride
get_FCN = args.get_FCN
report_interval = args.report_interval
val_interval = args.val_interval
iter_ckpt_interval = args.iter_ckpt_interval
weight_decay = args.weight_decay
lr_decay = args.lr_decay

# The path where the VOC dataset is located.

# The place to which the Tensorflow write summaries.
# VOC_dump_home = r'G:\tmp\FCN{}s_VOC_full_summary'.format(FCN_stride)
VOC_dump_home = join(dump_root, model_name)

if not os.path.exists(VOC_dump_home):
    os.makedirs(VOC_dump_home)

with open(join(VOC_dump_home, 'parameters.txt'), 'w') as f:
    f.write('\n'.join(args.__str__()[10:-1].split(', ')))

front_end = ResNet50(473, 473, 3, get_FCN, weight_decay)
model = PSPNet(front_end, num_classes)
d_train, d_val = load_VOC(VOC_home, 21, resize_shape, num_train=20, num_val=10, data_set=False)

if is_train:
    commission_training_task(model, VOC_dump_home, d_train, d_val, learning_rate, lr_decay, momentum, nr_iter, batch_size,
                             report_interval, val_interval, iter_ckpt_interval)

if is_predict:
    X_val, y_val, y_mask_val, id_val = d_val
    y_pred, acc, mIoU = commission_predict(model, None, VOC_dump_home, X_val, y_val, y_mask_val)
    save_pred_results(VOC_palette, y_pred, id_val, join(VOC_dump_home, 'output_{}@{}'.format(acc, mIoU)))
