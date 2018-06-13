from argparse import ArgumentParser

from estimators import FCN_estimator
from models import *
from palette_conversion import *

seg_parser = ArgumentParser('semantic Segmentation')
# IO
seg_parser.add_argument('--model_dir', type=str, default=summary_home)

# Model
seg_parser.add_argument('--num_classes', type=int, default=21)
seg_parser.add_argument('--ignore_label', type=int, default=255)
seg_parser.add_argument('--crop_size', type=int, default=513)
seg_parser.add_argument('--get_FCN', type=int, default=1)
seg_parser.add_argument('--FCN_stride', type=int, default=32)
seg_parser.add_argument('--resize_shape', default=473)

# Learning control
seg_parser.add_argument('--learning_rate', type=float, default=1e-4)
seg_parser.add_argument('--end_learning_rate', type=float, default=1e-6)
seg_parser.add_argument('--momentum', type=float, default=0.9)
seg_parser.add_argument('--lr_decay', type=str, default='poly')
seg_parser.add_argument('--decay_step', type=int, default=None)
seg_parser.add_argument('--weight_decay', type=float, default=5e-4)

seg_parser.add_argument('--epochs', type=int, default=30)
seg_parser.add_argument('--batch_size', type=int, default=4)
seg_parser.add_argument('--devices', type=str, default='0')
seg_parser.add_argument('--train_file', type=str, default='trainaug.txt')

args = seg_parser.parse_args()

train_file = args.train_file

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

nr_vocaug_train = 10582
# nr_voc_train = 1464
nr_voc_val = 1449

nr_train = nr_vocaug_train

epochs = args.epochs
learning_rate = args.learning_rate
end_learning_rate = args.end_learning_rate
batch_size = args.batch_size
max_steps = (1 + nr_train // batch_size) * epochs
if args.decay_step is None:
    decay_step = max_steps
else:
    decay_step = args.decay_step

crop_size = args.crop_size
ignore_label = args.ignore_label

model_name = 'RES50_PSPNet'
sub_model_dir = f'{model_name}#{args.train_file.split(".")[0]}#{epochs}#{batch_size}#{learning_rate}#{end_learning_rate}#{max_steps}'
run_config = tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_steps=1000, keep_checkpoint_max=15)
FCN32s = tf.estimator.Estimator(
    model_fn=FCN_estimator,
    model_dir=join(summary_home, sub_model_dir),
    config=run_config,
    params={
        'front_end': ResNet50,
        'image_height': 473,
        'image_width': 473,
        'model': PSPNet,
        'num_classes': args.num_classes,
        'ignore_label': args.ignore_label,
        'learning_rate': learning_rate,
        'end_learning_rate': end_learning_rate,
        'momentum': args.momentum,
        'lr_decay': args.lr_decay,
        'decay_step': decay_step,
        'palette': VOC_palette,
        'batch_size': batch_size,
        'weight_decay': args.weight_decay,
    }
)

train_hook = tf.train.LoggingTensorHook({
    'learning_rate': 'learning_rate',
    'loss': 'loss',
    'cross_entropy_loss': 'cross_entropy_loss',
    'acc': 'acc',
    'miou': 'miou',
}, 10)

train_spec = tf.estimator.TrainSpec(
    lambda: VOC_pattern_input_fn(VOC_home, train_file, batch_size=batch_size, num_epoches=epochs,
                                 data_aug=lambda image, label: data_augment(image, label, crop_size, crop_size,
                                                                            ignore_label)), max_steps, [train_hook])
eval_spec = tf.estimator.EvalSpec(lambda: VOC_pattern_input_fn(VOC_home, 'val.txt', batch_size=1, is_training=False),
                                  throttle_secs=1800)

tf.estimator.train_and_evaluate(FCN32s, train_spec, eval_spec)

