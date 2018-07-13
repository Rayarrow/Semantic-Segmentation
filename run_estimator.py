from argparse import ArgumentParser

from model_estimators import segmentation_model_fn
from models_front_end import *
from models_segmentation import *
from palette_conversion import *
import re
from tensorflow.python import debug as tf_debug

seg_parser = ArgumentParser('semantic Segmentation')
# IO
seg_parser.add_argument('--model_dir', type=str, default=summary_home)
seg_parser.add_argument('--front_end', type=str)
seg_parser.add_argument('--model', type=str)
seg_parser.add_argument('--init_model_path', type=str,
                        help='Used to fine-tune a trained task-specific model. for example: fine-tune a segmentation '
                             'model already trained on VOC. Note: leave this argument empty if your\'re going to '
                             'initialize your model from ImageNet pretrained model')
seg_parser.add_argument('--mode', type=str, default='t', choices=['t', 'e', 'p', 'o'],
                        help='train, evaluate, prediction, observe')
seg_parser.add_argument('--eval_dir', help='If this argument is provided, this will override the dir inferred from '
                                           'model name and training config information such as `batch_size`, '
                                           '`front_end` and so on.')

seg_parser.add_argument('--datalist_train', default='train.txt')
seg_parser.add_argument('--datalist_val', default='val.txt')
# Model
seg_parser.add_argument('--num_classes', type=int, default=21)
seg_parser.add_argument('--ignore_label', type=int, default=255)
seg_parser.add_argument('--crop_size', type=int, default=513)
seg_parser.add_argument('--get_FCN', type=int, default=1)

# Learning control
seg_parser.add_argument('--learning_rate', type=float, default=1e-4)
seg_parser.add_argument('--power', type=float, default=0.9)
seg_parser.add_argument('--end_learning_rate', type=float, default=1e-6)
seg_parser.add_argument('--momentum', type=float, default=0.9)
seg_parser.add_argument('--lr_decay', type=str, default='poly')
seg_parser.add_argument('--decay_step', type=int, default=None)
seg_parser.add_argument('--weight_decay', type=float, default=2e-4)

seg_parser.add_argument('--epochs', type=int, default=30)
seg_parser.add_argument('--batch_size', type=int, default=4)
seg_parser.add_argument('--devices', type=str, default='0')
seg_parser.add_argument('--eval_interval', type=int, default=1000)

seg_parser.add_argument('--extra', type=str, default='')

args = seg_parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

datalist_train_path = join(datalist_home, args.datalist_train)
datalist_val_path = join(datalist_home, args.datalist_val)

with open(datalist_train_path) as f:
    nr_train = len(f.readlines())

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

front_end = args.front_end
model = args.model
model_name = f'{front_end}_{model}'
init_model_path = args.init_model_path

if 'SRes' in front_end:
    parse_res = re.search(r'(S?Res)(\d+)@(\d+)', front_end)
    res_name = parse_res.group(1)
    res_depth = int(parse_res.group(2))
    res_stride = int(parse_res.group(3))
    front_end = res_name
elif 'SVGG' in front_end:
    parse_vgg = re.search(r'(S?VGG)(\d+)', front_end)
    vgg_name = parse_vgg.group(1)
    vgg_depth = int(parse_vgg.group(2))
    front_end = vgg_name

front_end_dict = {
    'VGG16': VGG16,
    'Res50': ResNet50,
    'SRes': lambda X_input, image_height, image_width, get_FCN, is_training: \
        SlimResNet(X_input, image_height, image_width, res_depth, res_stride, get_FCN, is_training,
                   init_model_path is None),
    'SVGG': lambda X_input, image_height, image_width, get_FCN, is_training, is_init: \
        SlimVGG(X_input, image_height, image_width, vgg_depth, get_FCN, is_training, init_model_path is None)
}

decoder_dict = {
    'FCN32s': lambda front_end, num_classes, is_training: FCN(front_end, 32, num_classes, is_training),
    'FCN16s': lambda front_end, num_classes, is_training: FCN(front_end, 16, num_classes, is_training),
    'FCN8s': lambda front_end, num_classes, is_training: FCN(front_end, 8, num_classes, is_training),
    'PSPNet': PSPNet,
    'UNet': UNet,
    'Deeplabv2': Deeplabv2,
    'Deeplabv3': Deeplabv3,
}

# Set up `model dir`. Provided 'eval_dir' will override the inferred model dir.
if args.eval_dir:
    model_dir = args.eval_dir
else:
    sub_model_dir = f'{model_name}#{data}#{args.datalist_train.split(".")[0]}#{epochs}#{batch_size}#{learning_rate}#{end_learning_rate}#{max_steps}#{crop_size}#{args.extra}'
    model_dir = join(summary_home, sub_model_dir)

run_config = tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_steps=1000, keep_checkpoint_max=15)
seg_model = tf.estimator.Estimator(
    model_fn=segmentation_model_fn,
    model_dir=model_dir,
    config=run_config,
    params={
        'front_end': front_end_dict[front_end],
        'image_height': crop_size,
        'image_width': crop_size,
        'model': decoder_dict[model],
        'init_model_path': args.init_model_path,
        'num_classes': args.num_classes,
        'ignore_label': args.ignore_label,
        'learning_rate': learning_rate,
        'end_learning_rate': end_learning_rate,
        'momentum': args.momentum,
        'lr_decay': args.lr_decay,
        'power': args.power,
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
    # 'debug_moving_mean': 'debug_moving_mean',
    # 'debug_moving_variance': 'debug_moving_variance',
}, 10)

eval_hook = tf.train.LoggingTensorHook({
    'loss': 'loss',
    'cross_entropy_loss': 'cross_entropy_loss',
    'acc': 'acc',
    'miou': 'miou',
    # 'debug_moving_mean': 'debug_moving_mean',
    # 'debug_moving_variance': 'debug_moving_variance',
}, 1)

train_input_fn = lambda: VOC_pattern_input_fn(
    image_home, label_home, datalist_train_path, 3, epochs, batch_size,
    data_aug=lambda image, label: data_augment(image, label, crop_size, crop_size, ignore_label))

val_input_fn = lambda: VOC_pattern_input_fn(image_home, label_home, datalist_train_path, batch_size=1,
                                            is_training=False)

if args.mode == 't':
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps, [train_hook])
    eval_spec = tf.estimator.EvalSpec(val_input_fn, throttle_secs=args.eval_interval)

    tf.estimator.train_and_evaluate(seg_model, train_spec, eval_spec)

elif args.mode == 'e':
    seg_model.evaluate(val_input_fn, hooks=[eval_hook])

elif args.mode == 'p':
    from skimage import io

    predictions = seg_model.predict(val_input_fn)
    with open(join(VOC_home, 'ImageSets/Segmentation/val.txt')) as f:
        ids = f.read().split()
    labels = [pred_dict['y_pred'] for pred_dict in predictions]
    save_images(join(model_dir, 'labels'), labels, ids, 'png')

elif args.mode == 'o':
    image_batch, label_batch = val_input_fn()
    with tf.Session() as sess:
        image_batch_val, label_batch_val = sess.run([image_batch, label_batch])
        print(image_batch_val.shape, '\t', label_batch_val.shape)

else:
    raise Exception('Invalid mode.')
