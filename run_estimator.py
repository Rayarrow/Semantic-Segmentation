import math
import re
from argparse import ArgumentParser
import os

from data_loader import VOC_pattern_input_fn
from data_preprocess import data_augment
from obsolete.data_process_vis_utils import maybe_create_dir
from model_estimators import segmentation_model_fn
from model_encoders import *
from model_segmentation import *
from palette_conversion import *

seg_parser = ArgumentParser('semantic Segmentation')

# IO
seg_parser.add_argument('--dataset', type=str, default='VOC',
                        help='Dataset name. It in turn defines image paths, label paths, palette and so on Refer to `config.py` for further information.')
# seg_parser.add_argument('--model_dir', type=str, default=summary_home,
#                         help='Used to store model. It defaults to `summary_home` defined in `config.py`.')
seg_parser.add_argument('--mode', type=str, default='t', choices=['t', 'e', 'p', 'o'],
                        help='It refers to "train, evaluate, prediction, observe" respectively. `observe` mode is used to help inspect the training dataset.')
seg_parser.add_argument('--inference_root', type=str,
                        help='Only used in non-training mode. A directory used to store prediction results. If not provided, `model_dir` will be used instead.')
seg_parser.add_argument('--eval_dir', help='If this argument is provided, this will override the dir inferred from '
                                           'model name and training config information such as `batch_size`, '
                                           '`encoder` and so on.')
seg_parser.add_argument('--infer_eval_dir', action='store_true',
                        help='Only valid in non-training mode and for models trained in this project, used to infer model hyper-parameters from the `eval_dir` name.')

seg_parser.add_argument('--encoder', type=str,
                        help='Specify the encoder of the model. Refer to `encoder_dict` defined below to choose valid encoder.')
seg_parser.add_argument('--decoder', type=str,
                        help='Specify the decoder of the model. Refer to `decoder_dict` defined below to choose valid encoder.')
seg_parser.add_argument('--init_model_path', type=str,
                        help='Used to fine-tune a trained task-specific model. for example: fine-tune a segmentation '
                             'model already trained on VOC. Note: leave this argument empty if your\'re going to '
                             'initialize your model from ImageNet pretrained model')

seg_parser.add_argument('--datalist_train', default='train.txt',
                        help='A txt file located in `datalist_home`, used to specify image ids (file BASEname) for training.')
seg_parser.add_argument('--datalist_val', default='val.txt',
                        help='A txt file located in `datalist_home`, used to specify image ids (file BASEname) for validation.')
# Model
seg_parser.add_argument('--num_classes', type=int, default=21,
                        help='Number of classes in the provided dataset. Actually, it can be inferred from palette defined in `config.py` if not specified.')
seg_parser.add_argument('--ignore_label', type=int, default=255,
                        help='Labels ignored when the loss is being calculated. This must be defined as a value different from valid class label, as it will be used in data augmentation step.')
seg_parser.add_argument('--crop_size', type=str, default='513',
                        help='The size of patch to be cropped in data augmentation step. "513" and "513,500" are two valid formats.')
seg_parser.add_argument('--get_FCN', type=int, default=1,
                        help='If set to False, FCN will not be desired and the top level of the encoders will be set as fully connected layers or global pooling layers.')
seg_parser.add_argument('--structure_mode', default='seg', choices=['seg', 'siamese', 'sup'],
                        help='seg=semantic_segmentation. sup=superimpose. The latter two are used for change detection tasks.')

# Learning control
seg_parser.add_argument('--learning_rate', type=float, default=1e-4)
seg_parser.add_argument('--power', type=float, default=0.9, help='Used for polynomial learning policy control.')
seg_parser.add_argument('--end_learning_rate', type=float, default=1e-6,
                        help='Used for polynomial learning policy control.')
seg_parser.add_argument('--momentum', type=float, default=0.9)
seg_parser.add_argument('--lr_decay', type=str, default='poly', choices=['poly', 'stable'])
seg_parser.add_argument('--decay_step', type=int, default=None,
                        help='After `decay_step`, learning rate will drop to `end_learning_rate`.')
seg_parser.add_argument('--weight_decay', type=float, default=2e-4)
seg_parser.add_argument('--bn_scale', action='store_true',
                        help='Set `gamma` in BN layers frozen if `bn_scale`=False. In most cases, False get a better result.')
seg_parser.add_argument('--frozen', action='store_true', help='freeze BN layers or not.')

seg_parser.add_argument('--epochs', type=int, default=30)
seg_parser.add_argument('--batch_size', type=int, default=8)
seg_parser.add_argument('--devices', type=str, default='0', help='Specify which GPU to use.')
seg_parser.add_argument('--eval_interval', type=int, default=1000,
                        help='Evaluate the model using validation dataset every `eval_interval` seconds.')

seg_parser.add_argument('--extra', type=str, default='',
                        help='Other comments on this training process. This will be append to the end of `model_dir`.')

args = seg_parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

# ========================= Parse dataset and its subdir =========================

dataset = args.dataset

# `image_home`, `label_home` and so on are defined based on `dataset` given above.
if dataset not in data_home_bundle:
    raise Exception('unregistered dataset name.')
else:
    dataset_config_bundle = data_home_bundle[dataset]
    data_home = dataset_config_bundle[0]
    image_home = join(data_home, dataset_config_bundle[1])
    label_home = join(data_home, dataset_config_bundle[2])
    datalist_home = join(data_home, dataset_config_bundle[3])
    palette = dataset_config_bundle[4]

datalist_train_path = join(datalist_home, args.datalist_train)
datalist_val_path = join(datalist_home, args.datalist_val)

if args.mode == 'p' and not os.path.exists(datalist_train_path):
    with open(datalist_val_path) as f:
        nr_train = len(f.readlines())
else:
    with open(datalist_train_path) as f:
        nr_train = len(f.readlines())

epochs = args.epochs
learning_rate = args.learning_rate
end_learning_rate = args.end_learning_rate
batch_size = args.batch_size

# `step` is necessary but not defined and is hard to defined due to the different number of images of different dataset. `step` will be inferred from your dataset and `batch_size.
max_steps = int(math.ceil(nr_train / batch_size)) * epochs
if args.decay_step is None:
    decay_step = max_steps
else:
    decay_step = args.decay_step

# Parse `crop_size`.
crop_size = args.crop_size
if ',' in crop_size:
    crop_size_height, crop_size_width = map(int, crop_size.split(','))
else:
    crop_size_height = crop_size_width = int(crop_size)

ignore_label = args.ignore_label

encoder = args.encoder
decoder = args.decoder
model_name = f'{encoder}_{decoder}'
init_model_path = args.init_model_path
structure_mode = args.structure_mode

# Set up `model dir`.
# If'eval_dir' is provided and `infer_eval_dir` is set to true, it will override the inferred model dir.
eval_dir = args.eval_dir
if eval_dir:
    if not os.path.exists(eval_dir):
        raise Exception('`eval_dir` does not exist.')

    if args.infer_eval_dir:
        logger.info('Inferring models from eval_dir...')
        session_name = os.path.basename(eval_dir)
        encoder, decoder = session_name.split('#')[0].split('_')

    if eval_dir.startswith('@'):
        model_dir = join(summary_home, eval_dir[1:])
    else:
        model_dir = eval_dir
else:
    session_name = f'{model_name}#{dataset}#{args.datalist_train.split(".")[0]}#{epochs}#{batch_size}#{learning_rate}#{end_learning_rate}#{max_steps}#{crop_size}#{args.bn_scale}#{ignore_label}#{structure_mode}#{args.extra}'
    model_dir = join(summary_home, dataset, session_name)

maybe_create_dir(model_dir)

# Parse the encoder.
if 'SRes' in encoder:
    parse_res = re.search(r'(S?Res)(\d+)@(\d+)', encoder)
    res_name = parse_res.group(1)
    res_depth = int(parse_res.group(2))
    res_stride = int(parse_res.group(3))
    encoder = res_name
elif 'SVGG' in encoder:
    parse_vgg = re.search(r'(S?VGG)(\d+)', encoder)
    vgg_name = parse_vgg.group(1)
    vgg_depth = int(parse_vgg.group(2))
    encoder = vgg_name

# Parse the decoder.
if decoder.startswith('SimpleDeconv'):
    if decoder == 'SimpleDeconv':
        fuse = []

    else:
        fuse = sorted(map(int, decoder[12:]))
        decoder = decoder[:12]

elif decoder.startswith('Deeplabv3'):
    # Parse the rates of the ASPP module.
    if '@' not in decoder:
        ASPP_rates = (6, 12, 18)
    elif '@' in decoder:
        decoder, ASPP_rates = decoder.split('@')
        ASPP_rates = map(int, ASPP_rates.split('x'))
        ASPP_rates = tuple(ASPP_rates)
    else:
        raise Exception('The provided Deeplabv3 format is not valid.')

    # Determine if it's "Deeplabv3 plus" or "Deeplab v3".
    if decoder == 'Deeplabv3p':
        plus = True
        decoder = 'Deeplabv3'
    else:
        plus = False

encoder_dict = {
    'VGG16': VGG16,
    'Res50': ResNet50,
    'SRes': lambda X_input, image_height, image_width, get_FCN, is_training: \
        SlimResNet(X_input, image_height, image_width, res_depth, res_stride, get_FCN, is_training,
                   init_model_path is None),
    'SVGG': lambda X_input, image_height, image_width, get_FCN, is_training: \
        SlimVGG(X_input, image_height, image_width, vgg_depth, get_FCN, is_training, init_model_path is None),
    'SimpleConv': lambda X_input, image_height, image_width, get_FCN, is_training: \
        SimpleConv(X_input, image_height, image_width, get_FCN, is_training, False),
    'SimpleConvStack': lambda X_input, image_height, image_width, get_FCN, is_training: \
        SimpleConv(X_input, image_height, image_width, get_FCN, is_training, True)
}

decoder_dict = {
    'FCN32s': lambda encoder, num_classes, is_training: FCN(encoder, 32, num_classes, is_training),
    'FCN16s': lambda encoder, num_classes, is_training: FCN(encoder, 16, num_classes, is_training),
    'FCN8s': lambda encoder, num_classes, is_training: FCN(encoder, 8, num_classes, is_training),
    'PSPNet': PSPNet,
    'UNet': UNet,
    'Deeplabv2': Deeplabv2,
    'Deeplabv3': lambda encoder, num_class, is_training: Deeplabv3(encoder, num_class, is_training, args.bn_scale,
                                                                   plus=plus, ASPP_rates=ASPP_rates),
    'SimpleDeconv': lambda encoder, num_class, is_training: SimpleDeconv(encoder, num_class, is_training,
                                                                         args.bn_scale, fuse),
}

with open(join(model_dir, 'parameters.txt'), 'w', encoding='utf8') as f:
    f.write(str(args))

file_handler = logging.FileHandler(join(model_dir, 'log.log'))
file_handler.setFormatter(common_formater)
logger.addHandler(file_handler)

run_config = tf.estimator.RunConfig(save_summary_steps=10, save_checkpoints_steps=1000, keep_checkpoint_max=15)
seg_model = tf.estimator.Estimator(
    model_fn=segmentation_model_fn,
    model_dir=model_dir,
    config=run_config,
    params={
        'encoder': encoder_dict[encoder],
        'image_height': crop_size_height,
        'image_width': crop_size_width,
        'decoder': decoder_dict[decoder],
        'init_model_path': args.init_model_path,
        # 'num_classes': args.num_classes,
        'num_classes': len(palette),
        'ignore_label': args.ignore_label,
        'learning_rate': learning_rate,
        'end_learning_rate': end_learning_rate,
        'momentum': args.momentum,
        'lr_decay': args.lr_decay,
        'power': args.power,
        'decay_step': decay_step,
        'palette': VOC_palette,
        'batch_size': batch_size,
        'frozen': args.frozen,
        'weight_decay': args.weight_decay,
        'structure_mode': structure_mode,
    }
)

train_hook = tf.train.LoggingTensorHook({
    'learning_rate': 'learning_rate',
    'loss': 'loss',
    'cross_entropy_loss': 'cross_entropy_loss',
    'acc': 'acc',
    'miou': 'miou',
    'f1': 'f1',
    # 'debug_moving_mean': 'debug_moving_mean',
    # 'debug_moving_variance': 'debug_moving_variance',
}, 10)

eval_hook = tf.train.LoggingTensorHook({
    'loss': 'loss',
    'cross_entropy_loss': 'cross_entropy_loss',
    'acc': 'acc',
    'miou': 'miou',
    'f1': 'f1',
    # 'debug_moving_mean': 'debug_moving_mean',
    # 'debug_moving_variance': 'debug_moving_variance',
}, 1)

load_pair = structure_mode in ['siamese', 'sup']

data_aug = lambda image, label: data_augment(image, label, crop_size_height, crop_size_width, ignore_label)
train_input_fn = lambda: VOC_pattern_input_fn(
    image_home, label_home, datalist_train_path, 3, epochs, batch_size, pair=load_pair, data_aug=data_aug)

val_input_fn = lambda: VOC_pattern_input_fn(image_home, label_home, datalist_val_path, batch_size=1, pair=load_pair,
                                            is_training=False)

if args.mode == 't':
    train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps, [train_hook])
    eval_spec = tf.estimator.EvalSpec(val_input_fn, throttle_secs=args.eval_interval)

    tf.estimator.train_and_evaluate(seg_model, train_spec, eval_spec)

elif args.mode == 'e':
    seg_model.evaluate(val_input_fn, hooks=[eval_hook])

elif args.mode == 'p':
    predictions = seg_model.predict(val_input_fn)
    with open(datalist_val_path) as f:
        ids = f.read().split()
    inference_root = args.inference_root
    if not inference_root:
        inference_root = model_dir
    inference_dir = join(inference_root, session_name) if inference_root else join(inference_root, 'label')

    with open(datalist_val_path) as f:
        ids = f.read().split()

    image_names = []
    entropy_maps = []
    labels = []
    prob_maps = []


    def get_heatmap(data, normalized=True):
        import numpy as np
        import matplotlib.pyplot as plt

        # a colormap and a normalization instance
        cmap = plt.cm.jet
        if normalized:
            norm = plt.Normalize(vmin=data.min(), vmax=data.max())
            image = cmap(norm(data))
        else:
            image = cmap(data)

        # map the normalized data to colors
        # image is now RGBA (512x512x4)

        # save the image
        return image


    for pred_dict, id in zip(predictions, ids):
        logger.info(f'Calculating information for {id}...')
        labels.append(np.squeeze(pred_dict['y_pred']))

        prob_map = pred_dict['y_prob']
        prob_maps.append(prob_map)
        entropy_map = np.mean(-(np.log(prob_map) * prob_map), axis=-1)
        entropy_maps.append(entropy_map)

    heatmaps_raw = [get_heatmap(each_ent_map, False) for each_ent_map in entropy_maps]
    heatmaps = [get_heatmap(each_ent_map) for each_ent_map in entropy_maps]

    save_images(join(inference_dir, 'labels'), labels, ids, 'png')
    save_images(join(inference_dir, 'heatmaps_raw'), heatmaps_raw, ids, 'png')
    save_images(join(inference_dir, 'heatmaps'), heatmaps, ids, 'png')

    from post_VOC_pattern import PostPrediction

    p = PostPrediction(inference_root, session_name, dataset, args.datalist_val, None)
    p.run_get_basic_info()


elif args.mode == 'o':
    image_batch, label_batch = val_input_fn()
    with tf.Session() as sess:
        image_batch_val, label_batch_val = sess.run([image_batch, label_batch])
        print(image_batch_val.shape, '\t', label_batch_val.shape)

else:
    raise Exception('Invalid mode.')
