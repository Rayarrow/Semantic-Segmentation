from data_loader import *
from data_preprocess import mean_substract
from models_constructor import *
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v2
from config import batch_norm_decay

RGB_MEAN = np.array([[[[123.68, 116.78, 103.94]]]], float)


# Not used for now.
def get_ele(model, upsample_mode):
    get_conv = model.front.cc.get_conv

    if upsample_mode == UPSAMPLE_BILINEAR:
        upsample = model.front.cc.bilinear
    elif upsample_mode == UPSAMPLE_DECONV:
        upsample = model.front.cc.get_deconv
    else:
        raise Exception('invalid upsample mode provided.')

    return get_conv, upsample


class VGG16():
    def __init__(self, X_input, image_height, image_width, get_FCN=0, is_training=None, is_init=None):
        self.cc = component_constructor(vgg16_npy_path, 0, 1)
        get_conv = self.cc.get_conv
        get_maxpooling = self.cc.get_maxpooling
        get_fc = self.cc.get_fc
        self.image_height = image_height
        self.image_width = image_width
        self.X_input = X_input

        # Declare the mean of each channel in "(b, g, r)" order, which is gotten from ImageNet dataset.
        with tf.variable_scope('input'):
            X_input = tf.cast(X_input, tf.float32)
            X_input = mean_substract(X_input)

        self.conv1_1 = get_conv('conv1_1', X_input, True, True, True)
        self.conv1_2 = get_conv('conv1_2', self.conv1_1, True, True, True)

        self.maxpool_1 = get_maxpooling('maxpool_1', self.conv1_2)
        self.conv2_1 = get_conv('conv2_1', self.maxpool_1, True, True, True)
        self.conv2_2 = get_conv('conv2_2', self.conv2_1, True, True, True)

        self.maxpool_2 = get_maxpooling('maxpool_2', self.conv2_2)
        self.conv3_1 = get_conv('conv3_1', self.maxpool_2, True, True, True)
        self.conv3_2 = get_conv('conv3_2', self.conv3_1, True, True, True)
        self.conv3_3 = get_conv('conv3_3', self.conv3_2, True, True, True)

        self.maxpool_3 = get_maxpooling('maxpool_3', self.conv3_3)
        self.conv4_1 = get_conv('conv4_1', self.maxpool_3, True, True, True)
        self.conv4_2 = get_conv('conv4_2', self.conv4_1, True, True, True)
        self.conv4_3 = get_conv('conv4_3', self.conv4_2, True, True, True)

        self.maxpool_4 = get_maxpooling('maxpool_4', self.conv4_3)
        self.conv5_1 = get_conv('conv5_1', self.maxpool_4, True, True, True)
        self.conv5_2 = get_conv('conv5_2', self.conv5_1, True, True, True)
        self.conv5_3 = get_conv('conv5_3', self.conv5_2, True, True, True)
        self.maxpool_5 = get_maxpooling('maxpool_5', self.conv5_3)

        # Due to the arbitrary-sized images for training the FCN, fully connected layers cannot be initialized
        # properly, which requires the input size of images fixed.

        logger.debug('get_FCN: {}'.format(get_FCN))

        # initialized the last few layers.
        if get_FCN == 0:
            self.fc_input = tf.reshape(self.maxpool_5, shape=[-1, np.prod(self.maxpool_5.get_shape().as_list()[1:])])
            self.fc6 = get_fc('fc6', self.fc_input)
            self.fc7 = get_fc('fc7', self.fc6)
            self.fc8 = get_fc('fc8', self.fc7)
            self.y_pred = tf.argmax(self.fc8, axis=1)

        elif get_FCN == 2:
            self.fcn6 = self.cc.get_fully_as_CNN('fc6', self.maxpool_5, [7, 7, 512, 4096])
            self.fcn7 = self.cc.get_fully_as_CNN('fc7', self.fcn6, [1, 1, 4096, 4096])

        elif get_FCN == 1:
            self.fcn6 = get_conv('fc6', self.maxpool_5, False, True, True, k_size=(7, 7), num_output=4096)
            self.fcn7 = get_conv('fc7', self.fcn6, False, True, True, num_output=4096)

        else:
            raise Exception('invalid get_FCN.')

        self.pool1 = self.maxpool_1
        self.pool2 = self.maxpool_2
        self.pool3 = self.maxpool_3
        self.pool4 = self.maxpool_4
        self.pool5 = self.maxpool_5

        self.f1 = self.conv1_2
        self.f2 = self.conv2_2
        self.f4 = self.conv3_3
        self.f8 = self.conv4_3
        self.f16 = self.conv5_3
        self.f32 = self.fcn7


class ResNet50():
    def __init__(self, X_input, image_height, image_width, get_FCN=False, rates=(2, 4), is_training=None, is_init=None):
        self.cc = component_constructor(res50_npy_path)
        self.X_input = X_input
        self.image_height = image_height
        self.image_width = image_width

        get_conv = self.cc.get_conv
        get_bn = self.cc.get_bn
        get_maxpooling = self.cc.get_maxpooling
        get_fc = self.cc.get_fc
        get_bottleneck = self.cc.get_bottleneck

        with tf.variable_scope('input'):
            X_input = tf.cast(X_input, tf.float32)
            X_input = mean_substract(X_input)

        with tf.variable_scope('block1'):
            self.conv1 = get_conv('conv1', X_input, strides=[1, 2, 2, 1])
            self.bn1 = get_bn('bn_conv1', self.conv1)
            self.pool1 = get_maxpooling('pool1', self.bn1, [3, 3])

        with tf.variable_scope('block2'):
            self.pool2 = get_bottleneck(self.pool1, 2, 3, 3)

        with tf.variable_scope('block3'):
            self.pool3 = get_bottleneck(self.pool2, 3, 4, 3, pooling=(0, 1))

        with tf.variable_scope('block4'):
            self.pool4 = get_bottleneck(self.pool3, 4, 6, 3, atrous=True, rates=rates[0])
            # self.pool4 = get_bottleneck(self.pool3, 4, 6, 3, pooling=(0, 1))

        with tf.variable_scope('block5'):
            self.pool5 = get_bottleneck(self.pool4, 5, 3, 3, atrous=True, rates=rates[1])
            # self.pool5 = get_bottleneck(self.pool4, 5, 3, 3, pooling=(0, 1))
            self.f32 = self.pool5

        if not get_FCN:
            self.global_pooling = tf.reduce_mean(self.pool5, axis=[1, 2])
            self.fc1000 = get_fc('fc1000', self.global_pooling)
            self.y_pred = tf.argmax(self.fc1000, axis=-1, output_type=tf.int32, name='y_pred')


class SlimResNet():
    def __init__(self, X_input, image_height, image_width, depth=50, stride=32, get_FCN=1, is_training=True,
                 is_init=True):
        self.X_input = X_input
        self.image_height = image_height
        self.image_width = image_width
        self.cc = component_constructor(None)
        X_input -= RGB_MEAN

        pool_name = [f'resnet_v2_{depth}/conv1', f'resnet_v2_{depth}/block1/unit_1/bottleneck_v2',
                     f'resnet_v2_{depth}/block1', f'resnet_v2_{depth}/block2', f'resnet_v2_{depth}/block3']
        f_name = [f'resnet_v2_{depth}/conv1', f'resnet_v2_{depth}/block1/unit_3/bottleneck_v2/conv1',
                  f'resnet_v2_{depth}/block2/unit_4/bottleneck_v2/conv1',
                  f'resnet_v2_{depth}/block3/unit_6/bottleneck_v2/conv1', f'resnet_v2_{depth}/block4']
        if depth == 50:
            model = resnet_v2.resnet_v2_50
            model_path = res50_ckpt_path
        elif depth == 101:
            model = resnet_v2.resnet_v2_101
            model_path = res101_ckpt_path
            f_name[3] = f'resnet_v2_101/block3/unit_23/bottleneck_v2/conv1'

        else:
            raise Exception(f'invalid resnet depth={depth}.')

        with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
            net, nodes = model(X_input, global_pool=not get_FCN, output_stride=stride, is_training=is_training)

        if is_training and is_init:
            variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            variables_to_restore.remove(tf.train.get_global_step())
            tf.train.init_from_checkpoint(model_path, {v.name.split(':')[0]: v for v in variables_to_restore})

            # print('===================================================\n\n')
            # print(is_training)
            # print('===================================================\n\n')
            #
            # with tf.Session() as sess:
            #     sess.run(tf.global_variables_initializer())
            #     with tf.variable_scope('', reuse=True):
            #         w = tf.get_variable('resnet_v2_50/block3/unit_2/bottleneck_v2/conv2/BatchNorm/moving_mean')
            #     print(sess.run(w))
            #     exit(1)

        self.pool1 = nodes[pool_name[0]]
        self.pool2 = nodes[pool_name[1]]
        self.pool3 = nodes[pool_name[2]]
        self.pool4 = nodes[pool_name[3]]
        self.pool5 = nodes[pool_name[4]]

        self.f2 = nodes[f_name[0]]
        self.f4 = nodes[f_name[1]]
        self.f8 = nodes[f_name[2]]
        self.f16 = nodes[f_name[3]]
        self.f32 = nodes[f_name[4]]
        self.net = net

        print(self.f32.get_shape())


class SlimVGG():
    def __init__(self, X_input, image_height, image_width, depth=16, get_FCN=1, is_training=None, is_init=True):
        from tensorflow.contrib.slim.nets import vgg
        self.X_input = X_input
        self.image_height = image_height
        self.image_width = image_width
        self.cc = component_constructor(None)
        X_input -= RGB_MEAN

        with tf.contrib.slim.arg_scope(vgg.vgg_arg_scope()):
            pool_name = [f'vgg_{depth}/pool{i}' for i in (1, 2, 3, 4, 5)]
            f_name = [f'vgg_{depth}/conv{i}/conv{i}_{j}' for i, j in ((1, 2), (2, 2), (3, 3), (4, 3), (5, 3))]
            if depth == 16:
                model = vgg.vgg_16
                model_path = vgg16_ckpt_path

            elif depth == 19:
                model = vgg.vgg_19
                model_path = vgg19_ckpt_path
            else:
                raise Exception(f'invalid vgg depth={depth}.')

        if is_training and is_init:
            variables_to_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            variables_to_restore.remove(tf.train.get_global_step())
            tf.train.init_from_checkpoint(model_path, {v.name.split(':')[0]: v for v in variables_to_restore})

            net, nodes = model(X_input, spatial_squeeze=not get_FCN)
            self.fcn6 = self.cc.get_conv('fcn6', nodes[pool_name[4]], False, True, True, k_size=(7, 7), num_output=4096)
            self.fcn7 = self.cc.get_conv('fcn7', self.fcn6, False, True, True, num_output=4096)

        self.pool1 = nodes[pool_name[0]]
        self.pool2 = nodes[pool_name[1]]
        self.pool3 = nodes[pool_name[2]]
        self.pool4 = nodes[pool_name[3]]
        self.pool5 = nodes[pool_name[4]]

        self.f2 = nodes[f_name[0]]
        self.f4 = nodes[f_name[1]]
        self.f8 = nodes[f_name[2]]
        self.f16 = nodes[f_name[3]]
        self.f32 = self.fcn7 if get_FCN else net
        self.net = net
