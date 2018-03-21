from models_constructor import *


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
    def __init__(self, image_height, image_width, image_channel=3, get_FCN=0):
        self.cc = component_constructor(vgg16_npy_path, 0, 1)
        get_conv = self.cc.get_conv
        get_maxpooling = self.cc.get_maxpooling
        get_fc = self.cc.get_fc
        self.image_height = image_height
        self.image_width = image_width

        # Declare the mean of each channel in "(b, g, r)" order, which is gotten from ImageNet dataset.
        with tf.variable_scope('input'):
            self.channel_mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[3], name='mean')
            self.X_input = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel], 'X_input')
            self.y_input = tf.placeholder(tf.int32, [None], name='y_input')

            r, g, b = tf.split(self.X_input, 3, axis=3, name='rgb_spliter')
            self.X_input_transformed = tf.concat([b, g, r], axis=3) - self.channel_mean

        self.conv1_1 = get_conv('conv1_1', self.X_input_transformed, True, True, True)
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
            self.cross_entropy = tf.losses.sparse_softmax_cross_entropy(self.y_input, self.fc8)

        elif get_FCN == 1:
            self.fcn6 = self.cc.get_fully_as_CNN('fc6', self.maxpool_5, [7, 7, 512, 4096])
            self.fcn7 = self.cc.get_fully_as_CNN('fc7', self.fcn6, [1, 1, 4096, 4096])

        elif get_FCN == 2:
            self.fcn6 = get_conv('fc6', self.maxpool_5, False, True, True, k_size=(7, 7), num_output=4096)
            self.fcn7 = get_conv('fc7', self.fcn6, False, True, True, num_output=4096)

        else:
            raise Exception('invalid get_FCN.')

        self.pool1 = self.maxpool_1
        self.pool2 = self.maxpool_2
        self.pool3 = self.maxpool_3
        self.pool4 = self.maxpool_4
        self.pool5 = self.maxpool_5 if not get_FCN else self.fcn7


class ResNet50():
    def __init__(self, image_height, image_width, image_channel=3, get_FCN=False, weight_decay=None, beta_decay=None,
                 gamma_decay=None, rates=(2, 4)):
        self.cc = component_constructor(res50_npy_path, weight_decay=weight_decay, beta_decay=beta_decay,
                                        gamma_decay=gamma_decay)
        self.image_height = image_height
        self.image_width = image_width

        get_conv = self.cc.get_conv
        get_bn = self.cc.get_bn
        get_maxpooling = self.cc.get_maxpooling
        get_fc = self.cc.get_fc
        get_bottleneck = self.cc.get_bottleneck

        with tf.variable_scope("input"):
            self.channel_mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[3])
            self.X_input = tf.placeholder(tf.float32, shape=[None, image_height, image_width, image_channel])
            self.y_input = tf.placeholder(tf.int32, shape=[None])

            r, g, b = tf.split(self.X_input, 3, axis=3)
            self.X_input_transformed = tf.concat([b, g, r], axis=3) - self.channel_mean

        # conv1
        with tf.variable_scope('block1'):
            self.conv1 = get_conv('conv1', self.X_input_transformed, strides=[1, 2, 2, 1])
            self.bn1 = get_bn('bn_conv1', self.conv1)
            self.pool1 = get_maxpooling('pool1', self.bn1, [3, 3])

        with tf.variable_scope('block2'):
            self.pool2 = get_bottleneck(self.pool1, 2, 3, 3)

        with tf.variable_scope('block3'):
            self.pool3 = get_bottleneck(self.pool2, 3, 4, 3, pooling=(0, 1))

        with tf.variable_scope('block4'):
            self.pool4 = get_bottleneck(self.pool3, 4, 6, 3, atrous=True, rates=rates[0])

        with tf.variable_scope('block5'):
            self.pool5 = get_bottleneck(self.pool4, 5, 3, 3, atrous=True, rates=rates[1])

        if not get_FCN:
            self.global_pooling = tf.reduce_mean(self.pool5, axis=[1, 2])
            self.fc1000 = get_fc('fc1000', self.global_pooling)
            self.y_pred = tf.argmax(self.fc1000, axis=-1, output_type=tf.int32, name='y_pred')


class FCN():
    def __init__(self, front_end, stride, num_classes, ignore=True):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.front = front_end
        self.X_input = self.front.X_input

        get_conv = self.front.cc.get_conv
        upsample = self.front.cc.get_deconv

        self.num_classes = num_classes
        image_height = self.front.image_height
        image_width = self.front.image_width

        # If there exists at least one label to be ignored, the masks for each image with identical shape are required.
        if ignore:
            self.y_mask_input = tf.placeholder(tf.float32, shape=[None, image_height, image_width], name='y_mask_input')

        # Overwrite `y_input` of the FCN network.
        self.y_input = tf.placeholder(tf.int32, shape=[None, None, None], name='y_input')

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.front.X_input, name='shape_input')
        self.output_shape = tf.stack([self.input_shape[0], self.input_shape[1], self.input_shape[2], num_classes],
                                     name='shape_output')

        self.score = get_conv('score', self.front.pool5, pretrained=False, bias=True, k_size=[1, 1],
                              num_output=num_classes)

        # skip-archiecture and layer fuse are carried out here.
        # Note that the shape of the output of a deconvolution layer is not a fixed value but a range, so a desired
        # output shape should
        if stride == 32:
            self.upsample32x = upsample('upsample32x', self.score, 32, self.output_shape)
        elif stride == 16:
            self.pool4_shape = tf.shape(self.front.pool4, name='shape_pool4')
            self.fuse2x_shape = tf.stack(
                [self.pool4_shape[0], self.pool4_shape[1], self.pool4_shape[2], num_classes],
                name='shape_fuse2x')
            self.upsample2x = upsample('upsample2x', self.score, 2, self.fuse2x_shape)
            self.score_pool4 = get_conv('score_pool4', self.front.pool4, False, False, True, k_size=[1, 1],
                                        num_output=num_classes)
            self.fuse2x = tf.add(self.upsample2x, self.score_pool4, name='fuse_2x')
            self.upsample32x = upsample('upsample32x', self.fuse2x, 16, self.output_shape)
        elif stride == 8:
            self.pool4_shape = tf.shape(self.front.pool4, name='shape_pool4')
            self.fuse2x_shape = tf.stack(
                [self.pool4_shape[0], self.pool4_shape[1], self.pool4_shape[2], num_classes],
                name='shape_fuse2x')
            self.pool3_shape = tf.shape(self.front.pool3)
            self.fuse4x_shape = tf.stack(
                [self.pool3_shape[0], self.pool3_shape[1], self.pool3_shape[2], num_classes])

            # 2x
            self.upsample2x = upsample('upsample2x', self.score, 2, self.fuse2x_shape)
            self.score_pool4 = get_conv('score_pool4', self.front.pool4, False, False, True, k_size=[1, 1],
                                        num_output=num_classes)
            self.fuse2x = tf.add(self.upsample2x, self.score_pool4, name='fuse_2x')

            # 4x
            self.upsample4x = upsample('upsample4x', self.fuse2x, 2, self.fuse4x_shape)
            self.score_pool3 = get_conv('score_pool3', self.front.pool3, False, False, True, k_size=[1, 1],
                                        num_output=num_classes)

            # 8x
            self.fuse4x = tf.add(self.upsample4x, self.score_pool3, name='fuse_4x')

            # 32x
            self.upsample32x = upsample('upsample32x', self.fuse4x, 8, self.output_shape)

        else:
            raise Exception('`stride` can only take on values from {32, 16, 8}.')

        self.logits = self.upsample32x
        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class PSPNet():
    def __init__(self, front_end, num_classes, ignore=True):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.front = front_end
        self.X_input = self.front.X_input
        self.ignore = ignore

        get_conv = self.front.cc.get_conv
        get_bn = self.front.cc.get_bn
        upsample = self.front.cc.bilinear
        avg_pooling = self.front.cc.get_avgpooling

        self.num_classes = num_classes

        image_height = front_end.image_height
        image_width = front_end.image_width


        # If there exists at least one label to be ignored, the masks for each image with identical shape are required.
        if ignore:
            self.y_mask_input = tf.placeholder(tf.bool, shape=[None, image_height, image_width], name='y_mask_input')

        # Overwrite `y_input` of the original classification network.
        self.y_input = tf.placeholder(tf.int32, shape=[None, None, None], name='y_input')

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.front.X_input, name='shape_input')
        self.output_shape = tf.stack([self.input_shape[0], self.input_shape[1], self.input_shape[2], num_classes],
                                     name='shape_output')

        # "ps" stands for "pyramid strides".
        if image_height == 473:
            ps = [60, 30, 20, 10]
        elif image_height == 713:
            ps = [90, 45, 30, 15]
        else:
            raise Exception('Invalid input image shape.')

        self.PSP_pool1 = avg_pooling('PSP_pool1', self.front.pool5, (ps[0], ps[0]), (ps[0], ps[0]))
        self.PSP_pool2 = avg_pooling('PSP_pool2', self.front.pool5, (ps[1], ps[1]), (ps[1], ps[1]))
        self.PSP_pool3 = avg_pooling('PSP_pool3', self.front.pool5, (ps[2], ps[2]), (ps[2], ps[2]))
        self.PSP_pool4 = avg_pooling('PSP_pool4', self.front.pool5, (ps[3], ps[3]), (ps[3], ps[3]))

        # number of feature maps of PSP module.
        nr_f = int(self.front.pool5.get_shape()[-1]) / 4

        def PSP_helper(name, bottom, pretrained, num_output):
            with tf.variable_scope(name):
                PSP_conv = get_conv('conv', bottom, pretrained=pretrained, num_output=num_output)
                PSP_bn = get_bn('bn', PSP_conv, pretrained)
                PSP_upsample = upsample('upsample', PSP_bn, output_shape=(ps[0], ps[0]))
                return PSP_upsample

        with tf.variable_scope('PSP_module'):
            self.PSP1 = PSP_helper('PSP1', self.PSP_pool1, False, nr_f)
            self.PSP2 = PSP_helper('PSP2', self.PSP_pool2, False, nr_f)
            self.PSP3 = PSP_helper('PSP3', self.PSP_pool3, False, nr_f)
            self.PSP4 = PSP_helper('PSP4', self.PSP_pool4, False, nr_f)

        with tf.variable_scope('post_stage'):
            self.concat = tf.concat([self.front.pool5, self.PSP1, self.PSP2, self.PSP3, self.PSP4], -1, 'PSP_concat')
            self.post_conv = get_conv('post_conv', self.concat, False, k_size=(3, 3), num_output=nr_f)
            self.post_bn = get_bn('post_bn', self.post_conv, False)
            self.post_dropout = tf.nn.dropout(self.post_bn, 0.9)
            self.reduce = get_conv('reduce', self.post_dropout, pretrained=False, bias=False, k_size=[1, 1],
                                   num_output=num_classes)
            self.logits = upsample('logits', self.reduce, output_shape=(image_height, image_width))

        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class Deeplabv2():
    def __init__(self, front_end, num_classes, ignore=True):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.front = front_end
        self.num_classes = num_classes
        self.ignore = ignore
        self.X_input = self.front.X_input
        self.input_shape = tf.shape(self.X_input)

        get_conv = self.front.cc.get_conv
        upsample = self.front.cc.bilinear

        self.num_classes = num_classes
        image_height = self.front.image_height
        image_width = self.front.image_width

        # If there exists at least one label to be ignored, the masks for each image with identical shape are required.
        if ignore:
            self.y_mask_input = tf.placeholder(tf.float32, shape=[None, image_height, image_width], name='y_mask_input')
        # `1.0` is the default parameter for `tf.losses.sparse_softmax_cross_entropy()`, indicating that no labels are
        # ignored.

        # Overwrite `y_input` of the original classification network.
        self.y_input = tf.placeholder(tf.int32, shape=[None, None, None], name='y_input')

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.front.X_input, name='shape_input')
        self.output_shape = tf.stack([self.input_shape[0], self.input_shape[1], self.input_shape[2], num_classes],
                                     name='shape_output')

        with tf.variable_scope('ASPP'):
            self.ASPP1 = get_conv('ASPP1', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=6)
            self.ASPP2 = get_conv('ASPP2', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=12)
            self.ASPP3 = get_conv('ASPP3', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=18)
            self.ASPP4 = get_conv('ASPP4', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=24)
            self.EXP = self.ASPP1 + self.ASPP2 + self.ASPP3 + self.ASPP4

        self.logits = upsample('logits', self.EXP, output_shape=self.input_shape[1:3])

        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')

# class FPN():
#     def __init__(self, front_name, num_classes, image_height=None, image_width=None, image_channel=3, ignore=True,
#                  upsample_mode=UPSAMPLE_BILINEAR, lateral_channel=256):
#         """
#         :param stride: specify the stride of the FCN network (FCN#{stride}s).
#         :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
#         """
#
#         self.front = get_front(front_name, image_height, image_width, image_channel)
#         self.X_input = self.front.X_input
#
#         get_conv = self.front.cc.get_conv
#         if upsample_mode == UPSAMPLE_BILINEAR:
#             upsample = self.front.cc.bilinear
#         elif upsample_mode == UPSAMPLE_DECONV:
#             upsample = self.front.cc.get_deconv
#         else:
#             raise Exception('invalid upsample mode provided.')
#
#         self.num_classes = num_classes
#
#         # If there exists at least one label to be ignored, the masks for each image with identical shape are required.
#         if ignore:
#             self.y_mask_input = tf.placeholder(tf.float32, shape=[None, image_height, image_width], name='y_mask_input')
#         # `1.0` is the default parameter for `tf.losses.sparse_softmax_cross_entropy()`, indicating that no labels are
#         # ignored.
#         else:
#             self.y_mask_input = 1.0
#
#         # Overwrite `y_input` of the FCN network.
#         self.y_input = tf.placeholder(tf.int32, shape=[None, None, None], name='y_input')
#
#         # The final output shape of the network, used to determine the output shape of the transpose convolution
#         # (deconvolution) layer.
#         self.input_shape = tf.shape(self.X_input, name='shape_input')
#         self.output_shape = tf.stack([self.input_shape[0], self.input_shape[1], self.input_shape[2], num_classes],
#                                      name='shape_output')
#
#         with tf.variable_scope('fuse1x'):
#             self.fuse1x = get_conv('lateral1x', self.front.pool5, pretrained=False, num_output=lateral_channel)
#
#         with tf.variable_scope('fuse2x'):
#             self.upsample2x = upsample('upsample2x', self.fuse1x, output_shape=tf.shape(self.front.pool4))
#             self.lateral2x = get_conv('lateral2x', self.front.pool4, pretrained=False, num_output=lateral_channel)
#             self.fuse2x = tf.add(self.upsample2x, self.lateral2x)
#
#         with tf.variable_scope('fuse4x'):
#             self.upsample4x = upsample('upsample4x', self.fuse2x, output_shape=tf.shape(self.front.pool3))
#             self.lateral4x = get_conv('lateral4x', self.front.pool3, pretrained=False, num_output=lateral_channel)
#             self.fuse4x = tf.add(self.upsample4x, self.lateral4x)
#
#         with tf.variable_scope('fuse8x'):
#             self.upsample8x = upsample('upsample8x', self.fuse4x, output_shape=tf.shape(self.front.pool2))
#             self.lateral8x = get_conv('lateral8x', self.front.pool2, pretrained=False, num_output=lateral_channel)
#             self.fuse8x = tf.add(self.upsample8x, self.lateral8x)
#
#         self.upsample32x = upsample('upsample32x', self.fuse8x, output_shape=tf.shape(self.front.X_input))
#         self.score = get_conv('score', self.upsample32x, pretrained=False, num_output=num_classes)
#
#         self.y_pred = tf.argmax(self.score, axis=-1, output_type=tf.int32, name='y_pred')
