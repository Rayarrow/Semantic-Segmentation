import tensorflow as tf
from config import batch_norm_decay, logger


class FCN():
    def __init__(self, front_end, stride, num_classes, is_training):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.front = front_end

        get_conv = self.front.cc.get_conv
        upsample = self.front.cc.bilinear

        self.num_classes = num_classes

        # If there exists at least one label to be ignored, the masks for each image with identical shape are required.

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.front.X_input, name='shape_input')
        self.output_shape = tf.stack([self.input_shape[0], self.input_shape[1], self.input_shape[2], num_classes],
                                     name='shape_output')

        self.score = get_conv('score', self.front.f32, pretrained=False, bias=True, k_size=[1, 1],
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
            self.upsample2x = upsample('upsample2x', self.score, None, self.fuse2x_shape)
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
        self.y_prob = tf.nn.softmax(self.logits)
        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class UNet():
    def __init__(self, front_end, num_classes):
        self.front = front_end

        self.num_classes = num_classes

        upsample = self.front.cc.bilinear
        get_conv = self.front.cc.get_conv

        with tf.variable_scope('d_block2'):
            self.unpool2 = tf.concat(
                [self.front.f4, upsample('upsample', self.front.f5, output_shape=tf.shape(self.front.f4))], -1)
            self.u_conv2_1 = get_conv('u_conv2_1', self.unpool2, False, True, True, k_size=(3, 3), num_output=512)
            self.u_conv2_2 = get_conv('u_conv2_2', self.u_conv2_1, False, True, True, k_size=(3, 3), num_output=512)

        with tf.variable_scope('d_block3'):
            self.unpool3 = tf.concat(
                [self.front.f3, upsample('upsample', self.u_conv2_2, output_shape=tf.shape(self.front.f3))], -1)
            self.u_conv3_1 = get_conv('u_conv3_1', self.unpool3, False, True, True, k_size=(3, 3), num_output=256)
            self.u_conv3_2 = get_conv('u_conv3_2', self.u_conv3_1, False, True, True, k_size=(3, 3), num_output=256)

        with tf.variable_scope('d_block4'):
            self.unpool4 = tf.concat(
                [self.front.f2, upsample('upsample', self.u_conv3_2, output_shape=tf.shape(self.front.f2))], -1)
            self.u_conv4_1 = get_conv('u_conv4_1', self.unpool4, False, True, True, k_size=(3, 3), num_output=128)
            self.u_conv4_2 = get_conv('u_conv4_2', self.u_conv4_1, False, True, True, k_size=(3, 3), num_output=128)

        with tf.variable_scope('d_block5'):
            self.unpool5 = tf.concat(
                [self.front.f1, upsample('upsample', self.u_conv4_2, output_shape=tf.shape(self.front.f1))], -1)
            self.u_conv5_1 = get_conv('u_conv5_1', self.unpool5, False, True, True, k_size=(3, 3), num_output=64)
            self.u_conv5_2 = get_conv('u_conv5_2', self.u_conv5_1, False, True, True, k_size=(3, 3), num_output=64)

        with tf.variable_scope('score'):
            self.logits = get_conv('logits', self.u_conv5_2, False, True, True, k_size=(1, 1), num_output=num_classes)
            self.y_prob = tf.nn.softmax(self.logits)
            self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class PSPNet():
    def __init__(self, front_end, num_classes, is_training=None):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.front = front_end

        get_conv = self.front.cc.get_conv
        get_bn = self.front.cc.get_bn
        avg_pooling = self.front.cc.get_avgpooling

        self.num_classes = num_classes

        image_height = front_end.image_height
        image_width = front_end.image_width

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.front.X_input, name='shape_input')
        self.output_shape = tf.stack([self.input_shape[0], self.input_shape[1], self.input_shape[2], num_classes],
                                     name='shape_output')

        # "ps" stands for "pyramid strides".
        logger.info(f'Image shape: {(image_height, image_width)}')
        if image_height == 473:
            ps = [60, 30, 20, 10]
        elif image_height == 713:
            ps = [90, 45, 30, 15]
        else:
            raise Exception('Invalid input image shape. Resize the imput shape to 473 or 713.')

        self.PSP_pool1 = avg_pooling('PSP_pool1', self.front.f32, (ps[0], ps[0]), (ps[0], ps[0]))
        self.PSP_pool2 = avg_pooling('PSP_pool2', self.front.f32, (ps[1], ps[1]), (ps[1], ps[1]))
        self.PSP_pool3 = avg_pooling('PSP_pool3', self.front.f32, (ps[2], ps[2]), (ps[2], ps[2]))
        self.PSP_pool4 = avg_pooling('PSP_pool4', self.front.f32, (ps[3], ps[3]), (ps[3], ps[3]))

        # number of feature maps of PSP module.
        nr_f = int(self.front.pool5.get_shape()[-1]) / 4

        def PSP_helper(name, bottom, pretrained, num_output):
            with tf.variable_scope(name):
                PSP_conv = get_conv('conv', bottom, pretrained=pretrained, num_output=num_output)
                PSP_bn = get_bn('bn', PSP_conv, pretrained, is_training=is_training)
                PSP_upsample = tf.image.resize_bilinear(PSP_bn, (ps[0], ps[0]), name='upsample')
                return PSP_upsample

        with tf.variable_scope('PSP_module'):
            self.PSP1 = PSP_helper('PSP1', self.PSP_pool1, False, nr_f)
            self.PSP2 = PSP_helper('PSP2', self.PSP_pool2, False, nr_f)
            self.PSP3 = PSP_helper('PSP3', self.PSP_pool3, False, nr_f)
            self.PSP4 = PSP_helper('PSP4', self.PSP_pool4, False, nr_f)

        with tf.variable_scope('post_stage'):
            self.concat = tf.concat([self.front.pool5, self.PSP1, self.PSP2, self.PSP3, self.PSP4], -1, 'PSP_concat')
            self.post_conv = get_conv('post_conv', self.concat, False, k_size=(3, 3), num_output=nr_f)
            self.post_bn = get_bn('post_bn', self.post_conv, False, is_training=is_training)
            # self.post_dropout = tf.nn.dropout(self.post_bn, 0.9)
            self.reduce = get_conv('reduce', self.post_bn, pretrained=False, bias=False, k_size=[1, 1],
                                   num_output=num_classes)
            self.logits = tf.image.resize_bilinear(self.reduce, (image_height, image_width), name='logits')

        self.y_prob = tf.nn.softmax(self.logits)
        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class Deeplabv2():
    def __init__(self, front_end, num_classes, is_training):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.front = front_end
        self.num_classes = num_classes
        self.X_input = self.front.X_input
        self.input_shape = tf.shape(self.X_input)

        get_conv = self.front.cc.get_conv
        upsample = self.front.cc.bilinear

        self.num_classes = num_classes

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.front.X_input, name='shape_input')

        with tf.variable_scope('ASPP'):
            self.ASPP1 = get_conv('ASPP1', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=6)
            self.ASPP2 = get_conv('ASPP2', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=12)
            self.ASPP3 = get_conv('ASPP3', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=18)
            self.ASPP4 = get_conv('ASPP4', self.front.pool5, False, strides=1, k_size=(3, 3), atrous=24)
            self.EXP = self.ASPP1 + self.ASPP2 + self.ASPP3 + self.ASPP4

        self.logits = upsample('logits', self.EXP, output_shape=self.input_shape[1:3])

        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')
        self.y_prob = tf.nn.softmax(self.logits)


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


class Deeplabv3():
    def __init__(self, front_end, num_classes, is_training):
        from tensorflow.contrib.slim.nets import resnet_v2
        from tensorflow.contrib import layers as layers_lib
        from tensorflow.contrib.framework.python.ops import arg_scope
        from tensorflow.contrib.layers.python.layers import layers

        self.front = front_end
        input_shape = tf.shape(self.front.X_input)

        get_conv = self.front.cc.get_conv
        upsample = self.front.cc.bilinear
        get_bn = self.front.cc.get_bn

        self.num_classes = num_classes
        # with tf.variable_scope("aspp"):
        #     # if output_stride not in [8, 16]:
        #     #     raise ValueError('output_stride must be either 8 or 16.')
        #     #
        #     atrous_rates = [6, 12, 18]
        #     inputs = self.front.f32
        #     depth = 256
        #
        #     # if output_stride == 8:
        #     #     atrous_rates = [2 * rate for rate in atrous_rates]
        #
        #     with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)):
        #         with arg_scope([layers.batch_norm], is_training=is_training):
        #             inputs_size = tf.shape(inputs)[1:3]
        #             # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
        #             # the rates are doubled when output stride = 8.
        #             conv_1x1 = layers_lib.conv2d(inputs, depth, [1, 1], stride=1, scope="conv_1x1")
        #             conv_3x3_1 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[0],
        #                                            scope='conv_3x3_1')
        #             conv_3x3_2 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[1],
        #                                            scope='conv_3x3_2')
        #             conv_3x3_3 = layers_lib.conv2d(inputs, depth, [3, 3], stride=1, rate=atrous_rates[2],
        #                                            scope='conv_3x3_3')
        #
        #             # (b) the image-level features
        #             with tf.variable_scope("image_level_features"):
        #                 # global average pooling
        #                 image_level_features = tf.reduce_mean(inputs, [1, 2], name='global_average_pooling',
        #                                                       keepdims=True)
        #                 # 1x1 convolution with 256 filters( and batch normalization)
        #                 image_level_features = layers_lib.conv2d(image_level_features, depth, [1, 1], stride=1,
        #                                                          scope='conv_1x1')
        #                 # bilinearly upsample features
        #                 image_level_features = tf.image.resize_bilinear(image_level_features, inputs_size,
        #                                                                 name='upsample')
        #
        #             net = tf.concat([conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features], axis=3,
        #                             name='concat')
        #             net = layers_lib.conv2d(net, depth, [1, 1], stride=1, scope='conv_1x1_concat')
        #
        # reduce = layers_lib.conv2d(net, num_classes, [1, 1], scope='reduce')
        # # upsample
        # self.logits = tf.image.resize_bilinear(reduce, (input_shape[1], input_shape[2]), name='logits')
        #
        # self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')
        # self.y_prob = tf.nn.softmax(self.logits)

        with tf.variable_scope('ASPP'):
            # global_average_pooling
            with tf.variable_scope('global_pooling'):
                self.global_pooling = tf.reduce_mean(self.front.f32, axis=[1, 2], keepdims=True, name='avg_pooling')
                self.global_pooling = get_conv('conv', self.global_pooling, False, False, False, k_size=[1, 1],
                                               num_output=256)
                self.global_pooling = get_bn('bn', self.global_pooling, pretrained=False, is_training=is_training)
                self.global_pooling = upsample('upsample', self.global_pooling, output_shape=tf.shape(self.front.f32))
            # ASPP
            self.ASPP1 = get_conv('ASPP1', self.front.f32, pretrained=False, k_size=[1, 1], num_output=256)
            self.ASPP2 = get_conv('ASPP2', self.front.f32, pretrained=False, k_size=[3, 3], num_output=256,
                                  atrous=6)
            self.ASPP3 = get_conv('ASPP3', self.front.f32, pretrained=False, k_size=[3, 3], num_output=256,
                                  atrous=12)
            self.ASPP4 = get_conv('ASPP4', self.front.f32, pretrained=False, k_size=[3, 3], num_output=256,
                                  atrous=18)
            # BN
            self.ASPP1 = get_bn('ASPP1_bn', self.ASPP1, pretrained=False, is_training=is_training)
            self.ASPP2 = get_bn('ASPP2_bn', self.ASPP2, pretrained=False, is_training=is_training)
            self.ASPP3 = get_bn('ASPP3_bn', self.ASPP3, pretrained=False, is_training=is_training)
            self.ASPP4 = get_bn('ASPP4_bn', self.ASPP4, pretrained=False, is_training=is_training)

        with tf.variable_scope('post_conv'):
            # concate
            self.ASPP_concat = tf.concat([self.ASPP1, self.ASPP2, self.ASPP3, self.ASPP4, self.global_pooling], -1)
            # 1*1 conv
            self.reduce = get_conv('pre_reduce', self.ASPP_concat, pretrained=False, bias=False, k_size=[1, 1],
                                   num_output=256)
            self.reduce = get_bn('pre_reduce_bn', self.reduce, pretrained=False, is_training=is_training)
            self.reduce = get_conv('final_reduce', self.reduce, pretrained=False, bias=False, k_size=[1, 1],
                                   num_output=num_classes)
            # upsample
            self.logits = tf.image.resize_bilinear(self.reduce, (input_shape[1], input_shape[2]), name='logits')

            self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')
            self.y_prob = tf.nn.softmax(self.logits)


class deeplab_v3_plus():
    def __init__(self, front_end, num_classes):
        self.front = front_end
        self.X_input = self.front.X_input
        self.y_input = self.front.y_input

        get_conv = self.front.cc.get_conv
        upsample = self.front.cc.bilinear
        avg_pooling = self.front.cc.get_avgpooling
        get_bn = self.front.cc.get_bn

        image_height = front_end.image_height
        image_width = front_end.image_width

        self.num_classes = num_classes

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.front.X_input, name='shape_input')
        self.output_shape = tf.stack([self.input_shape[0], self.input_shape[1], self.input_shape[2], num_classes],
                                     name='shape_output')

        # global_average_pooling
        with tf.variable_scope('ASPP'):
            # self.fc_image_pooling = avg_pooling('image_pooling', self.front.pool5, (60, 60), (60, 60))
            self.global_pooling = tf.reduce_mean(self.front.pool5, axis=[1, 2], keepdim=True, name='global_pooling')
            self.global_pooling = get_conv('global_pooling_conv', self.global_pooling, False, False, False,
                                           k_size=[1, 1], num_output=256)
            self.global_pooling = get_bn('global_pooling_bn', self.global_pooling, pretrained=False)
            self.global_pooling = upsample('global_pooling_upsample', self.global_pooling,
                                           output_shape=tf.shape(self.front.pool5))
            # ASPP
            self.ASPP1 = get_conv('ASPP1', self.front.pool5, pretrained=False, k_size=[1, 1], num_output=256)
            self.ASPP2 = get_conv('ASPP2', self.front.pool5, pretrained=False, k_size=[3, 3], num_output=256, atrous=6)
            self.ASPP3 = get_conv('ASPP3', self.front.pool5, pretrained=False, k_size=[3, 3], num_output=256, atrous=12)
            self.ASPP4 = get_conv('ASPP4', self.front.pool5, pretrained=False, k_size=[3, 3], num_output=256, atrous=18)
            # BN
            self.ASPP1 = get_bn('ASPP1_bn', self.ASPP1, pretrained=False)
            self.ASPP2 = get_bn('ASPP2_bn', self.ASPP2, pretrained=False)
            self.ASPP3 = get_bn('ASPP3_bn', self.ASPP3, pretrained=False)
            self.ASPP4 = get_bn('ASPP4_bn', self.ASPP4, pretrained=False)

        with tf.variable_scope('concatenation'):
            # concate
            self.ASPP_concat = tf.concat([self.ASPP1, self.ASPP2, self.ASPP3, self.ASPP4, self.global_pooling], 3)
            # 1*1 conv
        self.reduce = get_conv('final_reduce', self.ASPP_concat, pretrained=False, bias=False, k_size=[1, 1],
                               num_output=num_classes)
        self.reduce_up

        with tf.variable_scope('concate'):
            # concate bottom and skip layer
            self.skip_conv = get_conv('skip_conv', self.front.skip, pretrained=False, k_size=[1, 1], num_output=128)
            self.skip_conv = upsample('skip_conv', self.skip_conv, output_shape=(149, 149))
            self.skip_bn = get_bn('skip_bn', self.skip_conv, pretrained=False)
            self.fc = tf.concat((self.fc_c0, self.fc_c1), 3)
            self.fc = tf.concat((self.fc, self.fc_c2), 3)
            self.fc = tf.concat((self.fc, self.fc_c3), 3)
            self.fc = tf.concat((self.fc, self.fc_image_pooling), 3)
            print(self.skip_bn.shape)
            self.fc_upsampled_4 = upsample('upsample1', self.fc, output_shape=(149, 149))
            self.fc = tf.concat((self.fc_upsampled_4, self.skip_bn), 3)
            # 1*1 conv
            self.reduce = get_conv('reduce', self.fc, pretrained=False, bias=False, k_size=[1, 1],
                                   num_output=num_classes)
            # upsample
            self.logits = upsample('score', self.reduce, output_shape=self.input_shape[1:3])

        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')
        self.y_prob = tf.nn.softmax(self.logits)
