import tensorflow as tf
from config import deeplab_batch_norm_decay, logger


class FCN():
    def __init__(self, encoder, stride, num_classes, is_training):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.encoder = encoder

        get_conv = self.encoder.cc.get_conv
        upsample = self.encoder.cc.bilinear

        self.num_classes = num_classes

        # If there exists at least one label to be ignored, the masks for each image with identical shape are required.

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        input_shape = tf.shape(self.encoder.X_input, name='shape_input')[1:3]

        self.score = get_conv('score', self.encoder.net, pretrained=False, bias=True, k_size=[1, 1],
                              num_outputs=num_classes)

        # skip-archiecture and layer fuse are carried out here.
        # Note that the shape of the output of a deconvolution layer is not a fixed value but a range, so a desired
        # output shape should
        if stride == 32:
            self.upsample32x = tf.image.resize_bilinear(self.score, input_shape, name='upsample32x')

        elif stride == 16 or stride == 8:
            f16_shape = tf.shape(self.encoder.f16, name='shape_f16')[1:3]
            self.upsample2x = tf.image.resize_bilinear(self.score, f16_shape, name='upsample2x')
            self.score_pool4 = get_conv('score_pool4', self.encoder.pool4, False, False, True, k_size=[1, 1],
                                        num_outputs=num_classes)
            self.fuse2x = tf.add(self.upsample2x, self.score_pool4, name='fuse_2x')

            if stride == 16:
                self.upsample32x = tf.image.resize_bilinear(self.fuse2x, input_shape, name='upsample32x')
            elif stride == 8:
                f8_shape = tf.shape(self.encoder.f8, name='shape_f8')[1:3]
                self.upsample4x = tf.image.resize_bilinear(self.fuse2x, f8_shape, name='upsample4x')
                self.score_pool3 = get_conv('score_pool3', self.encoder.pool3, False, False, True, k_size=[1, 1],
                                            num_outputs=num_classes)

                self.fuse4x = tf.add(self.upsample4x, self.score_pool3, name='fuse_4x')
                self.upsample32x = tf.image.resize_bilinear(self.fuse4x, input_shape, name='upsample32x')

        else:
            raise Exception('`stride` can only take on values from {32, 16, 8}.')

        self.logits = self.upsample32x
        self.y_prob = tf.nn.softmax(self.logits)
        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class UNet():
    def __init__(self, encoder, num_classes, is_training):
        self.encoder = encoder

        self.num_classes = num_classes

        # get_conv = self.front.cc.get_conv
        get_conv = lambda name, bottom, k_size, num_outputs: \
            self.encoder.cc.get_conv_bn(name, bottom, is_training=is_training, k_size=(1, 1), num_outputs=1)

        with tf.variable_scope('unpool_block2'):
            self.unpool2 = tf.concat(
                [self.encoder.f16, tf.image.resize_bilinear(self.encoder.f32, tf.shape(self.encoder.f16)[1:3])], -1)
            self.u_conv2_1 = get_conv('u_conv2_1', self.unpool2, k_size=(3, 3), num_outputs=512)
            self.u_conv2_2 = get_conv('u_conv2_2', self.u_conv2_1, k_size=(3, 3), num_outputs=512)

        with tf.variable_scope('unpool_block3'):
            self.unpool3 = tf.concat(
                [self.encoder.f8, tf.image.resize_bilinear(self.u_conv2_2, tf.shape(self.encoder.f8)[1:3])], -1)
            self.u_conv3_1 = get_conv('u_conv3_1', self.unpool3, k_size=(3, 3), num_outputs=256)
            self.u_conv3_2 = get_conv('u_conv3_2', self.u_conv3_1, k_size=(3, 3), num_outputs=256)

        with tf.variable_scope('unpool_block4'):
            self.unpool4 = tf.concat(
                [self.encoder.f4, tf.image.resize_bilinear(self.u_conv3_2, tf.shape(self.encoder.f4)[1:3])], -1)
            self.u_conv4_1 = get_conv('u_conv4_1', self.unpool4, k_size=(3, 3), num_outputs=128)
            self.u_conv4_2 = get_conv('u_conv4_2', self.u_conv4_1, k_size=(3, 3), num_outputs=128)

        with tf.variable_scope('unpool_block5'):
            self.unpool2 = tf.concat(
                [self.encoder.f2, tf.image.resize_bilinear(self.u_conv4_2, tf.shape(self.encoder.f2)[1:3])], -1)
            self.u_conv5_1 = get_conv('u_conv5_1', self.unpool2, k_size=(3, 3), num_outputs=64)
            self.u_conv5_2 = get_conv('u_conv5_2', self.u_conv5_1, k_size=(3, 3), num_outputs=64)

        with tf.variable_scope('score'):
            self.score = tf.image.resize_bilinear(self.u_conv5_2, tf.shape(self.encoder.X_input)[1:3])
            self.logits = get_conv('logits', self.score, k_size=(1, 1), num_outputs=num_classes)
            self.y_prob = tf.nn.softmax(self.logits)
            self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class PSPNet():
    def __init__(self, encoder, num_classes, is_training=True):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.encoder = encoder

        get_conv = self.encoder.cc.get_conv
        avg_pooling = self.encoder.cc.get_avgpooling
        get_bn = lambda name, bottom: self.encoder.cc.get_bn(name, bottom, momentum=deeplab_batch_norm_decay,
                                                             pretrained=False, relu=True, scale=False,
                                                             is_training=is_training)

        self.num_classes = num_classes

        image_height = encoder.image_height
        image_width = encoder.image_width

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.encoder.X_input, name='shape_input')
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

        self.PSP_pool1 = avg_pooling('PSP_pool1', self.encoder.f32, (ps[0], ps[0]), (ps[0], ps[0]))
        self.PSP_pool2 = avg_pooling('PSP_pool2', self.encoder.f32, (ps[1], ps[1]), (ps[1], ps[1]))
        self.PSP_pool3 = avg_pooling('PSP_pool3', self.encoder.f32, (ps[2], ps[2]), (ps[2], ps[2]))
        self.PSP_pool4 = avg_pooling('PSP_pool4', self.encoder.f32, (ps[3], ps[3]), (ps[3], ps[3]))

        # number of feature maps of PSP module.
        nr_f = int(self.encoder.pool5.get_shape()[-1]) / 4

        def PSP_helper(name, bottom, pretrained, num_outputs):
            with tf.variable_scope(name):
                PSP_conv = get_conv('conv', bottom, pretrained=pretrained, num_outputs=num_outputs)
                PSP_bn = get_bn('bn', PSP_conv)
                PSP_upsample = tf.image.resize_bilinear(PSP_bn, (ps[0], ps[0]), name='upsample')
                return PSP_upsample

        with tf.variable_scope('PSP_module'):
            self.PSP1 = PSP_helper('PSP1', self.PSP_pool1, False, nr_f)
            self.PSP2 = PSP_helper('PSP2', self.PSP_pool2, False, nr_f)
            self.PSP3 = PSP_helper('PSP3', self.PSP_pool3, False, nr_f)
            self.PSP4 = PSP_helper('PSP4', self.PSP_pool4, False, nr_f)

        with tf.variable_scope('post_stage'):
            self.concat = tf.concat([self.encoder.pool5, self.PSP1, self.PSP2, self.PSP3, self.PSP4], -1, 'PSP_concat')
            self.post_conv = get_conv('post_conv', self.concat, False, k_size=(3, 3), num_outputs=nr_f)
            self.post_bn = get_bn('post_bn', self.post_conv)
            # self.post_dropout = tf.nn.dropout(self.post_bn, 0.9)
            self.reduce = get_conv('reduce', self.post_bn, pretrained=False, bias=False, k_size=[1, 1],
                                   num_outputs=num_classes)
            self.logits = tf.image.resize_bilinear(self.reduce, (image_height, image_width), name='logits')

        self.y_prob = tf.nn.softmax(self.logits)
        self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')


class Deeplabv2():
    def __init__(self, encoder, num_classes, is_training):
        """
        :param stride: specify the stride of the FCN network (FCN#{stride}s).
        :param ignore: If there exists at least one label (class) to be ignored when calculating the loss.
        """

        self.encoder = encoder
        self.num_classes = num_classes
        self.X_input = self.encoder.X_input
        self.input_shape = tf.shape(self.X_input)

        get_conv = self.encoder.cc.get_conv
        upsample = self.encoder.cc.bilinear

        self.num_classes = num_classes

        # The final output shape of the network, used to determine the output shape of the transpose convolution
        # (deconvolution) layer.
        self.input_shape = tf.shape(self.encoder.X_input, name='shape_input')

        with tf.variable_scope('ASPP'):
            self.ASPP1 = get_conv('ASPP1', self.encoder.pool5, False, strides=1, k_size=(3, 3), atrous=6)
            self.ASPP2 = get_conv('ASPP2', self.encoder.pool5, False, strides=1, k_size=(3, 3), atrous=12)
            self.ASPP3 = get_conv('ASPP3', self.encoder.pool5, False, strides=1, k_size=(3, 3), atrous=18)
            self.ASPP4 = get_conv('ASPP4', self.encoder.pool5, False, strides=1, k_size=(3, 3), atrous=24)
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
#             self.fuse1x = get_conv('lateral1x', self.front.pool5, pretrained=False, num_outputs=lateral_channel)
#
#         with tf.variable_scope('fuse2x'):
#             self.upsample2x = upsample('upsample2x', self.fuse1x, output_shape=tf.shape(self.front.pool4))
#             self.lateral2x = get_conv('lateral2x', self.front.pool4, pretrained=False, num_outputs=lateral_channel)
#             self.fuse2x = tf.add(self.upsample2x, self.lateral2x)
#
#         with tf.variable_scope('fuse4x'):
#             self.upsample4x = upsample('upsample4x', self.fuse2x, output_shape=tf.shape(self.front.pool3))
#             self.lateral4x = get_conv('lateral4x', self.front.pool3, pretrained=False, num_outputs=lateral_channel)
#             self.fuse4x = tf.add(self.upsample4x, self.lateral4x)
#
#         with tf.variable_scope('fuse8x'):
#             self.upsample8x = upsample('upsample8x', self.fuse4x, output_shape=tf.shape(self.front.pool2))
#             self.lateral8x = get_conv('lateral8x', self.front.pool2, pretrained=False, num_outputs=lateral_channel)
#             self.fuse8x = tf.add(self.upsample8x, self.lateral8x)
#
#         self.upsample32x = upsample('upsample32x', self.fuse8x, output_shape=tf.shape(self.front.X_input))
#         self.score = get_conv('score', self.upsample32x, pretrained=False, num_outputs=num_classes)
#
#         self.y_pred = tf.argmax(self.score, axis=-1, output_type=tf.int32, name='y_pred')


# class Deeplabv3():
#     def __init__(self, encoder, num_classes, is_training, bn_scale):
#         self.encoder = encoder
#         input_shape = tf.shape(self.encoder.X_input)[1:3]
#         f32_shape = tf.shape(self.encoder.f32)[1:3]
#
#         get_conv = self.encoder.cc.get_conv
#         get_bn = lambda name, bottom: self.encoder.cc.get_bn(name, bottom, momentum=deeplab_batch_norm_decay,
#                                                              pretrained=False, relu=True, scale=bn_scale,
#                                                              is_training=is_training)
#
#         self.num_classes = num_classes
#         with tf.variable_scope('ASPP'):
#             # global_average_pooling
#             with tf.variable_scope('global_pooling'):
#                 self.global_pooling = tf.reduce_mean(self.encoder.f32, axis=[1, 2], keepdims=True, name='avg_pooling')
#                 self.global_pooling = get_conv('conv', self.global_pooling, False, False, False, k_size=[1, 1],
#                                                num_outputs=256)
#                 self.global_pooling = get_bn('bn', self.global_pooling)
#                 self.global_pooling = tf.image.resize_bilinear(self.global_pooling, f32_shape)
#             # ASPP
#             self.ASPP1 = get_conv('ASPP1', self.encoder.f32, pretrained=False, k_size=[1, 1], num_outputs=256)
#             self.ASPP2 = get_conv('ASPP2', self.encoder.f32, pretrained=False, k_size=[3, 3], num_outputs=256,
#                                   atrous=6)
#             self.ASPP3 = get_conv('ASPP3', self.encoder.f32, pretrained=False, k_size=[3, 3], num_outputs=256,
#                                   atrous=12)
#             self.ASPP4 = get_conv('ASPP4', self.encoder.f32, pretrained=False, k_size=[3, 3], num_outputs=256,
#                                   atrous=18)
#             # BN
#             self.ASPP1 = get_bn('ASPP1_bn', self.ASPP1)
#             self.ASPP2 = get_bn('ASPP2_bn', self.ASPP2)
#             self.ASPP3 = get_bn('ASPP3_bn', self.ASPP3)
#             self.ASPP4 = get_bn('ASPP4_bn', self.ASPP4)
#
#         with tf.variable_scope('ASPP_reduce'):
#             # concate
#             self.ASPP_concat = tf.concat([self.ASPP1, self.ASPP2, self.ASPP3, self.ASPP4, self.global_pooling], -1)
#             # 1*1 conv
#             self.reduce = get_conv('pre_reduce', self.ASPP_concat, pretrained=False, bias=False, k_size=[1, 1],
#                                    num_outputs=256)
#             self.reduce = get_bn('pre_reduce_bn', self.reduce)
#             self.reduce = get_conv('final_reduce', self.reduce, pretrained=False, bias=False, k_size=[1, 1],
#                                    num_outputs=num_classes)
#             # upsample
#             self.logits = tf.image.resize_bilinear(self.reduce, input_shape, name='logits')
#
#             self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')
#             self.y_prob = tf.nn.softmax(self.logits)


class Deeplabv3():
    def __init__(self, encoder, num_classes, is_training, bn_scale, plus=True, ASPP_rates=(6, 12, 18)):
        logger.info(f'Using Deeplabv3 ASPP rates {ASPP_rates}. Skip layer={plus}.')
        self.encoder = encoder
        input_shape = tf.shape(self.encoder.X_input)[1:3]
        f32_shape = tf.shape(self.encoder.f32)[1:3]
        f4_shape = tf.shape(self.encoder.f4)[1:3]

        get_conv = self.encoder.cc.get_conv
        get_bn = lambda name, bottom: self.encoder.cc.get_bn(name, bottom, momentum=deeplab_batch_norm_decay,
                                                             pretrained=False, relu=True, scale=bn_scale,
                                                             is_training=is_training)

        self.num_classes = num_classes
        with tf.variable_scope('ASPP'):
            # global_average_pooling
            with tf.variable_scope('global_pooling'):
                self.global_pooling = tf.reduce_mean(self.encoder.f32, axis=[1, 2], keepdims=True, name='avg_pooling')
                self.global_pooling = get_conv('conv', self.global_pooling, False, False, False, k_size=[1, 1],
                                               num_outputs=256)
                self.global_pooling = get_bn('bn', self.global_pooling)
                self.global_pooling = tf.image.resize_bilinear(self.global_pooling, f32_shape)
            # ASPP
            self.ASPP1 = get_conv('ASPP1', self.encoder.f32, pretrained=False, k_size=[1, 1], num_outputs=256)
            self.ASPP2 = get_conv('ASPP2', self.encoder.f32, pretrained=False, k_size=[3, 3], num_outputs=256,
                                  atrous=ASPP_rates[0])
            self.ASPP3 = get_conv('ASPP3', self.encoder.f32, pretrained=False, k_size=[3, 3], num_outputs=256,
                                  atrous=ASPP_rates[1])
            self.ASPP4 = get_conv('ASPP4', self.encoder.f32, pretrained=False, k_size=[3, 3], num_outputs=256,
                                  atrous=ASPP_rates[2])
            # BN
            self.ASPP1 = get_bn('ASPP1_bn', self.ASPP1)
            self.ASPP2 = get_bn('ASPP2_bn', self.ASPP2)
            self.ASPP3 = get_bn('ASPP3_bn', self.ASPP3)
            self.ASPP4 = get_bn('ASPP4_bn', self.ASPP4)

        if plus:
            with tf.variable_scope('low_level'):
                stride4 = get_conv('stride_4', self.encoder.f4, pretrained=False, k_size=[1, 1], num_outputs=48)
                self.stride4 = get_bn('stride_4_bn', stride4)

        with tf.variable_scope('ASPP_reduce'):
            # concate
            self.ASPP_concat = tf.concat([self.ASPP1, self.ASPP2, self.ASPP3, self.ASPP4, self.global_pooling], -1)
            # 1*1 conv
            self.ASPP_reduce = get_conv('pre_reduce', self.ASPP_concat, pretrained=False, bias=False, k_size=[1, 1],
                                        num_outputs=256)
            self.ASPP_reduce = get_bn('pre_reduce_bn', self.ASPP_reduce)

            if plus:
                self.ASPP_reduce = tf.image.resize_bilinear(self.ASPP_reduce, f4_shape, name='ASPP_reduce')

                stride4_fuse = tf.concat([self.ASPP_reduce, self.stride4], -1)

                stride4_fuse = get_conv('stride_4_fuser_conv1', stride4_fuse, pretrained=False, k_size=[3, 3],
                                        num_outputs=256)
                stride4_fuse = get_bn('stride_4_fuser_conv1_bn', stride4_fuse)
                stride4_fuse = get_conv('stride_4_fuser_conv2', stride4_fuse, pretrained=False, k_size=[3, 3],
                                        num_outputs=256)
                self.ASPP_reduce = get_bn('stride_4_fuser_conv2_bn', stride4_fuse)
            self.reduce = get_conv('final_reduce', self.ASPP_reduce, pretrained=False, bias=False, k_size=[1, 1],
                                   num_outputs=num_classes)
            # upsample
            self.logits = tf.image.resize_bilinear(self.reduce, input_shape, name='logits')

            self.y_pred = tf.argmax(self.logits, axis=-1, output_type=tf.int32, name='y_pred')
            self.y_prob = tf.nn.softmax(self.logits)


class SimpleDeconv():
    def __init__(self, encoder, num_classes, is_training, bn_scale, fuse):
        logger.info(f'Simple Deconv with fuse: {fuse}')
        self.encoder = encoder
        input_shape = tf.shape(self.encoder.X_input)[1:3]
        f2_shape = tf.shape(self.encoder.f2)[1:3]
        f4_shape = tf.shape(self.encoder.f4)[1:3]
        f8_shape = tf.shape(self.encoder.f8)[1:3]

        get_conv = self.encoder.cc.get_conv
        get_bn = lambda name, bottom: self.encoder.cc.get_bn(name, bottom, momentum=deeplab_batch_norm_decay,
                                                             pretrained=False, relu=True, scale=bn_scale,
                                                             is_training=is_training)

        with tf.variable_scope('Simple_Deconv'):
            self.unpool1 = tf.image.resize_bilinear(self.encoder.net, f8_shape, name='unpool1')
            if 8 in fuse:
                logger.info('Simple Deconv: fuse 8')
                self.unpool1 = tf.concat([self.unpool1, self.encoder.f8], axis=-1, name='concat_unpool1')
            self.deconv1 = get_conv('deconv1', self.unpool1, pretrained=False, k_size=(7, 7), num_outputs=64)
            self.deconv1 = get_bn('deconv1_bn', self.deconv1)

            self.unpool2 = tf.image.resize_bilinear(self.deconv1, f4_shape, name='unpool2')
            if 4 in fuse:
                logger.info('Simple Deconv: fuse 4')
                self.unpool2 = tf.concat([self.unpool2, self.encoder.f4], axis=-1, name='concat_unpool2')
            self.deconv2 = get_conv('deconv2', self.unpool2, pretrained=False, k_size=(7, 7), num_outputs=64)
            self.deconv2 = get_bn('deconv2_bn', self.deconv2)

            self.unpool3 = tf.image.resize_bilinear(self.deconv2, f2_shape, name='unpool3')
            if 2 in fuse:
                logger.info('Simple Deconv: fuse 2')
                self.unpool3 = tf.concat([self.unpool3, self.encoder.f2], axis=-1, name='concat_unpool3')
            self.deconv3 = get_conv('deconv3', self.unpool3, pretrained=False, k_size=(7, 7), num_outputs=64)
            self.deconv3 = get_bn('deconv3_bn', self.deconv3)

            self.unpool4 = tf.image.resize_bilinear(self.deconv3, input_shape, name='unpool4')
            self.deconv4 = get_conv('deconv4', self.unpool4, pretrained=False, k_size=(7, 7), num_outputs=64)
            self.deconv4 = get_bn('deconv4_bn', self.deconv4)

            self.logits = get_conv('logits', self.deconv4, pretrained=False, k_size=(7, 7), num_outputs=num_classes)
