import numpy as np
import tensorflow.contrib.slim as slim

from config import *

# from deform_conv import *
# from msra_initializer import *

UPSAMPLE_BILINEAR = 'bilinear'
UPSAMPLE_DECONV = 'deconv'


class component_constructor():
    def __init__(self, weights_path, weights_idx='weights', bias_idx='biases',
                 bn_idx=('moving_mean', 'moving_variance', 'beta', 'gamma')):
        logger.info('loadding {} ...'.format(weights_path))
        if weights_path is not None:
            self.weights_dict = np.load(weights_path, encoding='latin1').item()
        else:
            self.weights_dict = None

        self.weights_idx = weights_idx
        self.bias_idx = bias_idx
        self.bn_idx = bn_idx

    def get_conv(self, name, bottom, pretrained=True, relu=False, bias=False, strides=(1, 1, 1, 1), k_size=(1, 1),
                 num_outputs=1, atrous=False):
        # Use this to determine the third dimension of W.
        num_input = bottom.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            # Load pretrained weights if they exist.
            # W_init = tf.constant_initializer(self.weights_dict[name][self.weights_idx]) if pretrained else tf.contrib.layers.xavier_initializer()
            W_init = tf.constant_initializer(self.weights_dict[name][self.weights_idx]) if pretrained else None
            W_shape = self.weights_dict[name][self.weights_idx].shape if pretrained else list(k_size) + [num_input,
                                                                                                         num_outputs]

            strides = atrous if atrous else strides
            W = tf.get_variable('W', shape=W_shape, dtype=tf.float32, initializer=W_init)
            conv_op = tf.nn.atrous_conv2d if atrous else tf.nn.conv2d
            conv_name = 'atrous_conv' if atrous else 'conv2d'
            conv = conv_op(bottom, W, strides, padding='SAME', name=conv_name)

            if bias:
                # b_init = tf.constant_initializer(self.weights_dict[name][self.bias_idx]) if pretrained else tf.contrib.layers.xavier_initializer()
                b_init = tf.constant_initializer(self.weights_dict[name][self.bias_idx]) if pretrained else None
                b_shape = self.weights_dict[name][self.bias_idx].shape if pretrained else num_outputs
                b = tf.get_variable('b', shape=b_shape, dtype=tf.float32, initializer=b_init)
                conv = conv + b

            return tf.nn.relu(conv) if relu else conv

    def get_bn(self, name, bottom, momentum=0.9997, pretrained=True, relu=True, scale=False, is_training=True):
        with tf.variable_scope(name):
            bn = tf.layers.batch_normalization(bottom, momentum=momentum, scale=scale,
                                               moving_mean_initializer=tf.constant_initializer(
                                                   self.weights_dict[name][self.bn_idx[0]]),
                                               moving_variance_initializer=tf.constant_initializer(
                                                   self.weights_dict[name][self.bn_idx[1]]),
                                               beta_initializer=tf.constant_initializer(
                                                   self.weights_dict[name][self.bn_idx[2]]),
                                               gamma_initializer=tf.constant_initializer(
                                                   self.weights_dict[name][self.bn_idx[3]]),
                                               training=is_training) if pretrained else tf.layers.batch_normalization(
                bottom, momentum=momentum, scale=scale, training=is_training)
            return tf.nn.relu(bn) if relu else bn

    def get_conv_bn(self, name, bottom, pretrained=False, relu=(False, True), bias=False, scale=False, momentum=0.9997,
                    is_training=True, strides=(1, 1, 1, 1), k_size=(1, 1), num_outputs=1, atrous=False):
        """
        get both conv and bn in a single run.
        """
        bottom = self.get_conv(name, bottom, pretrained, relu[0], bias, strides, k_size, num_outputs, atrous)
        bottom = self.get_bn(f'{name}_bn', bottom, momentum, pretrained, relu[1], scale, is_training)
        return bottom

    '''
    def get_bottleneck(self, bottom, block_idx, nr_blocks, nr_layers=3, conv_pretrained=True, bn_pretrained=True,
                       pooling=None, atrous=False, rates=None):
        # the default block strides.
        block_strides = [[[1, 1, 1, 1] for _ in range(nr_layers)] for _ in range(nr_blocks)]
        # change the stride of the first layer of the first block.
        if pooling:
            block_strides[pooling[0]][pooling[1]] = [1, 2, 2, 1]

        if atrous:
            if not rates:
                raise Exception('for atrous convolution, `rates` must be specified.')
            # Only the middle 3*3 conv layer can perform atrous convolution.
            atrous = [[False, True, False]] * nr_blocks

            # If the input `rates` is an integer, then all the rates of every block are `rates`.
            if isinstance(rates, int):
                rates = [rates] * nr_blocks
            # Note that under atrous convolution conditions, the middle element in each nested list within
            # `block_stride` # is the rate (an integer) for the atrous convolution instead of the actual strides
            # (a list) for standard the convolution.
            for idx, each_rate in enumerate(rates):
                block_strides[idx][1] = each_rate
        else:
            atrous = [[False, False, False]] * nr_blocks

        res_pattern = 'res{}{}_branch{}{}'
        bn_pattern = 'bn{}{}_branch{}{}'
        skip_stride = (1, 2, 2, 1) if pooling else (1, 1, 1, 1)
        res_skip = self.get_conv(res_pattern.format(block_idx, 'a', 1, ''), bottom, strides=skip_stride)
        bn_skip = self.get_bn(bn_pattern.format(block_idx, 'a', 1, ''), res_skip)

        for block in range(nr_blocks):
            for layer in range(nr_layers):
                bottom = self.get_conv(res_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)),
                                       bottom, conv_pretrained, strides=block_strides[block][layer],
                                       atrous=atrous[block][layer])
                bottom = self.get_bn(bn_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)), bottom,
                                     bn_pretrained)
            bottom = self.add_relu(bn_skip, bottom, 'fuse{}{}'.format(block_idx, chr(97 + block)))
            bn_skip = bottom
        return bottom
    '''

    def get_bottleneck(self, bottom, block_idx, nr_blocks, nr_layers=3, conv_pretrained=True, bn_pretrained=True,
                       pooling=None, atrous=False, rates=None, deformable=False):
        # the default block strides.
        block_strides = [[[1, 1, 1, 1] for _ in range(nr_layers)] for _ in range(nr_blocks)]
        # change the stride of the first layer of the first block of this bottleneck.
        if pooling:
            block_strides[pooling[0]][pooling[1]] = [1, 2, 2, 1]

        if atrous:
            if not rates:
                raise Exception('for atrous convolution, `rates` must be specified.')
            # Only the middle 3*3 conv layer can perform atrous convolution.
            atrous = [[False, True, False]] * nr_blocks

            # If the input `rates` is an integer, then all the rates of every block are `rates`.
            if isinstance(rates, int):
                rates = [rates] * nr_blocks
            # Note that under atrous convolution conditions, the middle element in each nested list within
            # `block_stride` # is the rate (an integer) for the atrous convolution instead of the actual strides
            # (a list) for standard the convolution.
            for idx, each_rate in enumerate(rates):
                block_strides[idx][1] = each_rate

            if len(rates) != 1:
                atrous = []
                for i in range(len(rates)):
                    atrous += [[False, rates[i], False]]

        else:
            atrous = [[False, False, False]] * nr_blocks

        res_pattern = 'res{}{}_branch{}{}'
        bn_pattern = 'bn{}{}_branch{}{}'
        deform_pattern = 'deform{}{}_branch{}{}'
        conv1_pattern = 'conv1{}{}_branch{}{}'
        conv2_pattern = 'conv2{}{}_branch{}{}'

        deform = [False, False, True]

        skip_stride = (1, 2, 2, 1) if pooling else (1, 1, 1, 1)
        res_skip = self.get_conv(res_pattern.format(block_idx, 'a', 1, ''), bottom, strides=skip_stride)
        bn_skip = self.get_bn(bn_pattern.format(block_idx, 'a', 1, ''), res_skip)

        '''
        for block in range(nr_blocks):
            for layer in range(nr_layers):
                bottom = self.get_conv(res_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)),
                                       bottom,conv_pretrained, relu=relu[layer],  strides=block_strides[block][layer],
                                       atrous=atrous[block][layer])
                if deformable and deform[layer]:
                    bottom = self.deformable_conv(bottom, deform_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)))
                bottom = self.get_bn(bn_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)), bottom,
                                     bn_pretrained)
            bottom = self.add_relu(bn_skip, bottom, 'fuse{}{}'.format(block_idx, chr(97 + block)))
            bn_skip = bottom       
            '''
        for block in range(nr_blocks):
            for layer in range(nr_layers):
                bottom = self.get_conv(res_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)),
                                       bottom, conv_pretrained, strides=block_strides[block][layer],
                                       atrous=atrous[block][layer])
                bottom = self.get_bn(bn_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)), bottom,
                                     bn_pretrained)
                if deformable and deform[layer]:
                    kernel_num = bottom.shape[3]
                    debottom = self.get_conv(conv1_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)),
                                             bottom, pretrained=False, num_outputs=512)
                    debottom = self.deformable_conv(debottom, deform_pattern.format(block_idx, chr(97 + block), 2,
                                                                                    chr(97 + layer)))
                    concatebottom = tf.concat((bottom, debottom), 3)
                    bottom = self.get_conv(conv2_pattern.format(block_idx, chr(97 + block), 2, chr(97 + layer)),
                                           concatebottom, pretrained=False, num_outputs=kernel_num)
            bottom = self.add_relu(bn_skip, bottom, 'fuse{}{}'.format(block_idx, chr(97 + block)))
            bn_skip = bottom

        return bottom

    def add_relu(self, x, y, name=None):
        with tf.variable_scope(name):
            return tf.nn.relu(tf.add(x, y))

    def get_maxpooling(self, name, bottom, ksize=(2, 2), strides=(2, 2)):
        with tf.variable_scope(name):
            return tf.nn.max_pool(bottom, (1, ksize[0], ksize[1], 1), (1, strides[0], strides[1], 1), padding='SAME')

    def get_avgpooling(self, name, bottom, ksize=(2, 2), strides=(2, 2)):
        with tf.variable_scope(name):
            return tf.nn.avg_pool(bottom, (1, ksize[0], ksize[1], 1), (1, strides[0], strides[1], 1), padding='SAME')

    def get_fc(self, name, bottom, pretrained=True):
        with tf.variable_scope(name):
            W_init = tf.constant_initializer(self.weights_dict[name][self.weights_idx]) if pretrained else None
            W_shape = self.weights_dict[name][self.weights_idx].shape if pretrained else None
            b_init = tf.constant_initializer(self.weights_dict[name][self.bias_idx]) if pretrained else None
            b_shape = self.weights_dict[name][self.bias_idx].shape if pretrained else None

            W = tf.get_variable('W', shape=W_shape, dtype=tf.float32, initializer=W_init)
            b = tf.get_variable('b', shape=b_shape, dtype=tf.float32, initializer=b_init)

            return tf.matmul(bottom, W) + b

    def bilinear(self, name, bottom, factor=None, output_shape=None):
        with tf.variable_scope(name):
            if output_shape is None:
                bottom_shape = tf.shape(bottom, name='bottom_shape')
                return tf.image.resize_bilinear(bottom, (bottom_shape[1] * factor, bottom_shape[2] * factor))
            else:
                return tf.image.resize_bilinear(bottom, (output_shape[1], output_shape[2]))

    def get_deconv(self, name, bottom, factor, output_shape=None):
        """
        upsample using (learnable) bilinear interpolation kernel.
        :param factor: the scale to upsample.
        """
        with tf.variable_scope(name):
            if output_shape is None:
                bottom_shape = tf.shape(bottom, name='bottom_shape')
                # output_shape = tf.stack(
                #     [bottom_shape[0], bottom_shape[1] * factor, bottom_shape[2] * factor, bottom_shape[3]],
                #     name='output_shape_stacker')
                output_shape = tf.stack(
                    [bottom_shape[0], bottom_shape[1] * factor, bottom_shape[2] * factor,
                     bottom.get_shape().as_list()[-1]],
                    name='output_shape_stacker')

            # The numbers of input and output channels are the same.
            deconv_filter_weights = self._get_deconv_filter_init_weights(factor, bottom.get_shape().as_list()[-1])
            W_init = tf.constant_initializer(deconv_filter_weights)
            W = tf.get_variable('W', shape=deconv_filter_weights.shape, dtype=tf.float32, initializer=W_init)
            return tf.nn.conv2d_transpose(bottom, W, output_shape, strides=[1, factor, factor, 1],
                                          name='deconv')

    def _get_deconv_filter_init_weights(self, factor, input_depth):
        """
        Initialize the learnable transpose convolution weights using the bilinear interpolation kernel.
        """
        k_size = 2 * factor - factor % 2
        center = (k_size - 1) / 2
        bilinear = np.zeros((k_size, k_size))
        for i in range(bilinear.shape[0]):
            for j in range(bilinear.shape[1]):
                bilinear[i, j] = (factor - abs(i - center)) / factor * (factor - abs(j - center)) / factor
        weights = np.zeros((k_size, k_size, input_depth, input_depth))
        for i in range(input_depth):
            # only one upsampling kernel for each output class.
            weights[:, :, i, i] = bilinear
        return weights

    def get_fully_compressed_as_CNN(self, name, bottom, uncompressed_kernel_size, output_classes=20):
        """
        Get a `output_classes`-depth kernel from the original 1000-depth kernel trained on ImageNet.
        """
        bottom_shape = bottom.get_shape().as_list()
        fc_weights = self.weights_dict[name][self.weights_idx]

        # reshape from [4096, 1000] to [1, 1, 4096, 1000]
        fc_as_cnn_weights = fc_weights.reshape(uncompressed_kernel_size)

        fc_bias = self.weights_dict[name][self.bias_idx]

        # determine the number of layers of the original kernel to be taken average.
        block_stride = fc_as_cnn_weights.shape[-1] // output_classes

        compressed_weights = list()
        compressed_bias = list()
        # Take the mean of adjacent `block_stride` layers along the last axis as one layer of the compressed kernel.
        for i in range(output_classes):
            compressed_weights.append(np.mean(fc_as_cnn_weights[:, :, :, i * block_stride:(i + 1) * block_stride], 3))
            compressed_bias.append(np.mean(fc_bias[i * block_stride:(i + 1) * block_stride]))
        compressed_weights = np.stack(compressed_weights, axis=3)
        compressed_bias = np.array(compressed_bias)

        with tf.variable_scope(name):
            W_init = tf.constant_initializer(compressed_weights)
            W = tf.get_variable('W', compressed_weights.shape, dtype=tf.float32, initializer=W_init)
            b_init = tf.constant_initializer(compressed_bias)
            b = tf.get_variable('b', shape=compressed_bias.shape, initializer=b_init)
            conv = tf.nn.conv2d(bottom, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            return conv + b

    def get_fully_as_CNN(self, name, bottom, kernel_size):
        """
        The conversion of fully connected layers to convolution layers.
        """
        with tf.variable_scope(name):
            fc_weights = self.weights_dict[name][self.weights_idx]
            fc_as_cnn_weights = fc_weights.reshape(kernel_size)
            W_init = tf.constant_initializer(fc_as_cnn_weights)
            W = tf.get_variable('W', shape=fc_as_cnn_weights.shape, dtype=tf.float32, initializer=W_init)
            b_init = tf.constant_initializer(self.weights_dict[name][self.bias_idx])
            b = tf.get_variable('b', shape=self.weights_dict[name][self.bias_idx].shape, initializer=b_init)
            conv = tf.nn.conv2d(bottom, W, strides=[1, 1, 1, 1], padding='SAME', name='conv')
            return tf.nn.relu(conv + b)

    def fuser_concat(self, values, name='fuser_concat'):
        return tf.concat(values, -1, name)

    def fuser_add(self, values, name='fuser_add'):
        return tf.add(*values, name)

    def lateral_identity(self, values, output_channel=None, name='lateral_identity'):
        return tf.identity(values, name)

    def lateral_conv(self, bottom, output_channel=0, name='lateral_conv'):
        return self.get_conv(name, bottom, False, False, True, num_outputs=output_channel)

    def get_variance(self, bottom, name, axis=[1, 2], keep_dims=False):
        with tf.variable_scope(name):
            mean, variance = tf.nn.moments(bottom, axis, keep_dims, name='variance')
        return variance

    def get_variance_for_all(self, name, bottom, ksize=(1, 1), num_outputs=256):
        get_variance = self.get_variance
        upsample = self.bilinear
        get_conv = self.get_conv
        get_bn = self.get_bn
        print(bottom.shape)
        origin_size = (bottom.shape[1], bottom.shape[2])

        with tf.variable_scope(name):
            if ksize == origin_size:
                self.variance = get_variance(bottom, name='anameaa')
                print(self.variance.shape)
                self.variance = tf.expand_dims(tf.expand_dims(self.variance, 1), 1)
                self.variance = upsample('v_pooling_upsample', self.variance, output_shape=ksize)
                self.variance = get_conv('vconv', self.variance, pretrained=False, num_outputs=num_outputs)
                self.variance = get_bn('vbn', self.variance, pretrained=False, relu=False)
                return self.variance
            else:
                self.split1 = tf.split(bottom, [ksize[0] for i in range(origin_size[0] // ksize[0])], 1)
                print(np.asarray(self.split1).shape)
                print(self.split1)
                split_sizey = [ksize[1] for i in range(origin_size[1] // ksize[1])]
                self.split2 = []
                for i in range(np.asarray(self.split1).shape[0]):
                    self.split2.append(tf.split(self.split1[0], split_sizey, 2))
                    print(self.split2[i])
                print(self.split2[1][1])
                print(np.asarray(self.split2).shape)

                self.variance = []
                v_pattern = '{}{}'
                numx = origin_size[0] // ksize[0]
                # self.variance = np.ones((numx, numx))
                for i in range(np.asarray(self.split2).shape[0]):
                    for j in range(np.asarray(self.split2).shape[1]):
                        self.variance.append(tf.expand_dims(
                            tf.expand_dims(get_variance(tf.to_float(self.split2[i][j]), axis=[1, 2], name='variance_'),
                                           1), 1))
                        print(self.variance[i * np.asarray(self.split2).shape[0] + j].shape)
                        self.variance[i * np.asarray(self.split2).shape[0] + j] = upsample(
                            bottom=self.variance[i * np.asarray(self.split2).shape[0] + j],
                            name=v_pattern.format('variance', chr(97 + i)), factor=1, output_shape=ksize)
                        print(self.variance[i * np.asarray(self.split2).shape[0] + j].shape)
                self.variance = np.reshape(self.variance, (numx, numx))
                print(self.variance.shape)
                self.v_concat = []
                for i in range(numx):
                    self.v_concat.append(self.variance[i][0])
                self.v_concat = np.reshape(self.v_concat, (1, numx))
                for i in range(self.v_concat.shape[1]):
                    for j in range(1, numx):
                        self.v_concat[0][i] = tf.concat([self.v_concat[0][i], self.variance[i][j]], 2, 'aname')

                self.concat = self.v_concat[0][0]
                for i in range(1, numx):
                    self.concat = tf.concat([self.concat, self.v_concat[0][i]], 1, 'conv2')
                self.concat = get_conv('vconv', self.concat, pretrained=False, num_outputs=num_outputs)
                self.concat = get_bn('vbn', self.concat, pretrained=False, relu=False)
                print(self.concat.shape)
                return self.concat

    # def deformable_conv(self, bottom, name):
    #     input_shape = bottom.shape
    #     with tf.variable_scope(name):
    #         offsets = self.get_conv('offset_field', bottom, pretrained=False, num_output=input_shape[3]*2)
    #         # offsets: (b*c, h, w, 2)
    #         offsets = self._to_bc_h_w_2(offsets, input_shape)
    #
    #         # x: (b*c, h, w)
    #         x = self._to_bc_h_w(bottom, input_shape)
    #
    #         # X_offset: (b*c, h, w)
    #         x_offset = tf_batch_map_offsets(x, offsets)
    #
    #         # x_offset: (b, h, w, c)
    #         x_offset = self._to_b_h_w_c(x_offset, input_shape)
    #         x_output = self.get_conv('deformable_conv', x_offset, pretrained=False, num_output=input_shape[3])
    #         return x_output

    def compute_output_shape(self, input_shape):
        """Output shape is the same as input shape

        Because this layer does only the deformation part
        """
        return input_shape

    @staticmethod
    def _to_bc_h_w_2(x, x_shape):
        """(b, h, w, 2c) -> (b*c, h, w, 2)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2]), 2))
        return x

    @staticmethod
    def _to_bc_h_w(x, x_shape):
        """(b, h, w, c) -> (b*c, h, w)"""
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.reshape(x, (-1, int(x_shape[1]), int(x_shape[2])))
        return x

    @staticmethod
    def _to_b_h_w_c(x, x_shape):
        """(b*c, h, w) -> (b, h, w, c)"""
        x = tf.reshape(
            x, (-1, int(x_shape[3]), int(x_shape[1]), int(x_shape[2]))
        )
        x = tf.transpose(x, [0, 2, 3, 1])
        return x


def _fixed_padding(inputs, kernel_size, mode='CONSTANT'):
    """
    pad the input feature map independently of input size.
    """
    total_padding = kernel_size - 1
    padding_begin = total_padding // 2
    padding_end = total_padding - padding_begin
    return tf.pad(inputs, [[0, 0], [padding_begin, padding_end], [padding_begin, padding_end], [0, 0]], mode)


def conv2d_fixed_padding(input, num_outputs, kernel_size, stride=1):
    """
    When `stride` is greater than 1, use the input-size-independent-padding convolution layer.
    """
    if stride > 1:
        input = _fixed_padding(input, kernel_size)
    input = slim.conv2d(input, num_outputs, kernel_size, stride, ('SAME' if stride == 1 else 'VALID'))
    return input


def get_darknet53_block(input, base_num_outputs):
    """
    Get the internal block of darknet53. Similar to bottlenecks of ResNet.
    """
    shortcut = input
    input = conv2d_fixed_padding(input, base_num_outputs, 1)
    input = conv2d_fixed_padding(input, base_num_outputs * 2, 3)
    return input + shortcut


def get_yolo3_block(input, base_num_outputs):
    """
    convolution blocks inside detection-end.
    """
    for _ in range(2):
        input = conv2d_fixed_padding(input, base_num_outputs, 1)
        input = conv2d_fixed_padding(input, base_num_outputs * 2, 3)
    input = conv2d_fixed_padding(input, base_num_outputs, 1)
    route = input
    input = conv2d_fixed_padding(input, base_num_outputs * 2, 3)
    return route, input


def get_yolo3_detection(feature_map, num_classes, anchors, image_size):
    num_anchors = len(anchors)
    num_box_attrs = 5 + num_classes
    predictions = slim.conv2d(feature_map, num_anchors * num_box_attrs, 1, 1, normalizer_fn=None, activation_fn=None,
                              biases_initializer=tf.zeros_initializer())
    grid_size = tf.shape(feature_map)[1:3]
    predictions = tf.reshape(predictions, [-1, grid_size[0] * grid_size[1] * num_anchors, num_box_attrs])

    # Get anchors on the feature maps of current resolution based on its strides.
    strides = [image_size[0] / grid_size[0], image_size[1] / grid_size[1]]
    anchors_on_feature_map = [[each_anchor[0] / strides[0], each_anchor[1] / strides[1]] for each_anchor in anchors]

    box_center, box_size, confidence, logits = tf.split(predictions, [2, 2, 1, num_classes], -1)

    # Get the center of each grid on the feature map.
    # Iteration order: 1. x and y 2. anchors 3. grids
    offsets_height, offsets_width = tf.meshgrid(tf.range(grid_size[0]), tf.range(grid_size[1]), dtype=tf.float32)
    offsets = tf.concat([tf.reshape(offsets_height, [1, -1]), tf.reshape(offsets_width, [1, -1])], -1)
    offsets = tf.tile(offsets, [1, num_anchors])
    offsets = tf.reshape(offsets, [1, -1, 2])

    # Get the center of the bounding boxes on the original image.
    box_center = tf.nn.sigmoid(box_center)
    box_center += offsets
    box_center *= strides

    # Get the exact heights and widths of bounding boxes.
    anchors_on_feature_map = tf.tile(anchors_on_feature_map, [grid_size[0] * grid_size[1], 1])
    anchors_on_feature_map = tf.expand_dims(anchors_on_feature_map, 0)
    box_size = tf.exp(box_size) * anchors_on_feature_map
    box_size *= strides

    confidence = tf.nn.sigmoid(confidence)
    probabilities = tf.nn.sigmoid(logits)

    return tf.concat([box_center, box_size, confidence, probabilities], -1)
