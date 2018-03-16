import numpy as np
import tensorflow as tf

from config import *

UPSAMPLE_BILINEAR = 'bilinear'
UPSAMPLE_DECONV = 'deconv'


class component_constructor():
    def __init__(self, weights_path, weights_idx='weights', bias_idx='biases',
                 bn_idx=('moving_mean', 'moving_variance', 'beta', 'gamma'), weight_decay=None, beta_decay=None,
                 gamma_decay=None):
        logger.info('loadding {} ...'.format(weights_path))
        self.weights_dict = np.load(weights_path, encoding='latin1').item()
        self.weights_idx = weights_idx
        self.bias_idx = bias_idx
        self.bn_idx = bn_idx
        self.weight_regularizer = tf.contrib.layers.l2_regularizer(weight_decay) if weight_decay else None
        self.beta_regularizer = tf.contrib.layers.l2_regularizer(beta_decay) if beta_decay else None
        self.gamma_regularizer = tf.contrib.layers.l2_regularizer(gamma_decay) if gamma_decay else None

    def get_conv(self, name, bottom, pretrained=True, relu=False, bias=False, strides=(1, 1, 1, 1), k_size=(1, 1),
                 num_output=1, atrous=False):
        # Use this to determine the third dimension of W.
        num_input = bottom.get_shape().as_list()[-1]
        with tf.variable_scope(name):
            # Load pretrained weights if they exist.
            W_init = tf.constant_initializer(self.weights_dict[name][self.weights_idx]) if pretrained else None
            W_shape = self.weights_dict[name][self.weights_idx].shape if pretrained else list(k_size) + [num_input,
                                                                                                         num_output]
            W = tf.get_variable('W', shape=W_shape, dtype=tf.float32, initializer=W_init,
                                regularizer=self.weight_regularizer)
            conv_op = tf.nn.atrous_conv2d if atrous else tf.nn.conv2d
            conv_name = 'atrous_conv' if atrous else 'conv2d'
            conv = conv_op(bottom, W, strides, padding='SAME', name=conv_name)

            if bias:
                b_init = tf.constant_initializer(self.weights_dict[name][self.bias_idx]) if pretrained else None
                b_shape = self.weights_dict[name][self.bias_idx].shape if pretrained else num_output
                b = tf.get_variable('b', shape=b_shape, dtype=tf.float32, initializer=b_init)
                conv = conv + b

            return tf.nn.relu(conv) if relu else conv

    def get_bn(self, name, bottom, pretrained=True, relu=True):
        with tf.variable_scope(name):
            bn = tf.layers.batch_normalization(bottom, moving_mean_initializer=tf.constant_initializer(
                self.weights_dict[name][self.bn_idx[0]]), moving_variance_initializer=tf.constant_initializer(
                self.weights_dict[name][self.bn_idx[1]]), beta_initializer=tf.constant_initializer(
                self.weights_dict[name][self.bn_idx[2]]), gamma_initializer=tf.constant_initializer(
                self.weights_dict[name][self.bn_idx[3]]), beta_regularizer=self.beta_regularizer,
                                               gamma_regularizer=self.gamma_regularizer) if pretrained else tf.layers.batch_normalization(
                bottom, beta_regularizer=self.beta_regularizer, gamma_regularizer=self.gamma_regularizer)
            return tf.nn.relu(bn) if relu else bn

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
                return tf.image.resize_bilinear(bottom, output_shape)

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
        return self.get_conv(name, bottom, False, False, True, num_output=output_channel)
