import tensorflow as tf
from config import darknet_batch_norm_decay
import tensorflow.contrib.slim as slim
from models_constructor import *

YOLOv3_anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]


class FasterRCNN():
    def __init__(self, front_end, num_classes, is_training):
        RPN_conv1 = front_end


class YOLOv3():
    def __init__(self, front_end, num_classes, is_training):
        batch_norm_params = {
            'decay': darknet_batch_norm_decay,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': is_training,
            'fused': None
        }
        image_size = tf.shape(front_end.X_input)[1:3]
        f16_shape = tf.shape(front_end.f16)[1:3]
        f8_shape = tf.shape(front_end.f8)[1:3]

        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                            biases_initialzier=None, activation_fn=lambda x: tf.nn.leaky_relu(x, 0.1)):
            with tf.variable_scope('YOLOv3'):
                with tf.variable_scope('detection32'):
                    route, yolo32 = get_yolo3_block(front_end, 512)
                    detection32 = get_yolo3_detection(yolo32, num_classes, YOLOv3_anchors[6:9], image_size)
                    net = conv2d_fixed_padding(route, 256, 1)
                    net = tf.image.resize_bilinear(net, f16_shape)
                    net = tf.concat([net, front_end.f16], -1)

                with tf.variable_scope('detection16'):
                    route, yolo16 = get_yolo3_block(net, 256)
                    detection16 = get_yolo3_detection(yolo16, num_classes, YOLOv3_anchors[3:6], image_size)
                    net = conv2d_fixed_padding(route, 128, 1)
                    net = tf.image.resize_bilinear(net, f8_shape)
                    net = tf.concat([net, front_end.f8], -1)

                with tf.variable_scope('detection8'):
                    _, yolo8 = get_yolo3_block(net, 128)
                    detection8 = get_yolo3_detection(yolo8, num_classes, YOLOv3_anchors[0:3], image_size)

        self.detections = tf.concat([detection32, detection16, detection8], 1)
