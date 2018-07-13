import tensorflow as tf

class FasterRCNN():
    def __init__(self, front_end, num_classes, is_training):
        RPN_conv1 = front_end