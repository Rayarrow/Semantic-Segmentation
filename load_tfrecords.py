import tensorflow as tf
import skimage.io as sio


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _create_example(feature):
    return tf.train.Example(features=tf.train.Features(feature=feature))


def load_image(path):
    return sio.imread(path)


data_path = '/Volumes/Transcend/summary/test_tfrecords/cat.tfrecords'

feature = {
    'label': tf.FixedLenFeature([], tf.int64),
    'image': tf.FixedLenFeature([], tf.string),
}

queue = tf.train.string_input_producer([data_path], num_epochs=1)
record_reader = tf.TFRecordReader()
serialized_example = record_reader.read(queue)
features = tf.parse_single_example(serialized_example, feature)

image = tf.decode_raw(feature['image'], tf.float32)
label = tf.cast(feature['label'], tf.int32)

sio.imshow(image)
print(label)

