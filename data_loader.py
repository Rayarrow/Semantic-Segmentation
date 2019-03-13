from data_preprocess import data_augment
import os
import tensorflow as tf


def VOC_pattern_input_fn(image_home, label_home, datalist_path, nr_channel=3, num_epochs=1, batch_size=8, pair=False,
                         is_training=True, data_aug=lambda image, label: data_augment(image, label, 513, 513, 255)):
    def _parse_single(image_paths, label_path):
        images = []
        for each_image_path in image_paths:
            raw_image = tf.read_file(each_image_path, 'rb')
            image = tf.to_float(tf.image.decode_jpeg(raw_image))
            images.append(image)
        images = tuple(images)

        raw_label = tf.read_file(label_path, 'rb')
        label = tf.to_int32(tf.image.decode_png(raw_label))

        if is_training and data_aug:
            images, label = data_aug(images, label)

        for i in range(len(images)):
            images[i].set_shape([None, None, nr_channel])
        images = tuple(images)

        label = tf.squeeze(label)
        label.set_shape([None, None])

        if len(images) == 1:
            images = images[0]
        return images, label

    with open(datalist_path) as f:
        ids = f.read().split()
        image_paths = [os.path.join(image_home, f'{each_id}.jpg') for each_id in ids]
        if pair:
            image_comp_paths = ([os.path.join(image_home + '_comp', f'{each_id}.jpg') for each_id in ids])
            image_paths = (image_paths, image_comp_paths)
        else:
            image_paths = (image_paths,)

        label_paths = [os.path.join(label_home, f'{each_id}.png') for each_id in ids]

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths))

    if is_training:
        dataset = dataset.shuffle(len(ids))
    dataset = dataset.map(_parse_single, num_parallel_calls=2)

    image_batch, label_batch = dataset.repeat(num_epochs).batch(batch_size).prefetch(
        batch_size).make_one_shot_iterator().get_next()
    return image_batch, label_batch


# def VOC_pattern_pair_input_fn(image_home, label_home, datalist_path, nr_channel=3, num_epoches=1, batch_size=8,
#                               is_training=True,
#                               data_aug_pair=lambda image, image_comp, label: data_augment(image, image_comp, label,
#                                                                                           513, 513, 255)):
#     def _parse_single(image_pair_path, label_path):
#         image_path = image_pair_path[0]
#         image_comp_path = image_pair_path[1]
#         raw_image = tf.read_file(image_path, 'rb')
#         image = tf.to_float(tf.image.decode_jpeg(raw_image))
#
#         raw_image_comp = tf.read_file(image_comp_path, 'rb')
#         image_comp = tf.to_float(tf.image.decode_jpeg(raw_image_comp))
#
#         raw_label = tf.read_file(label_path, 'rb')
#         label = tf.to_int32(tf.image.decode_png(raw_label))
#
#         if is_training:
#             image, image_comp, label = data_aug_pair(image, image_comp, label)
#
#         image.set_shape([None, None, nr_channel])
#         image_comp.set_shape([None, None, nr_channel])
#         label = tf.squeeze(label)
#         label.set_shape([None, None])
#         return (image, image_comp), label
#
#     with open(datalist_path) as f:
#         ids = f.read().split()
#         image_paths = [os.path.join(image_home, f'{each_id}.jpg') for each_id in ids]
#         image_comp_paths = [os.path.join(image_home + '_comp', f'{each_id}.jpg') for each_id in ids]
#         label_paths = [os.path.join(label_home, f'{each_id}.png') for each_id in ids]
#
#     dataset = tf.data.Dataset.from_tensor_slices(((image_paths, image_comp_paths), label_paths))
#
#     if is_training:
#         dataset = dataset.shuffle(len(ids))
#     dataset = dataset.map(_parse_single, num_parallel_calls=2)
#
#     image_pair_batch, label_batch = dataset.repeat(num_epoches).batch(batch_size).prefetch(
#         batch_size).make_one_shot_iterator().get_next()
#     return image_pair_batch, label_batch
#
