from PIL import Image

from config import *


def data_augment(image, label, crop_height, crop_width, ignore_label=0, min_scale=0.5, max_scale=2.0):
    nr_image = len(image)
    image = tf.concat(image, axis=-1)

    image, label = random_rescale_image_and_label(image, label, min_scale, max_scale)
    image, label = random_crop_or_pad_image_and_label(image, label, crop_height, crop_width, ignore_label)
    image, label = random_left_right_flip(image, label)

    res_images = []
    for i in range(nr_image):
        res_images.append(image[:, :, i * 3:(i + 1) * 3])
    res_images = tuple(res_images)

    return res_images, label


# def resize_single_pair(image, label, height=224, width=224):
#     return tf.image.resize_images(image, (height, width)), \
#            tf.image.resize_images(label, (height, width), tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def mean_substract(images):
    bgr_mean = tf.constant([103.939, 116.779, 123.68], dtype=tf.float32, shape=[3], name='mean')
    r, g, b = tf.split(images, 3, axis=3, name='rgb_spliter_subtract')
    return tf.concat([b, g, r], axis=3) - bgr_mean


def mean_addition(images):
    rgb_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[3], name='mean')
    b, g, r = tf.split(images, 3, axis=3, name='rgb_spliter_addition')
    return tf.concat([r, g, b], axis=3) + rgb_mean


def mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.
    For example:
      means = [123.68, 116.779, 103.939]
      image = _mean_image_subtraction(image, means)
    Note that the rank of `image` must be known.
    Args:
      image: a tensor of size [height, width, C].
      means: a C-vector of values to subtract from each channel.
    Returns:
      the centered image.
    Raises:
      ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError('len(means) must match the number of channels')

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def random_rescale_image_and_label(image, label, min_scale, max_scale):
    """Rescale an image and label with in target scale.

    Rescales an image and label within the range of target scale.

    Args:
      image: 3-D Tensor of shape `[height, width, channels]`.
      label: 3-D Tensor of shape `[height, width, 1]`.
      min_scale: Min target scale.
      max_scale: Max target scale.

    Returns:
      Cropped and/or padded image.
      If `images` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, channels]`.
      If `labels` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, 1]`.
    """
    if min_scale <= 0:
        raise ValueError('\'min_scale\' must be greater than 0.')
    elif max_scale <= 0:
        raise ValueError('\'max_scale\' must be greater than 0.')
    elif min_scale >= max_scale:
        raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

    shape = tf.shape(image)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.random_uniform(
        [], minval=min_scale, maxval=max_scale, dtype=tf.float32)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    # Since label classes are integers, nearest neighbor need to be used.
    label = tf.image.resize_images(label, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    return image, label


def random_crop_or_pad_image_and_label(image, label, crop_height, crop_width, ignore_label):
    """Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by rondomly
    cropping the image or padding it evenly with zeros.

    Args:
      image: 3-D Tensor of shape `[height, width, channels]`.
      label: 3-D Tensor of shape `[height, width, 1]`.
      crop_height: The new height.
      crop_width: The new width.
      ignore_label: Label class to be ignored.

    Returns:
      Cropped and/or padded image.
      If `images` was 3-D, a 3-D float Tensor of shape
      `[new_height, new_width, channels]`.
    """
    label = label - ignore_label  # Subtract due to 0 padding.
    label = tf.to_float(label)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_and_label = tf.concat([image, label], axis=2)
    image_and_label_pad = tf.image.pad_to_bounding_box(
        image_and_label, 0, 0,
        tf.maximum(crop_height, image_height),
        tf.maximum(crop_width, image_width))
    image_and_label_crop = tf.random_crop(
        image_and_label_pad, [crop_height, crop_width, tf.shape(image)[-1] + 1])

    image_crop = image_and_label_crop[:, :, :-1]
    label_crop = image_and_label_crop[:, :, -1:]
    label_crop += ignore_label
    label_crop = tf.to_int32(label_crop)

    return image_crop, label_crop


def random_left_right_flip(image, label):
    prob = tf.random_uniform([], 0, 1, dtype=tf.float32)
    image, label = tf.cond(prob < 0.5,
                           lambda: (tf.reverse(image, [-2]), tf.reverse(label, [-2])), lambda: (image, label))
    return image, label


def decode_labels(mask, palette, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    mask = np.expand_dims(mask, -1)
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' \
                              % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = palette[k]
        outputs[i] = np.array(img)
    return outputs
