from data_process_vis_utils import *
from data_preprocess import data_augment


def VOC_pattern_input_fn(datahome, datalist, image_path='JPEGImages', label_path='SegmentationClassLabelImages',
                         datalist_path='ImageSets/Segmentation', num_epoches=1, batch_size=8, is_training=True,
                         data_aug=lambda image, label: data_augment(image, label, 513, 513, 255)):
    def _parse_single(image_path, label_path):
        raw_image = tf.read_file(image_path, 'rb')
        raw_label = tf.read_file(label_path, 'rb')
        image = tf.to_float(tf.image.decode_jpeg(raw_image))
        label = tf.to_int32(tf.image.decode_png(raw_label))
        if is_training:
            image, label = data_aug(image, label)
        image.set_shape([None, None, None])
        label = tf.squeeze(label)
        label.set_shape([None, None])
        return image, label

    with open(join(datahome, datalist_path, datalist)) as f:
        ids = f.read().split()
        image_ids = [os.path.join(datahome, image_path, f'{each_id}.jpg') for each_id in ids]
        label_ids = [os.path.join(datahome, label_path, f'{each_id}.png') for each_id in ids]
    dataset = tf.data.Dataset.from_tensor_slices((image_ids, label_ids))
    if is_training:
        dataset = dataset.shuffle(len(ids))
    dataset = dataset.map(_parse_single, num_parallel_calls=2)
    image_batch, label_batch = dataset.repeat(num_epoches).batch(batch_size).prefetch(
        batch_size).make_one_shot_iterator().get_next()
    return image_batch, label_batch


def _load_VOC_single_type(image_path, label_path, data_id):
    # Load the training images, labels and weights. The weights are used to specify the labels to be ignored during
    # the training process.
    images = list()
    labels = list()
    ids = list()
    for idx, each_id in enumerate(data_id):
        if idx % 100 == 0:
            logger.info('loadding data {}/{}'.format(idx, len(data_id)))
        images.append(io.imread(join(image_path, each_id + '.jpg')))
        cur_label = io.imread(join(label_path, each_id + '.png'))
        labels.append(cur_label)
        ids.append(each_id)
    return images, labels, ids


def load_VOC_pattern_data(data_home, load_train=True, load_val=True, num_train=None, num_val=None,
                          image_path='JPEGImages', label_path='SegmentationClassLabelImages',
                          datalist='ImageSets/Segmentation', train_datalist='train.txt', val_datalist='val.txt'):
    # Define the absolute path of the sub directories.
    image_path = join(data_home, image_path)
    label_path = join(data_home, label_path)
    train_idx_path = join(data_home, datalist, train_datalist)
    val_idx_path = join(data_home, datalist, val_datalist)

    # load the ids of validation images.
    X_train = y_train = train_ids = X_val = y_val = val_ids = []

    if load_train:
        with open(join(data_home, train_idx_path)) as f:
            train_ids = f.read().split()[:num_train]

    if load_val:
        with open(join(data_home, val_idx_path)) as f:
            val_ids = f.read().split()[:num_val]

    if load_train:
        logger.info('Loading training data...')
        X_train, y_train, train_ids = _load_VOC_single_type(image_path, label_path, train_ids)
    if load_val:
        logger.info('Loading validation data...')
        X_val, y_val, val_ids = _load_VOC_single_type(image_path, label_path, val_ids)

    if load_train and load_val:
        return list(map(np.array, [X_train, y_train, train_ids])), list(map(np.array, [X_val, y_val, val_ids]))
    elif load_train:
        return list(map(np.array, [X_train, y_train, train_ids]))
    elif load_val:
        return list(map(np.array, [X_val, y_val, val_ids]))
    else:
        raise Exception('Load what?')


def load_minhou(data_home, nr_random_sampling, sampling_size=None, patch_row=5, patch_col=5):
    """
    :param nr_random_sampling:
        None: load images without cropping.
        >0: the number of patches to be randomly sampled from each image. (`sampling_size` must be specified)
        <=0: divide each of the images into small patches. (`patch_row` and `patch_col` must be specified)
    :return:
    """
    X = list()
    y = list()
    mask = list()
    id = list()
    data_dirs = [each_file for each_file in os.listdir(data_home) if
                 os.path.isdir(join(data_home, each_file)) and each_file.startswith('minhou')]
    for idx, each_minhou_dir in enumerate(data_dirs):
        logger.info('Loading {} ... {}/{}'.format(each_minhou_dir, idx + 1, len(data_dirs)))
        cur_id = re.search(r'(minhou_patch_\d+)', each_minhou_dir).group(1)

        # read the image from disk.
        cur_image = io.imread(join(data_home, each_minhou_dir, cur_id + '.tif'))[1:-1, 1:-1]

        # read the label.
        for each_file in os.listdir(join(data_home, each_minhou_dir)):
            if each_file.startswith(cur_id + '_') and each_file.endswith('.tif'):
                cur_label = io.imread(join(data_home, each_minhou_dir, each_file))

        # get the mask

        if not nr_random_sampling:
            X.append(cur_image)
            y.append(cur_label)
            id.append(cur_id)

        else:
            if nr_random_sampling > 0:
                (X_patches, y_patches), id_patches = random_sampler([cur_image, cur_label], cur_id, nr_random_sampling,
                                                                    sampling_size)
            else:
                (X_patches, y_patches), id_patches = grid_cropping_helper([cur_image, cur_label],
                                                                          cur_id, patch_row, patch_col)
            X.extend(X_patches)
            y.extend(y_patches)
            id.extend(id_patches)

    return list(map(np.array, [X, y, id]))
