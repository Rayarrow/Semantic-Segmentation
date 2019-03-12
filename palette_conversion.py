from config import *
from obsolete.data_process_vis_utils import load_VOC_pattern_image, save_images


# This function converts from segmentation class colormap images to labeled images.
def colormap_2_label(palette, colormaps, ids):
    labels = []
    pixel2label = dict(zip(map(tuple, palette), range(len(palette))))

    # Convert raw colormap images to label images.
    for (each_colormap, each_id) in zip(colormaps, ids):
        logger.info(f'colormap to label: {each_id}')
        cur_label = np.zeros(each_colormap.shape[:2], dtype=np.int32)
        for i in range(each_colormap.shape[0]):
            for j in range(each_colormap.shape[1]):
                cur_label[i, j] = pixel2label[tuple(each_colormap[i, j])]
        labels.append(cur_label)

    return labels


def label_2_colormap(palette, labels, ignore_label=255):
    colormaps = []
    for idx, each_label in enumerate(labels):
        logger.info('label to colormap {} / {}'.format(idx + 1, len(labels)))
        this_colormap = np.zeros((each_label.shape[0], each_label.shape[1], 3))
        for i in range(each_label.shape[0]):
            for j in range(each_label.shape[1]):
                if each_label[i, j] == ignore_label:
                    this_colormap[i, j] = [255, 255, 255]
                else:
                    this_colormap[i, j] = palette[each_label[i, j]]

        colormaps.append(this_colormap.astype('uint8'))

    return colormaps


# def run_test():
#     label_home = r'D:\Datasets\change_detection_dataset\GSV\labels'
#     colormap_home = r'D:\Datasets\change_detection_dataset\GSV\colormaps'
#     labels, ids = load_VOC_pattern_image(label_home, join(datalist_home, 'all_data.txt'), 'bmp')
#     colormaps = label_2_colormap(palette, labels)
#     save_images(colormap_home, colormaps, ids, 'png')


if __name__ == '__main__':
    dataset = 'ISPRS'
    datalist = 'trval.txt'
    dataset_config_bundle = data_home_bundle[dataset]
    data_home = dataset_config_bundle[0]
    image_home = join(data_home, dataset_config_bundle[1])
    label_home = join(data_home, dataset_config_bundle[2])
    datalist_home = join(data_home, dataset_config_bundle[3])
    palette = dataset_config_bundle[4]

    colormap_home = join(data_home, 'colormaps')
    labels, ids = load_VOC_pattern_image(label_home, join(datalist_home, datalist), 'png')
    colormaps = label_2_colormap(palette, labels)
    save_images(colormap_home, colormaps, ids, 'png')
