from data_process_vis_utils import *
from palette_conversion import *

data = 'voc'
if data == 'isprs':
    data_home = '/Volumes/Transcend/Dataset/ISPRS/cropped'
    palette = ISPRS_palette
    image_ext = 'tif'
    datalist_path = 'datalist'
    image_path = 'images'
elif data == 'voc':
    data_home = '/Volumes/Transcend/Dataset/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'
    palette = VOC_palette
    image_ext = 'jpg'
    datalist_path = 'ImageSets/Segmentation'
    image_path = 'JPEGImages'
    gt_label_path = 'SegmentationClass'
else:
    raise Exception('invalid data.')


def get_gt_comparison():
    # compare images, colormaps and masked images.
    data_home = r'G:\Documents\Exp Data\data-li\segmentation-remote sensing\ISPRS\original_images'
    images, ids = load_VOC_pattern_image(data_home, 'trval.txt', 'raw_images', ext='tif')
    colormaps, _ = load_VOC_pattern_image(data_home, 'trval.txt', 'label_colormap', ext='tif')
    masked = get_masked_images(images, colormaps)
    comparisons = get_comparison([images, colormaps, masked])
    save_images(join(data_home, 'trval_comparison'), comparisons, ids)


def stitch_labels(datalist):
    labels, ids = stitch_patches(data_home, datalist, 2, 2, join(inferenced_home, 'label_patches'))
    save_images(join(inferenced_home, 'labels'), labels, ids, 'png')
    return labels, ids


def comparison_of_comparison(infer_dir1, infer_dir2):
    # compare two pairs of 3-tuple comparisons got from `get_gt_comparison()`.
    res = []
    comparisons1, ids = load_VOC_pattern_image(data_home, 'complete_test.txt', join(infer_dir1, 'complete_comparison'))
    comparisons2, _ = load_VOC_pattern_image(data_home, 'complete_test.txt', join(infer_dir2, 'complete_comparison'))
    for each_c1, each_c2 in zip(comparisons1, comparisons2):
        coc = np.concatenate([each_c1, each_c2])
        res.append(coc)
    save_images(join(data_home, f'coc_{infer_dir1}#{infer_dir2}'), res, ids)


def acc_map(labels, gts, ignore_label=255):
    # fill correct label with green and wrong ones with red.
    acc_labels = [(each_label == each_gt) | (gts == ignore_label) for each_label, each_gt in zip(labels, gts)]
    acc_colormaps = label_2_colormap([[255, 0, 0], [0, 255, 0]], acc_labels)
    return acc_colormaps


def get_acc_diff_map(infer_dir1, infer_dir2):
    # Corrections in 1 but mistakes in 2 are colored as megenta.
    infer_path1 = join(inferenced_root, infer_dir1)
    infer_path2 = join(inferenced_root, infer_dir2)
    acc_colormap1, ids = load_VOC_pattern_image(data_home, 'val.txt', join(infer_path1, 'acc_colormaps'), datalist_path,
                                                ext='tif')
    acc_colormap2, _ = load_VOC_pattern_image(data_home, 'val.txt', join(infer_path2, 'acc_colormaps'), datalist_path,
                                              ext='tif')
    acc_diff_colormaps = [255 - (each_acc_colormap1 - each_acc_colormap2) for (each_acc_colormap1, each_acc_colormap2)
                          in
                          zip(acc_colormap1, acc_colormap2)]
    save_images(join(inferenced_root, 'comparisons', f'{infer_dir1}ccc{infer_dir2}'), acc_diff_colormaps, ids)
    return acc_diff_colormaps


def compare_difference_two(acc_diff_colormaps, infer_dir1, infer_dir2):
    images, ids = load_VOC_pattern_image(data_home, 'val.txt', image_path, datalist_path, ext='tif')
    masked = get_masked_images(images, acc_diff_colormaps, alpha=0.4)
    comparisons = get_comparison([images, acc_diff_colormaps, masked])
    save_images(join(inferenced_root, 'comparisons', f'{infer_dir1}ccc{infer_dir2}'), comparisons,
                [f'comp_{each_id}' for each_id in ids])


def comparison_multiple_sets(image_type, datalist, ext, *tasks):
    image_list = []
    for each_task in tasks:
        images, ids = load_VOC_pattern_image(data_home, datalist, join(inferenced_root, each_task, image_type), ext=ext)
        image_list.append(images)
    comparisons = get_comparison(image_list)
    save_images(join(inferenced_root, 'comparisons', f'ccc#{image_type}#{".".join(tasks)}'), comparisons, ids, ext=ext)


# comparison_of_comparison('inference_crop_B', 'inference_crop_B_lowlevel')

# labels, ids = stitch_labels('test_patches.txt')
# colormaps = label_2_colormap(ISPRS_palette, labels)
# save_images(join(inferenced_home, 'colormaps'), colormaps, ids)
# images, _ = load_VOC_pattern_image(data_home, 'test.txt', 'images', ext='tif')
# masked = get_masked_images(images, colormaps, alpha=0.4)
# save_images(join(inferenced_home, 'masked'), masked, ids)
# comparisons = get_comparison([images, colormaps, masked])
# save_images(join(inferenced_home, 'comparison'), comparisons, ids)

task_name = 'trainaug#43#10#0.007#1e-06#45500#2'
label_path = join('inference', task_name, 'labels')
label_ext = 'png'
inferenced_home = join(data_home, 'inference', task_name)
inferenced_root = join(data_home, 'inference')

# labels, ids = stitch_labels('val_patches.txt')
labels, ids = load_VOC_pattern_image(data_home, 'val.txt', label_path, datalist_path=datalist_path, ext=label_ext)
colormaps = label_2_colormap(palette, labels)
save_images(join(inferenced_home, 'colormaps'), colormaps, ids)
images, _ = load_VOC_pattern_image(data_home, 'val.txt', image_path, datalist_path=datalist_path, ext=image_ext)
masked = get_masked_images(images, colormaps, alpha=0.4)
save_images(join(inferenced_home, 'masked'), masked, ids)
gt_labels, _ = load_VOC_pattern_image(data_home, 'val.txt', gt_label_path, datalist_path=datalist_path, ext=label_ext)
gt_label_colormaps = label_2_colormap(palette, gt_labels)
acc_colormaps = acc_map(labels, gt_labels)
save_images(join(inferenced_home, 'acc_colormaps'), acc_colormaps, ids, ext=image_ext)
acc_masked = get_masked_images(images, acc_colormaps, alpha=0.4)
comparisons = get_comparison([images, gt_label_colormaps, colormaps, masked, acc_colormaps, acc_masked])
save_images(join(inferenced_home, 'comparison'), comparisons, ids)

task_name2 = 'trainaug#43#10#0.007#1e-06#45500#2'

# comparison_multiple_sets('heatmaps', 'valtest.txt', 'png', 'train#30#8#0.007#1e-06#36000#0',
#                          'train#30#8#0.007#1e-06#36000#1', 'train#30#8#0.007#1e-06#36000#2')
