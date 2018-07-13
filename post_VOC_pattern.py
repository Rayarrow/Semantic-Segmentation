from pprint import pprint

from data_process_vis_utils import *
from palette_conversion import *
from skimage.color import rgba2rgb
import skimage
import sys
import pandas as pd
from config import *

inference_root = r'/Volumes/Transcend/summary/git_deeplabv3p/fujian/inference'
main_task_name = r'train_patches#200#8#0.007#1e-06#13500#0#fujian'

secondary_task_name = 'trainaug#43#10#0.007#1e-06#45500#2'
secondary_inference_home = join(inference_root, secondary_task_name)

inference_home = join(inference_root, main_task_name)
infer_label_path = join(inference_home, 'labels')
infer_heat_path = join(inference_home, 'heatmaps')
datalist_path = join(datalist_home, datalist)

task_names = [f'trainaug#43#10#0.007#1e-06#45500#{i}' for i in [0, 2, 1]]


def get_gt_masked(alpha=0.3):
    # compare images, colormaps and masked images.
    images, ids = load_VOC_pattern_image(image_home, datalist_path, 'tif')
    colormaps, _ = load_VOC_pattern_image(colormap_home, datalist_path, 'png')
    masked = get_masked_images(images, colormaps, alpha=alpha)
    comparisons = get_comparison([images, colormaps, masked])
    save_images(join(data_home, 'gt_masked_group'), comparisons, ids)


# get_gt_masked(fujian_home, alpha=0.15)


# def stitch_labels(datalist):
#     labels, ids = stitch_patches(data_home, datalist, 2, 2, join(inferenced_home, 'label_patches'))
#     save_images(join(inferenced_home, 'labels'), labels, ids, 'png')
#     return labels, ids


def acc_map(labels, gts, ignore_label=255):
    """
    generate an image to display the correctly and wrongly classified regions.
    Green indicates the right classifications and red indicates the wrong ones.
    """
    acc_labels = [(each_label == each_gt) | (gts == ignore_label) for each_label, each_gt in zip(labels, gts)]
    acc_colormaps = label_2_colormap([[255, 0, 0], [0, 255, 0]], acc_labels)
    return acc_colormaps


def _compute_mean_iou(total_cm, name='mean_iou'):
    """
    Compute the mean intersection-over-union via the confusion matrix.
    Note: Not necessary to call this.
    """
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(tf.cast(
        tf.not_equal(denominator, 0), dtype=tf.float32))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0),
        denominator,
        tf.ones_like(denominator))
    iou = tf.div(cm_diag, denominator)

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name=name) / num_valid_entries,
        0)
    return result


def _get_acc_miou(labels, gt_labels, ids, task_name, ignore_label=255):
    """
    Calculate the mean iou for the task `task_name` and save it to a txt file.
    Note: Not necessary to call this.
    """
    from sklearn.metrics import confusion_matrix
    cm_input = tf.placeholder(tf.float32, shape=[21, 21])
    tf_miou = _compute_mean_iou(cm_input)
    session = tf.Session()
    mious = []
    total_cm = np.zeros((21, 21))
    for idx, (each_label, each_gt, each_id) in enumerate(zip(labels, gt_labels, ids)):
        if idx % (len(labels) // 100 + 1) == 0:
            logger.info(f'getting miou...{idx}/{len(labels)}')
        valid_idx = each_gt != ignore_label
        each_label = each_label[valid_idx]
        each_gt = each_gt[valid_idx]
        cm = confusion_matrix(each_gt, each_label, labels=np.arange(21))
        total_cm += cm
        tf_miou_val = session.run(tf_miou, feed_dict={cm_input: cm})
        mious.append(tf_miou_val)

    res = pd.DataFrame(
        {'id': ids + ['overall'], 'miou': mious + [session.run(tf_miou, feed_dict={cm_input: total_cm})]})
    res.to_csv(join(inference_root, task_name, 'miou.txt'), sep='\t', float_format='%.4f', index=False, header=False)
    session.close()


def get_miou_txt(task_name):
    """
    Get the mean iou for the task named `task_name` and save it to a txt file inside `task_name` dir.
    """
    labels, ids = load_VOC_pattern_image(infer_label_path, datalist_path, label_ext)
    gt_labels, _ = load_VOC_pattern_image(label_home, datalist_path, ext=label_ext)
    _get_acc_miou(labels, gt_labels, ids, task_name)


# compare miou.txt.
def compare_miou_txt(task_names):
    """
    Generate a single txt file including all mean ious for each task in `task_names`.
    """
    res = pd.read_csv(join(inference_root, task_names[0], 'miou.txt'), sep='\t', header=None)
    for each_task in task_names[1:]:
        this_df = pd.read_csv(join(inference_root, each_task, 'miou.txt'), sep='\t', header=None)
        res = pd.concat([res, this_df.iloc[:, -1]], axis=1)
    res.to_csv(join(inference_root, 'comparisons', 'miou_comp.txt'), sep='\t', index=False, header=False,
               float_format='%.4f')


def compare_comparson(task_names):
    """
    !!!Prerequisite: Call this after `run_get_basic_info()` is called in case dir `comparisons` is not exists.
    Generate a single image containting all aggregated images of each task in `task_names`.
    """
    if not task_names:
        raise ValueError('Empty task name list.')
    image_list = []
    for each_task in task_names:
        images, ids = load_VOC_pattern_image(join(inference_root, each_task, 'comparison'), datalist_path)
        image_list.append(images)
    comparison = get_comparison(image_list)
    save_images(join(inference_root, 'comparisons', 'comparisons3groups'), comparison, ids)


# labels, ids = stitch_labels('val_patches.txt')
def run_get_basic_info(task_name):
    # generate a colormap, a masked colormap, an acc_colormap, a masked_acc_colormaps, a masked_heatmaps, and an
    # aggregated image including all of the above for a single image.
    labels, ids = load_VOC_pattern_image(infer_label_path, datalist_path, label_ext)
    colormaps = label_2_colormap(palette, labels)
    save_images(join(inference_home, 'colormaps'), colormaps, ids)
    images, _ = load_VOC_pattern_image(image_home, datalist_path, ext=image_ext)
    masked = get_masked_images(images, colormaps, alpha=0.4)
    save_images(join(inference_home, 'masked'), masked, ids)
    gt_labels, _ = load_VOC_pattern_image(label_home, datalist_path, ext=label_ext)
    gt_label_colormaps = label_2_colormap(palette, gt_labels)
    acc_colormaps = acc_map(labels, gt_labels)
    save_images(join(inference_home, 'acc_colormaps'), acc_colormaps, ids, ext='tif')
    acc_masked = get_masked_images(images, acc_colormaps, alpha=0.4)
    heatmaps, ids = load_VOC_pattern_image(join(inference_home, 'heatmaps'), datalist_path, ext=label_ext)
    heatmaps = [skimage.img_as_ubyte(rgba2rgb(each_heat)) for each_heat in heatmaps]
    heap_masked = get_masked_images(images, heatmaps, alpha=0.4)

    comparisons = get_comparison(
        [images, gt_label_colormaps, masked, colormaps, acc_masked, acc_colormaps, heap_masked, heatmaps], (4, 2))
    save_images(join(inference_home, 'comparison'), comparisons, ids)
    get_miou_txt(task_name)


# def comparison_of_comparison(infer_dir1, infer_dir2):
#     compare two pairs of 3-tuple comparisons got from `get_gt_comparison()`.
# res = []
# comparisons1, ids = load_VOC_pattern_image(data_home, 'complete_test.txt', join(infer_dir1, 'complete_comparison'))
# comparisons2, _ = load_VOC_pattern_image(data_home, 'complete_test.txt', join(infer_dir2, 'complete_comparison'))
# for each_c1, each_c2 in zip(comparisons1, comparisons2):
#     coc = np.concatenate([each_c1, each_c2])
#     res.append(coc)
# save_images(join(data_home, f'coc_{infer_dir1}#{infer_dir2}'), res, ids)


def get_acc_diff_map(task_name1, task_name2, ext='tif'):
    """
    !!!Prerequisite: dir `acc_colormaps` is generated. (call `run_get_basic_info()` first)
    Generate an image to indicates the regions of improvement and degeneracy of `task_name1` over `task_name2`.
    Regions correctly classified in `task_name1` but wrongly in `task_name2` are colored as magenta.
    Regions correctly classified in `task_name2` but wrongly in `task_name1` are colored as cyan.
    """
    acc_colormap1, ids = load_VOC_pattern_image(join(inference_root, task_name1, 'acc_colormaps'), datalist_path,
                                                ext=ext)
    acc_colormap2, ids = load_VOC_pattern_image(join(inference_root, task_name2, 'acc_colormaps'), datalist_path,
                                                ext=ext)
    acc_diff_colormaps = [255 - (each_acc_colormap1 - each_acc_colormap2) for (each_acc_colormap1, each_acc_colormap2)
                          in
                          zip(acc_colormap1, acc_colormap2)]
    return acc_diff_colormaps


def compare_tasks_improve(task_name1, task_name2):
    acc_diff_colormaps = get_acc_diff_map(task_name1, task_name2)
    images, ids = load_VOC_pattern_image(join(data_home, 'JPEGImages'), datalist_path, ext=image_ext)
    colormaps1, _ = load_VOC_pattern_image(join(inference_root, task_name1, 'colormaps'), datalist_path, ext=image_ext)
    colormaps2, _ = load_VOC_pattern_image(join(inference_root, task_name2, 'colormaps'), datalist_path, ext=image_ext)
    masked = get_masked_images(images, acc_diff_colormaps, alpha=0.4)
    comparisons = get_comparison([images, acc_diff_colormaps, masked, images, colormaps1, colormaps2])
    save_images(join(inference_root, 'comparisons', f'imporve_{task_name1}_%_{task_name2}'), comparisons,
                [f'{each_id}' for each_id in ids], 'tif')


def run():
    # Get basic info for each task.
    [run_get_basic_info(each_task_name) for each_task_name in task_names]
    # Get a txt including all miou for each task.
    compare_miou_txt(task_names)
    # Get a image showing the regions of improvements and degeneracies.
    compare_tasks_improve(main_task_name, secondary_task_name)
    # Put 3 stitched aggregated images into a single image.
    compare_comparson(task_names)

    comparison3groups, ids = load_VOC_pattern_image(join(inference_root, 'comparisons', 'comparisons3groups'),
                                                    datalist_path, ext='jpg')
    impro, _ = load_VOC_pattern_image(join(inference_root, 'comparisons',
                                           'imporve_trainaug#43#10#0.007#1e-06#45500#0_%_trainaug#43#10#0.007#1e-06#45500#2'),
                                      datalist_path, ext='tif')
    aggregated_com_impro = get_comparison([comparison3groups, impro], shape=(2, 1))
    save_images(join(inference_root, 'comparisons', 'aggregated3groups'), aggregated_com_impro, ids, 'tif')

    # compare_tasks_improve(task_name, task_name2)

    # comparison_multiple_sets('heatmaps', 'valtest.txt', 'png', 'train#30#8#0.007#1e-06#36000#0',
    #                          'train#30#8#0.007#1e-06#36000#1', 'train#30#8#0.007#1e-06#36000#2')
