from obsolete.data_process_vis_utils import *
from palette_conversion import *
from skimage.color import rgba2rgb
import skimage
import pandas as pd
from model_estimators import compute_mean_iou


class PostPrediction:
    def __init__(self, inference_root, task_name, dataset, datalist, secondary_task_name=None):
        self.inference_root = inference_root
        self.task_name = task_name
        inference_home = join(inference_root, task_name)

        dataset_config_bundle = data_home_bundle[dataset]
        data_home = dataset_config_bundle[0]
        self.image_home = join(data_home, dataset_config_bundle[1])
        self.label_home = join(data_home, dataset_config_bundle[2])
        self.datalist_home = join(data_home, dataset_config_bundle[3])
        self.palette = dataset_config_bundle[4]

        self.datalist_path = join(self.datalist_home, datalist)
        self.inference_home = join(inference_root, task_name)
        self.inference_label_home = join(inference_home, 'labels')
        self.inference_heat_home = join(inference_home, 'heatmaps')
        self.label_ext = 'png'
        self.image_ext = 'jpg'

    def get_gt_masked(self, alpha=0.3):
        # compare images, colormaps and masked images.
        colormap_home = join(data_home, 'colormaps')
        images, ids = load_VOC_pattern_image(self.image_home, self.datalist_path, 'jpg')
        colormaps, _ = load_VOC_pattern_image(colormap_home, self.datalist_path, 'png')
        masked = get_masked_images(images, colormaps, alpha=alpha)
        comparisons = get_comparison([images, colormaps, masked])
        save_images(join(data_home, 'gt_masked_group'), comparisons, ids)

    def acc_map(self, labels, gts, ignore_label=255):
        """
        generate an image to display the correctly and wrongly classified regions.
        Green indicates the right classifications and red indicates the wrong ones.
        """
        acc_labels = [(each_label == each_gt) | (gts == ignore_label) for each_label, each_gt in zip(labels, gts)]
        acc_colormaps = label_2_colormap([[255, 0, 0], [0, 255, 0]], acc_labels)
        return acc_colormaps

    def _get_acc_miou(self, labels, gt_labels, ids, task_name, ignore_label=255):
        """
        Calculate the mean iou for the task `task_name` and save it to a txt file.
        Note: Not necessary to call this.
        """
        from sklearn.metrics import confusion_matrix
        nr_classes = len(self.palette)
        cm_input = tf.placeholder(tf.float32, shape=[nr_classes, nr_classes])
        tf_miou = compute_mean_iou(cm_input, len(self.palette))
        tf_f1 = compute_mean_iou(cm_input, len(self.palette), 'f1')
        session = tf.Session()
        mious = []
        f1s = []
        accs = []
        total_cm = np.zeros((nr_classes, nr_classes))
        for idx, (each_label, each_gt, each_id) in enumerate(zip(labels, gt_labels, ids)):
            if idx % (len(labels) // 100 + 1) == 0:
                logger.info(f'getting miou...{idx+1}/{len(labels)}')
            valid_idx = each_gt != ignore_label
            each_label = each_label[valid_idx]
            each_gt = each_gt[valid_idx]
            cm = confusion_matrix(each_gt, each_label, labels=np.arange(nr_classes))
            print(cm)
            total_cm += cm

            tf_miou_val = session.run(tf_miou, feed_dict={cm_input: cm})
            mious.append(tf_miou_val)
            tf_f1_val = session.run(tf_f1, feed_dict={cm_input: cm})
            f1s.append(tf_f1_val)

            accs.append(np.diag(cm).sum() / cm.sum())

        res = pd.DataFrame(
            {'id': ids + ['overall'], 'miou': mious + [session.run(tf_miou, feed_dict={cm_input: total_cm})],
             'f1': f1s + [session.run(tf_f1, feed_dict={cm_input: total_cm})],
             'acc': accs + [np.diag(total_cm).sum() / total_cm.sum()]})
        res.to_csv(join(self.inference_root, task_name, 'metrics.txt'), sep='\t', float_format='%.4f', index=False,
                   header=True)
        session.close()

    def get_miou_txt(self, task_name):
        """
        Get the mean iou for the task named `task_name` and save it to a txt file inside `task_name` dir.
        """
        labels, ids = load_VOC_pattern_image(self.inference_label_home, self.datalist_path, self.label_ext)
        gt_labels, _ = load_VOC_pattern_image(self.label_home, self.datalist_path, ext=self.label_ext)
        self._get_acc_miou(labels, gt_labels, ids, task_name)

    # compare miou.txt.
    def compare_miou_txt(self, task_names):
        """
        Generate a single txt file including all mean ious for each task in `task_names`.
        """
        res = pd.read_csv(join(self.inference_root, task_names[0], 'miou.txt'), sep='\t', header=None)
        for each_task in task_names[1:]:
            this_df = pd.read_csv(join(self.inference_root, each_task, 'miou.txt'), sep='\t', header=None)
            res = pd.concat([res, this_df.iloc[:, -1]], axis=1)
        res.to_csv(join(self.inference_root, 'comparisons', 'miou_comp.txt'), sep='\t', index=False, header=False,
                   float_format='%.4f')

    def compare_comparson(self, task_names):
        """
        !!!Prerequisite: Call this after `run_get_basic_info()` is called in case dir `comparisons` is not exists.
        Generate a single image containting all aggregated images of each task in `task_names`.
        """
        if not task_names:
            raise ValueError('Empty task name list.')
        image_list = []
        for each_task in task_names:
            images, ids = load_VOC_pattern_image(join(self.inference_root, each_task, 'comparison'), self.datalist_path)
            image_list.append(images)
        comparison = get_comparison(image_list)
        save_images(join(self.inference_root, 'comparisons', 'comparisons3groups'), comparison, ids)

    # labels, ids = stitch_labels('val_patches.txt')
    def run_get_basic_info(self):
        # generate a colormap, a masked colormap, an acc_colormap, a masked_acc_colormaps, a masked_heatmaps, and an
        # aggregated image including all of the above for a single image.
        labels, ids = load_VOC_pattern_image(self.inference_label_home, self.datalist_path, self.label_ext)
        colormaps = label_2_colormap(self.palette, labels)
        save_images(join(self.inference_home, 'colormaps'), colormaps, ids, 'png')
        images, _ = load_VOC_pattern_image(self.image_home, self.datalist_path, ext=self.image_ext)
        masked = get_masked_images(images, colormaps, alpha=0.4)
        save_images(join(self.inference_home, 'masked'), masked, ids)
        gt_labels, _ = load_VOC_pattern_image(self.label_home, self.datalist_path, ext=self.label_ext)
        gt_label_colormaps = label_2_colormap(self.palette, gt_labels)
        acc_colormaps = self.acc_map(labels, gt_labels)
        save_images(join(self.inference_home, 'acc_colormaps'), acc_colormaps, ids, ext='tif')
        acc_masked = get_masked_images(images, acc_colormaps, alpha=0.4)
        heatmaps, ids = load_VOC_pattern_image(join(self.inference_home, 'heatmaps'), self.datalist_path,
                                               ext=self.label_ext)
        heatmaps = [skimage.img_as_ubyte(rgba2rgb(each_heat)) for each_heat in heatmaps]
        heap_masked = get_masked_images(images, heatmaps, alpha=0.4)

        comparisons = get_comparison(
            [images, gt_label_colormaps, masked, colormaps, acc_masked, acc_colormaps, heap_masked, heatmaps], (4, 2))
        save_images(join(self.inference_home, 'comparison'), comparisons, ids)
        self.get_miou_txt(self.task_name)

    # def comparison_of_comparison(infer_dir1, infer_dir2):
    #     compare two pairs of 3-tuple comparisons got from `get_gt_comparison()`.
    # res = []
    # comparisons1, ids = load_VOC_pattern_image(data_home, 'complete_test.txt', join(infer_dir1, 'complete_comparison'))
    # comparisons2, _ = load_VOC_pattern_image(data_home, 'complete_test.txt', join(infer_dir2, 'complete_comparison'))
    # for each_c1, each_c2 in zip(comparisons1, comparisons2):
    #     coc = np.concatenate([each_c1, each_c2])
    #     res.append(coc)
    # save_images(join(data_home, f'coc_{infer_dir1}#{infer_dir2}'), res, ids)

    def get_acc_diff_map(self, task_name1, task_name2, ext='tif'):
        """
        !!!Prerequisite: dir `acc_colormaps` is generated. (call `run_get_basic_info()` first)
        Generate an image to indicates the regions of improvement and degeneracy of `task_name1` over `task_name2`.
        Regions correctly classified in `task_name1` but wrongly in `task_name2` are colored as magenta.
        Regions correctly classified in `task_name2` but wrongly in `task_name1` are colored as cyan.
        """
        acc_colormap1, ids = load_VOC_pattern_image(join(self.inference_root, task_name1, 'acc_colormaps'),
                                                    self.datalist_path,
                                                    ext=ext)
        acc_colormap2, ids = load_VOC_pattern_image(join(self.inference_root, task_name2, 'acc_colormaps'),
                                                    self.datalist_path,
                                                    ext=ext)
        acc_diff_colormaps = [255 - (each_acc_colormap1 - each_acc_colormap2) for
                              (each_acc_colormap1, each_acc_colormap2) in zip(acc_colormap1, acc_colormap2)]
        return acc_diff_colormaps

    def compare_tasks_improve(self, task_name1, task_name2):
        acc_diff_colormaps = self.get_acc_diff_map(task_name1, task_name2)
        images, ids = load_VOC_pattern_image(join(data_home, 'JPEGImages'), self.datalist_path, ext=self.image_ext)
        colormaps1, _ = load_VOC_pattern_image(join(self.inference_root, task_name1, 'colormaps'), self.datalist_path,
                                               ext=self.image_ext)
        colormaps2, _ = load_VOC_pattern_image(join(self.inference_root, task_name2, 'colormaps'), self.datalist_path,
                                               ext=self.image_ext)
        masked = get_masked_images(images, acc_diff_colormaps, alpha=0.4)
        comparisons = get_comparison([images, acc_diff_colormaps, masked, images, colormaps1, colormaps2])
        save_images(join(self.inference_root, 'comparisons', f'imporve_{task_name1}_%_{task_name2}'), comparisons,
                    [f'{each_id}' for each_id in ids], 'tif')

    def run(self, task_names):
        # Get basic info for each task.
        [self.run_get_basic_info(each_task_name) for each_task_name in task_names]
        # Get a txt including all miou for each task.
        self.compare_miou_txt(task_names)
        # Get a image showing the regions of improvements and degeneracies.
        self.compare_tasks_improve(self.task_name, self.secondary_task_name)
        # Put 3 stitched aggregated images into a single image.
        self.compare_comparson(task_names)

        comparison3groups, ids = load_VOC_pattern_image(join(self.inference_root, 'comparisons', 'comparisons3groups'),
                                                        self.datalist_path, ext='jpg')
        impro, _ = load_VOC_pattern_image(join(self.inference_root, 'comparisons',
                                               'imporve_trainaug#43#10#0.007#1e-06#45500#0_%_trainaug#43#10#0.007#1e-06#45500#2'),
                                          self.datalist_path, ext='tif')
        aggregated_com_impro = get_comparison([comparison3groups, impro], shape=(2, 1))
        save_images(join(self.inference_root, 'comparisons', 'aggregated3groups'), aggregated_com_impro, ids, 'tif')

        # compare_tasks_improve(task_name, task_name2)

        # comparison_multiple_sets('heatmaps', 'valtest.txt', 'png', 'train#30#8#0.007#1e-06#36000#0',
        #                          'train#30#8#0.007#1e-06#36000#1', 'train#30#8#0.007#1e-06#36000#2')
