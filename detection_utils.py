from operator import itemgetter

import tensorflow as tf
import numpy as np
import skimage.io as sio
from os.path import join
from skimage.draw import polygon
import cv2
import os
from pprint import pprint
from collections import Counter


def to_top_left_box_coordiantes(detections):
    """
    Convert bounding box of attributes (center_height, center_center_width, height, width) to top_left-bottom_right form.
    """
    center_height, center_width, height, width, others = tf.split(detections, [1, 1, 1, 1, -1], -1)
    half_height = height / 2
    half_width = width / 2

    top = center_height - half_height
    left = center_width - half_width
    bottom = center_height + half_height
    right = center_width + half_width

    return tf.concat([top, left, bottom, right, others], -1)


def bounding_box_iou(box1, box2):
    """
    Calculate the iou of bounding boxes.
    """
    b1_top, b1_left, b1_bottom, b1_right = box1
    b2_top, b2_left, b2_bottom, b2_right = box2

    intersection_top = max(b1_top, b2_top)
    intersection_bottom = min(b1_bottom, b2_bottom)
    intersection_left = max(b1_left, b2_left)
    intersection_right = min(b1_right, b2_right)

    intersection_area = (intersection_bottom - intersection_top) * (intersection_right - intersection_left)
    box1_area = (b1_bottom - b1_top) * (b1_right - b1_left)
    box2_area = (b2_bottom - b2_top) * (b2_right - b2_left)

    return intersection_area / (box1_area + box2_area - intersection_area + 1e-5)


def non_max_supression(predictions, confidence, iou_threshold=0.4):
    pass


anchors = []
anchor_dict = {}
anchor_file = r'D:\Datasets\competitions\DC\DC Traffic\data\train_1w.csv'
with open(anchor_file, encoding='utf8') as f:
    f.readline()
    for each_line in f:
        # print(each_line)
        file_name, each_line = each_line.strip().split(',')
        if not each_line:
            continue
        raw_anchors = each_line.split(';')
        this_anchors = [list(map(float, each_raw_anchor.split('_'))) for each_raw_anchor in raw_anchors if
                        each_raw_anchor]
        anchor_dict[file_name] = this_anchors
        anchors += this_anchors
anchors = np.array(anchors)

image_home = r'D:\Datasets\competitions\DC\DC Traffic\data\train_1w_processed\IMG'
bbox_home = r'D:\Datasets\competitions\DC\DC Traffic\data\train_1w_processed\bbox'

# for each_image in os.listdir(image_home):
#     this_image = sio.imread(join(image_home, each_image))
#     this_anchors = anchor_dict[each_image]
#     for each_anchor in this_anchors:
#         top_left = tuple([int(each_anchor[0]), int(each_anchor[1])])
#         bottom_right = tuple([int(each_anchor[0] + each_anchor[2]), int(each_anchor[1] + each_anchor[3])])
#
#         this_image_bbox = cv2.rectangle(this_image, top_left, bottom_right, (255, 0, 0), 2)
#         sio.imsave(join(bbox_home, each_image), this_image_bbox)

print(anchors)

ratios = anchors[:, 2] / anchors[:, 3]
ratios = np.reshape(ratios, (-1, 1))

areas = np.sqrt((anchors[:, 2] * anchors[:, 3]).reshape((-1, 1)))
print(np.min(areas))
print(np.max(areas))

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

clusters = KMeans(3)
clusters.fit(ratios)
counts = sorted(Counter(clusters.labels_).items())
centers = list(clusters.cluster_centers_.ravel())
pprint(sorted(list(zip(centers, counts))))
t = plt.hist(ratios, 100)
t = list(zip(list(t[0].ravel()), list(t[1].ravel())))
print(t)


plt.show()

clusters = KMeans(4)
clusters.fit(areas)
counts = sorted(Counter(clusters.labels_).items())
centers = list(clusters.cluster_centers_.ravel())
pprint(sorted(list(zip(centers, counts))))
t = plt.hist(areas, 100)
t = list(zip(list(t[0].ravel()), list(t[1].ravel())))


