import os
import cv2
import numpy as np
from os.path import join, split
import random


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def issmallobject(bbox, thresh):
    if bbox[0] * bbox[1] <= thresh:
        return True
    else:
        return False


def read_label_txt(label_dir):
    labels = []
    with open(label_dir) as fp:
        for f in fp.readlines():
            labels.append(f.strip().split(' '))
    return labels


def load_txt_label(label_dir):
    return np.loadtxt(label_dir, dtype=str)


def load_txt_labels(label_dir):
    labels = []
    for l in label_dir:
        la = load_txt_label(l)
        labels.append(la)
    return labels


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def rescale_yolo_labels(labels, img_shape):
    height, width, nchannel = img_shape
    rescale_boxes = []
    for box in list(labels):
        x_c = float(box[1]) * width
        y_c = float(box[2]) * height
        w = float(box[3]) * width
        h = float(box[4]) * height
        x_left = x_c - w * .5
        y_left = y_c - h * .5
        x_right = x_c + w * .5
        y_right = y_c + h * .5
        rescale_boxes.append([box[0], int(x_left), int(y_left), int(x_right), int(y_right)])
    return rescale_boxes


def draw_annotation_to_image(img, annotation, save_img_dir):
    for anno in annotation:
        cl, x1, y1, x2, y2 = anno
        cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 0, 0))
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, cl, (int((x1 + x2) / 2), y1 - 5), font, fontScale=0.8, color=(0, 0, 255))
    cv2.imwrite(save_img_dir, img)


def bbox_iou(box1, box2):
    cl, b1_x1, b1_y1, b1_x2, b1_y2 = box1
    cl, b2_x1, b2_y1, b2_x2, b2_y2 = box2
    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = max(b1_x1, b2_x1)
    inter_rect_y1 = max(b1_y1, b2_y1)
    inter_rect_x2 = min(b1_x2, b2_x2)
    inter_rect_y2 = min(b1_y2, b2_y2)
    # Intersection area
    inter_width = inter_rect_x2 - inter_rect_x1 + 1
    inter_height = inter_rect_y2 - inter_rect_y1 + 1
    if inter_width > 0 and inter_height > 0:  # strong condition
        inter_area = inter_width * inter_height
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
        iou = inter_area / (b1_area + b2_area - inter_area)
    else:
        iou = 0
    return iou


def swap(x1, x2):
    if (x1 > x2):
        temp = x1
        x1 = x2
        x2 = temp
    return x1, x2


def norm_sampling(search_space):
    # 随机生成点
    search_x_left, search_y_left, search_x_right, search_y_right = search_space

    search_x_left = int(search_x_left)
    search_x_right = int(search_x_right)
    search_y_left = int(search_y_left)
    search_y_right = int(search_y_right)

    new_bbox_x_center = random.randint(search_x_left, search_x_right)
    # print(search_y_left, search_y_right, '=')
    new_bbox_y_center = random.randint(search_y_left, search_y_right)
    return [new_bbox_x_center, new_bbox_y_center]


def flip_bbox(roi):
    roi = roi[:, ::-1, :]
    return roi


def sampling_new_bbox_center_point(img_shape, bbox):
    #### sampling space ####
    height, width, nc = img_shape
    cl, x_left, y_left, x_right, y_right = bbox
    bbox_w, bbox_h = x_right - x_left, y_right - y_left
    ### left top ###
    if x_left <= width / 2:
        search_x_left, search_y_left, search_x_right, search_y_right = width * 0.6, height / 2, width * 0.75, height * 0.75
    if x_left > width / 2:
        search_x_left, search_y_left, search_x_right, search_y_right = width * 0.25, height / 2, width * 0.5, height * 0.75
    return [search_x_left, search_y_left, search_x_right, search_y_right]


def random_add_patches(bbox, rescale_boxes, shape, paste_number, iou_thresh):
    temp = []
    for rescale_bbox in rescale_boxes:
        temp.append(rescale_bbox)
    cl, x_left, y_left, x_right, y_right = bbox
    bbox_w, bbox_h = x_right - x_left, y_right - y_left
    center_search_space = sampling_new_bbox_center_point(shape, bbox)
    success_num = 0
    new_bboxes = []
    while success_num < paste_number:
        new_bbox_x_center, new_bbox_y_center = norm_sampling(center_search_space)
        print(norm_sampling(center_search_space))
        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = new_bbox_x_center - 0.5 * bbox_w, \
                                                                               new_bbox_y_center - 0.5 * bbox_h, \
                                                                               new_bbox_x_center + 0.5 * bbox_w, \
                                                                               new_bbox_y_center + 0.5 * bbox_h
        new_bbox = [cl, int(new_bbox_x_left), int(new_bbox_y_left), int(new_bbox_x_right), int(new_bbox_y_right)]
        ious = [bbox_iou(new_bbox, bbox_t) for bbox_t in rescale_boxes]
        if max(ious) <= iou_thresh:
            # for bbox_t in rescale_boxes:
            # iou =  bbox_iou(new_bbox[1:],bbox_t[1:])
            # if(iou <= iou_thresh):
            success_num += 1
            temp.append(new_bbox)
            new_bboxes.append(new_bbox)
        else:
            continue
    return new_bboxes


def sampling_new_bbox_center_point2(img_shape, bbox):
    #### sampling space ####
    height, width, nc = img_shape
    bbox_h, bbox_w, bbox_c = bbox
    ### left top ###
    '''
    search_x_left, search_y_left, search_x_right, search_y_right = width * 0.55 , height * 0.5 , \
                                                                   width * 0.9 , height * 0.95
    '''
    search_x_left, search_y_left, search_x_right, search_y_right = width * 0.35 , height * 0.6 , \
                                                                   width * 1 , height * 0.95

    return [search_x_left, search_y_left, search_x_right, search_y_right]


def random_add_patches2(bbox_img, rescale_boxes, shape, paste_number, iou_thresh):
    temp = []
    for rescale_bbox in rescale_boxes:
        temp.append(rescale_bbox)
    bbox_h, bbox_w, bbox_c = bbox_img
    img_h,img_w,img_c = shape
    center_search_space = sampling_new_bbox_center_point2(shape, bbox_img)  # 选取生成随机点区域
    success_num = 0
    new_bboxes = []
    cl = 0

    # print(center_search_space,'+')

    while success_num < paste_number:
        new_bbox_x_center, new_bbox_y_center = norm_sampling(center_search_space)   # 随机生成点坐标
        if new_bbox_x_center-0.5*bbox_w < 0 or new_bbox_x_center+0.5*bbox_w > img_w:
            continue
        if new_bbox_y_center-0.5*bbox_h < 0 or new_bbox_y_center+0.5*bbox_h > img_h:
            continue
        new_bbox_x_left, new_bbox_y_left, new_bbox_x_right, new_bbox_y_right = new_bbox_x_center - 0.5 * bbox_w, \
                                                                               new_bbox_y_center - 0.5 * bbox_h, \
                                                                               new_bbox_x_center + 0.5 * bbox_w, \
                                                                               new_bbox_y_center + 0.5 * bbox_h
        new_bbox = [cl, int(new_bbox_x_left), int(new_bbox_y_left), int(new_bbox_x_right), int(new_bbox_y_right)]

        ious = [bbox_iou(new_bbox, bbox_t) for bbox_t in rescale_boxes]
        ious2 = [bbox_iou(new_bbox,bbox_t1) for bbox_t1 in new_bboxes]
        if ious2 == []:
            ious2.append(0)
        if max(ious) <= iou_thresh and max(ious2) <= iou_thresh:
            success_num += 1
            temp.append(new_bbox)
            new_bboxes.append(new_bbox)
        else:
            continue

    return new_bboxes
