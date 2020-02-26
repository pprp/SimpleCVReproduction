# coding=utf-8
import math
import os
import os.path as osp

import cv2
import numpy as np

SIZE_IMG = 255


#-中心旋转(x,y)坐标angle角度-----
def rotate_onepoint(x, y, angle):
    angle = math.pi * 2 - angle * math.pi / 180
    n = SIZE_IMG
    m = SIZE_IMG
    X = x * math.cos(angle) - y * math.sin(angle) - 0.5 * n * math.cos(
        angle) + 0.5 * m * math.sin(angle) + 0.5 * n
    Y = y * math.cos(angle) + x * math.sin(angle) - 0.5 * n * math.sin(
        angle) - 0.5 * m * math.cos(angle) + 0.5 * m
    return int(X), int(Y)


#-中心旋转标准飞机关键点集合p_sets中的点坐标，并存入savepath文件中-----
def rotate_anno(savepath, angle, p_sets):
    f_save = open(savepath, 'w')
    for p in p_sets:
        #-------------to do-----------------------
        x = p_sets[p][0]
        y = p_sets[p][1]
        #-------------to do-----------------------
        x, y = rotate_onepoint(x, y, angle)
        newline = str(p) + ' ' + str(x) + ' ' + str(y) + '\n'
        f_save.writelines(newline)
    f_save.close()


#-------------对path_read文件夹下的img_file图像文件，根据其对应的anno_file文件，绘制关键点并将结果存入path_save文件夹中-----
def draw_image_labels(img_file, anno_file, path_save, path_read):
    image = cv2.imread(os.path.join(path_read, img_file))
    with open(os.path.join(path_read, anno_file)) as f_anno:
        lines = f_anno.readlines()
        for line in lines:
            name_id, x, y = line.split()
            cv2.circle(image, (int(x), int(y)),
                       radius=1,
                       color=(0, 0, 255),
                       thickness=5)
            cv2.putText(image, str(name_id), (int(x) + 10, int(y) + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(os.path.join(path_save, img_file), image)


#-------------对数据集文件夹root_data下的子文件夹进行遍历，对里面的样本文件利用draw_image_labels绘制其关键点，并将所有结果保存在root_save文件夹下-----
def draw_points(root_data, root_save):
    flight_name = []
    for rr, dn, files in os.walk(root_data):
        if rr == root_data:
            flight_name = dn
            for d in flight_name:
                if not osp.exists(os.path.join(root_save, d)):
                    os.makedirs(os.path.join(root_save, d))
        elif rr.split(os.path.sep)[-1] in flight_name and len(
                flight_name) != 0 and len(files) != 0:
            data_name = rr.split(os.path.sep)[-1]
            for img_file in files:
                if osp.splitext(img_file)[1] == '.jpg':
                    anno_file = img_file.replace(img_file[-4:], '.txt')
                    draw_image_labels(img_file, anno_file,
                                      os.path.join(root_save, data_name),
                                      os.path.join(root_data, data_name))
