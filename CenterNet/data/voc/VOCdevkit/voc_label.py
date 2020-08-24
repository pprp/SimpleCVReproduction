# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 11:42:13 2018
将本文件放到VOC2007目录下，然后就可以直接运行
需要修改的地方：
1. sets中替换为自己的数据集
2. classes中替换为自己的类别
3. 将本文件放到VOC2007目录下
4. 直接开始运行
"""

import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
sets = [('2007', 'train'), ('2007', 'val'), ('2007', 'test')]  # 替换为自己的数据集
classes = ["raccoon"]  # 修改为自己的类别
#classes = ["eye", "nose"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x, y, w, h)


def convert_annotation(year, image_id):
    in_file = open('VOC%s/Annotations/%s.xml' %
                   (year, image_id))  # 将数据集放于当前目录下
    out_file = open('VOC%s/labels/%s.txt' % (year, image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(
            xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " +
                       " ".join([str(a) for a in bb]) + '\n')


wd = getcwd()
for year, image_set in sets:
    if not os.path.exists('VOC%s/labels/' % (year)):
        os.makedirs('VOC%s/labels/' % (year))
    image_ids = open('VOC%s/ImageSets/Main/%s.txt' %
                     (year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt' % (year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('VOC%s/JPEGImages/%s.jpg\n' % (year, image_id))
        convert_annotation(year, image_id)
    list_file.close()
# os.system("cat 2007_train.txt 2007_val.txt > train.txt")     #修改为自己的数据集用作训练
