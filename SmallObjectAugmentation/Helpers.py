import glob
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
# import random
import math
from tqdm import tqdm


def load_images(path):
    image_list = []
    images = glob.glob(path)
    for index in range(len(images)):
        image = cv2.cvtColor(cv2.imread(images[index]), cv2.COLOR_BGR2RGB)
        image_list.append(image)
        # image_list.append(cv2.resize(image,(1280,720)))

    return image_list


def read_images(path):
    images = glob.glob(path)
    return images


def load_images_from_path(path):
    image_list = []
    for p in tqdm(path):
        image = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        image_list.append(image)
    return image_list


def replace_labels(path):
    labelpath = []
    for p in path:
        labelpath.append(p.replace('.jpg', '.txt'))
    return labelpath
