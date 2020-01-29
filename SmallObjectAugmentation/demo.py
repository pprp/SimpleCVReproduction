import aug as am
import Helpers as hp
from util import *
import os
from os.path import join
from tqdm import tqdm
import random

base_dir = os.getcwd()

save_base_dir = join(base_dir, 'save')

check_dir(save_base_dir)

imgs_dir = [f.strip() for f in open(join(base_dir, 'train.txt')).readlines()]
labels_dir = hp.replace_labels(imgs_dir)

small_imgs_dir = [f.strip() for f in open(join(base_dir, 'small.txt')).readlines()]
random.shuffle(small_imgs_dir)

for image_dir, label_dir in tqdm(zip(imgs_dir, labels_dir)):
    small_img = []
    for x in range(8):
        if small_imgs_dir == []:
            #exit()
            small_imgs_dir = [f.strip() for f in open(join(base_dir,'small.txt')).readlines()]
            random.shuffle(small_imgs_dir)
        small_img.append(small_imgs_dir.pop())
    am.copysmallobjects2(image_dir, label_dir, save_base_dir,small_img)
