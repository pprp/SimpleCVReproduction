import os
import random
from os.path import join

# from tqdm import tqdm

import aug as am
import Helpers as hp
from util import *

base_dir = os.getcwd()

save_base_dir = join(base_dir, 'dpjsave')

check_dir(save_base_dir)

imgs_dir = [f.strip() for f in open(join(base_dir, 'sea.txt')).readlines()]
labels_dir = hp.replace_labels(imgs_dir)

small_imgs_dir = [f.strip() for f in open(join(base_dir, 'dpj_small.txt')).readlines()]
random.shuffle(small_imgs_dir)

times = 3

for image_dir, label_dir in zip(imgs_dir, labels_dir):
    print(image_dir, label_dir)
    small_img = []
    for x in range(times):
        if small_imgs_dir == []:
            #exit()
            small_imgs_dir = [f.strip() for f in open(join(base_dir,'dpj_small.txt')).readlines()]
            random.shuffle(small_imgs_dir)
        small_img.append(small_imgs_dir.pop())
    # print("ok")
    am.copysmallobjects2(image_dir, label_dir, save_base_dir,small_img, times)
