import os
import shutil
import numpy as np
import random
from os.path import join
save_path = "./dpjsamplecrop"



image_root_dir = r"D:\GitHub\voc2007_for_yolo_torch\images"
label_root_dir = r"D:\GitHub\voc2007_for_yolo_torch\labels"

fo = open("dpj_train.txt", "w")

for ids in os.listdir(image_root_dir):
    print("ids:%s" % ids)

    dir_image = os.path.join(image_root_dir, ids)
    dir_label = os.path.join(label_root_dir, ids)
    print(len(os.listdir(dir_image)))

    selected = random.sample(os.listdir(dir_image), 8)
    print(selected)

    for s in selected:
        fo.write(join(save_path, s)+"\n")
        shutil.copy(join(dir_image, s), join(save_path, s))
        s = s.replace("jpg", 'txt')
        shutil.copy(join(dir_label, s), join(save_path, s))

fo.close()