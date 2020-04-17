import os

import shutil

root_dir = "./data/WIDER_FACE_rect/images"

for dir_name in os.listdir(root_dir):
    new_dir = os.path.join(root_dir, dir_name)
    for img_name in os.listdir(new_dir):
        new_name = dir_name + "_" + img_name

        shutil.move(os.path.join(new_dir, img_name), os.path.join(root_dir, new_name))