import os
import glob
import shutil
from os.path import join
import random

root_dir = "./data/reid"
train_dir = "./data/train"
val_dir = "./data/val"

train_percent = 0.6
val_percent = 0.4


def mkdir_if_not_exist(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        print("%s exists." % dir)


class_full_path = glob.glob(join(root_dir, "*"))

for i in range(len(class_full_path)):
    class_name = os.path.basename(class_full_path[i])

    train_new_dir = join(train_dir, class_name)
    val_new_dir = join(val_dir, class_name)

    mkdir_if_not_exist(train_new_dir)
    mkdir_if_not_exist(val_new_dir)

    all_class_files = glob.glob(join(class_full_path[i], "*.jpg"))

    train_class_files = random.sample(
        all_class_files, int(len(all_class_files) * train_percent))

    for file_path in all_class_files:
        print("processing %s." % (file_path))
        if file_path in train_class_files:
            # assign to train folder
            shutil.copy(file_path, join(train_new_dir, os.path.basename(file_path)))
        else:
            # assign to val folder
            shutil.copy(file_path, join(val_new_dir, os.path.basename(file_path)))
