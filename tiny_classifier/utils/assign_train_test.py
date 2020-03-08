import os
import glob
import shutil
import random

root = "./ROI_data"

from_dir = os.path.join(root, "ALL")
train_dir = os.path.join(root, "train")
test_dir = os.path.join(root, "test")


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


mkdir(train_dir)
mkdir(test_dir)

ratio_train = 0.8

for sub_dir in os.listdir(from_dir):
    print("processing %s ...." % sub_dir)
    new_dir = os.path.join(from_dir, sub_dir)
    file_list = glob.glob(new_dir + "/*.jpg")

    total_len = len(file_list)
    num_train = int(total_len * ratio_train)

    train_list = random.sample(file_list, num_train)
    # print(len(train_list))

    for jpg in file_list:
        basename = os.path.basename(jpg)
        if jpg in train_list:
            train_subclass_dir = os.path.join(train_dir, sub_dir)
            mkdir(train_subclass_dir)
            shutil.copyfile(jpg, os.path.join(train_subclass_dir, basename))
        else:
            test_subclass_dir = os.path.join(test_dir, sub_dir)
            mkdir(test_subclass_dir)
            shutil.copyfile(jpg, os.path.join(test_subclass_dir, basename))
