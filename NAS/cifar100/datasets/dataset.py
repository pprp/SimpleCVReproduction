# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import random

import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100
from torchvision.datasets.cifar import CIFAR10, CIFAR100

from datasets.transforms import DatasetTransforms


class ArchDataSet(torch.utils.data.Dataset):
    def __init__(self, path):
        super(ArchDataSet, self).__init__()
        assert path is not None
        with open(path, "r") as f:
            self.arch_dict = json.load(f)

        self.arc_list = []
        self.arc_key = []

        for key, v in self.arch_dict.items():
            self.arc_list.append(v["arch"])
            self.arc_key.append(key)

    def __getitem__(self, index):
        tmp_arc = self.arc_list[index]
        tmp_key = self.arc_key[index]
        return tmp_key, tmp_arc

    def __len__(self):
        return len(self.arch_dict)


class ArchLoader():
    '''
    load arch from json file
    '''

    def __init__(self, path):
        super(ArchLoader, self).__init__()

        self.arc_list = []
        self.arc_dict = {}
        self.get_arch_list_dict(path)
        self.idx = -1

        self.level_config = {
            "level1": [4, 8, 12, 16],
            "level2": [4, 8, 12, 16, 20, 24, 28, 32],
            "level3": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        }

    def get_arch_list(self):
        arc_int_list = []
        for str_arc in self.arc_list:
            arc_int_list.append(self.convert_str_arc_list(str_arc))
        return arc_int_list

    def get_arch_dict(self):
        return self.arc_dict

    def get_arch_list_dict(self, path):
        with open(path, "r") as f:
            self.arc_dict = json.load(f)

        self.arc_list = []

        for _, v in self.arc_dict.items():
            self.arc_list.append(v["arch"])

    def convert_list_arc_str(self, arc_list):
        arc_str = ""
        arc_list = [str(item)+"-" for item in arc_list]
        for item in arc_list:
            arc_str += item
        return arc_str[:-1]

    def convert_str_arc_list(self, arc_str):
        return [int(i) for i in arc_str.split('-')]

    def generate_spos_like_batch(self):
        rngs = []
        for i in range(0, 7):
            rngs += np.random.choice(
                self.level_config["level1"], size=1).tolist()
        for i in range(7, 13):
            rngs += np.random.choice(
                self.level_config['level2'], size=1).tolist()
        for i in range(13, 20):
            rngs += np.random.choice(
                self.level_config['level3'], size=1).tolist()
        return np.array(rngs)

    def generate_niu_fair_batch(self, seed):
        rngs = []
        seed = seed
        random.seed(seed)
        # level1
        for i in range(0, 7):
            tmp_rngs = []
            for _ in range(4):
                tmp_rngs.extend(random.sample(self.level_config['level1'],
                                              len(self.level_config['level1'])))
            rngs.append(tmp_rngs)
        # level2
        for i in range(7, 13):
            tmp_rngs = []
            for _ in range(2):
                tmp_rngs.extend(random.sample(self.level_config['level2'],
                                              len(self.level_config['level2'])))
            rngs.append(tmp_rngs)

        # level3
        for i in range(13, 20):
            rngs.append(random.sample(self.level_config['level3'],
                                      len(self.level_config['level3'])))
        return np.transpose(rngs)

    def generate_width_to_narrow(self, current_epoch, total_epoch):
        current_p = float(current_epoch) / total_epoch
        opposite_p = 1 - current_p
        # print(current_p, opposite_p)

        def p_generator(length):
            rng_list = np.linspace(current_p, opposite_p, length)
            return self.softmax(rng_list)

        rngs = []
        for i in range(0, 7):
            rngs += np.random.choice(self.level_config["level1"], size=1, p=p_generator(
                len(self.level_config['level1']))).tolist()
        for i in range(7, 13):
            rngs += np.random.choice(self.level_config['level2'], size=1, p=p_generator(
                len(self.level_config['level2']))).tolist()
        for i in range(13, 20):
            rngs += np.random.choice(self.level_config['level3'], size=1, p=p_generator(
                len(self.level_config['level3']))).tolist()
        return np.array(rngs)

    @staticmethod
    def softmax(rng_lst):
        if isinstance(rng_lst, list):
            x = np.asarray(rng_lst)
        else:
            x = rng_lst
        x_max = x.max()
        x = x - x_max
        x_exp = np.exp(x)
        x_exp_sum = x_exp.sum(axis=0)
        softmax = x_exp / x_exp_sum
        return softmax


def get_train_loader(batch_size, num_workers, clss='cifar10', cutout=0):
    assert clss in ['cifar10', 'cifar100']

    # 1. get transform
    dt = DatasetTransforms(clss=clss, cutout=cutout)

    # 2. get dataset
    if clss == 'cifar10':
        train_dataset = CIFAR10(root="./data", train=True,
                                download=True, transform=dt.get_train_transforms())
    elif clss == 'cifar100':
        train_dataset = CIFAR100(
            root="./data", train=True, download=True, transform=dt.get_train_transforms())
    else:
        raise "Not support %s" % clss

    # 3. get dataloader

    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size, drop_last=True)

    return train_loader


def get_val_loader(batch_size, num_workers, clss='cifar10'):
    assert clss in ['cifar10', 'cifar100']

    # 1. get transform
    dt = DatasetTransforms(clss)

    # 2. get dataset
    if clss == 'cifar10':
        val_dataset = CIFAR10(root="./data", train=False,
                              download=True, transform=dt.get_val_transform())
    elif clss == 'cifar100':
        val_dataset = CIFAR100(
            root="./data", train=False, download=True, transform=dt.get_val_transform())
    else:
        raise "Not support %s" % clss

    # 3. get dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return val_loader
