# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json

import torch
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datasets.transforms import Cutout


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


class DatasetTransforms:
    def __init__(self, clss, cutout=0):
        if clss == 'cifar10':
            self.mean = CIFAR10_MEAN
            self.std = CIFAR10_STD
        elif clss == 'cifar100':
            self.mean = CIFAR100_MEAN
            self.std = CIFAR100_STD
        else:
            print("Not Support %s dataset." % clss)
        self.cutout = 0

    def _get_cutout(self):
        if self.cutout == 0:
            return None
        else:
            return Cutout(self.cutout)

    def _get_default_transforms(self):
        default_configure = T.Compose([
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(),
            T.RandomResizedCrop((32, 32)),  # for cifar10 or cifar100
            T.RandomRotation(15)
        ])
        return default_configure

    def get_train_transforms(self):
        default_conf = self._get_default_transforms()
        cutout = self._get_cutout()
        if cutout is None:
            train_transform = T.Compose([default_conf,
                                         T.ToTensor(),
                                         T.Normalize(self.mean, self.std)
                                         ])
        else:

            train_transform = T.Compose([default_conf,
                                         T.ToTensor(),
                                         cutout,
                                         T.Normalize(self.mean, self.std)
                                         ])

        return train_transform

    def get_val_transform(self):
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])


def get_train_loader(batch_size, num_workers, clss='cifar10', cutout=0):
    assert clss in ['cifar10', 'cifar100']

    # 1. get transform
    dt = DatasetTransforms(clss=clss, cutout=cutout)

    # 2. get dataset
    if clss == 'cifar10':
        train_dataset = CIFAR10(root="./data", train=True,
                                download=True, transforms=dt)
    elif clss == 'cifar100':
        train_dataset = CIFAR100(
            root="./data", train=True, download=True, transforms=dt)
    else:
        raise "Not support %s" % clss

    # 3. get dataloader

    train_loader = torch.utils.data.DataLoader(
        train_dataset, num_workers=num_workers, pin_memory=True, batch_size=batch_size, drop_last=True)

    return train_loader


def get_valid_loader(batch_size, num_workers, clss='cifar10'):
    assert clss in ['cifar10', 'cifar100']

    # 1. get transform
    dt = DatasetTransforms(clss)

    # 2. get dataset
    if clss == 'cifar10':
        val_dataset = CIFAR10(root="./data", train=False,
                              download=True, transforms=dt)
    elif clss == 'cifar100':
        val_dataset = CIFAR100(
            root="./data", train=False, download=True, transforms=dt)
    else:
        raise "Not support %s" % clss

    # 3. get dataloader
    val_loader = torch.utils.data.Dataloader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return val_loader
