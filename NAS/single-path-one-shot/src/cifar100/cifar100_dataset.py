# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import numpy as np
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR100


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

def get_dataset(cls, cutout_length=0):
    MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
    STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),

    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]
    cutout = []
    if cutout_length > 0:
        cutout.append(Cutout(cutout_length))

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((32,32)),
        transforms.ColorJitter(0.2,0.2,0.2,0.2),
        normalize
    ])
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar100":
        dataset_train = CIFAR100(
            root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR100(
            root="./data", train=False, download=True, transform=valid_transform)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid
