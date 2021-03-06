#! -*- coding: utf-8 -*-

from __future__ import print_function, division

import os

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TRAIN_DIR = 'train'
VALIDATION_DIR = 'valid'

MEAN_RGB = (0.485, 0.456, 0.406)
VAR_RGB = (0.229, 0.224, 0.225)

transform_train = transforms.Compose([
    transforms.RandomSizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_RGB, VAR_RGB),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN_RGB, VAR_RGB),
])


def get_imagenet_dataset(batch_size, dataset_root='./dataset/imagenet/', dataset_tpye='train'):
    if dataset_tpye == 'train':
        train_dataset_root = os.path.join(dataset_root, TRAIN_DIR)
        trainset = datasets.ImageFolder(root=train_dataset_root, transform=transform_train)
        trainloader = DataLoader(trainset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=8,
                                 pin_memory=True,
                                 drop_last=False)
        print('Succeed to init ImageNet train DataLoader!')
        return trainloader
    elif dataset_tpye == 'val' or dataset_tpye == 'valid':
        val_dataset_root = os.path.join(dataset_root, VALIDATION_DIR)
        valset = datasets.ImageFolder(root=val_dataset_root, transform=transform_test)
        valloader = DataLoader(valset,
                               batch_size=batch_size,
                               shuffle=False,
                               num_workers=8,
                               pin_memory=False,
                               drop_last=False)
        print('Succeed to init ImageNet val DataLoader!')
        return valloader
    else:
        raise Exception('IMAGENET DataLoader: Unknown dataset type -- %s' % dataset_tpye)
