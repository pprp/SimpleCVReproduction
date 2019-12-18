import torch
import torch.nn as nn

import os
import argparse

from config import cfg
from utils.visdom import Visualizer
from torchvision import datasets, models, transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    pass


if __name__ == "__main__":
    train_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train()
