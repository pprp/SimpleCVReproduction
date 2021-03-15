

'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from cutout import Cutout
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import sys
import argparse
import logging
import time
# from models.vgg import CNN
from utils import progress_bar
from dropblock import DropBlock2D, LinearScheduler
from torch.nn import Module
import nni

# torch.multiprocessing.set_sharing_strategy('file_system')


# from torchviz import make_dot

import os
# os.environ["PATH"] += os.pathsep + 'D:/Graphviz2.38/bin'


class SiLU(Module):
    """SiLU activation function (also known as Swish): x * sigmoid(x)."""

    # Note: will be part of Pytorch 1.7, at which point can remove this.

    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


def active_fn(cfg):
    activation_fun = cfg.active_function
    if activation_fun == "relu":
        return nn.ReLU(inplace=True)
    elif activation_fun == "silu":
        return SiLU()


class SqueezeAndExcitation(nn.Module):
    """Squeeze-and-Excitation module.

    See: https://arxiv.org/abs/1709.01507
    """

    def __init__(self,
                 n_feature,
                 n_hidden,
                 spatial_dims=[2, 3],
                 activation_fun=None,
                 ):
        super(SqueezeAndExcitation, self).__init__()
        self.n_feature = n_feature
        self.n_hidden = n_hidden
        self.spatial_dims = spatial_dims
        self.se_reduce = nn.Conv2d(n_feature, n_hidden, 1, bias=True)
        self.se_expand = nn.Conv2d(n_hidden, n_feature, 1, bias=True)
        if activation_fun == "relu":
            self.active_fn = nn.ReLU(inplace=True)
        elif activation_fun == "silu":
            self.active_fn = SiLU()

    def forward(self, x):
        se_tensor = x.mean(self.spatial_dims, keepdim=True)
        se_tensor = self.se_expand(self.active_fn(self.se_reduce(se_tensor)))
        return torch.sigmoid(se_tensor) * x
#


class CXH_plain(Module):
    def __init__(self):
        super(CXH_plain, self).__init__()
        self.stem = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3)
        )
        self.a1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(48, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256)
        )
        self.a2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.b = nn.Sequential(
            nn.Conv2d(259, 256, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),

        )
        self.c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0),
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.25)
        )
        self.f1 = nn.Linear(64, 64)
        self.active = nn.ReLU(inplace=True)
        self.f2 = nn.Linear(64, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stem = self.stem(x)
        a1 = self.a1(stem)
        a2 = self.a2(stem)
        bf = torch.cat((a1, a2), 1)
        b = self.b(bf)
        c1 = self.c1(b)
        c2 = self.c2(b)
        df = c1+c2
        d = self.d(df)
        d = torch.flatten(d, 1)
        d = self.f1(d)
        d = self.active(d)
        d = self.f2(d)
        return d


class CXH_SE(Module):
    def __init__(self):
        super(CXH_SE, self).__init__()
        self.stem = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(3)
        )
        self.a1 = nn.Sequential(
            nn.BatchNorm2d(3),
            nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=1),
            SqueezeAndExcitation(48, 3, activation_fun='relu'),
            nn.Conv2d(48, 48, kernel_size=1, stride=1, padding=0),
            SqueezeAndExcitation(48, 3, activation_fun='relu'),
            nn.Conv2d(48, 512, kernel_size=3, stride=1, padding=1),
            SqueezeAndExcitation(512, 32, activation_fun='relu'),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0),
            SqueezeAndExcitation(256, 16, activation_fun='relu'),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            SqueezeAndExcitation(128, 8, activation_fun='relu'),
            nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2),
            SqueezeAndExcitation(256, 16, activation_fun='relu'),
            nn.BatchNorm2d(256)
        )
        self.a2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True)
        )
        self.b = nn.Sequential(
            nn.Conv2d(259, 256, kernel_size=1, stride=1, padding=0),
            SqueezeAndExcitation(256, 16, activation_fun='relu'),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            SqueezeAndExcitation(512, 32, activation_fun='relu'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, kernel_size=5, stride=1, padding=2),
            SqueezeAndExcitation(1024, 64, activation_fun='relu'),
        )
        self.c1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 64, kernel_size=3, stride=1, padding=1),
            SqueezeAndExcitation(64, 4, activation_fun='relu'),
            nn.ReLU(inplace=True)
        )
        self.c2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0),
            SqueezeAndExcitation(64, 4, activation_fun='relu'),
        )
        self.d = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.25)
        )
        self.f1 = nn.Linear(64, 64)
        self.active = nn.ReLU(inplace=True)
        self.f2 = nn.Linear(64, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stem = self.stem(x)
        a1 = self.a1(stem)
        a2 = self.a2(stem)
        bf = torch.cat((a1, a2), 1)
        b = self.b(bf)
        c1 = self.c1(b)
        c2 = self.c2(b)
        df = c1+c2
        d = self.d(df)
        d = torch.flatten(d, 1)
        d = self.f1(d)
        d = self.active(d)
        d = self.f2(d)
        return d


class CXH(Module):
    def __init__(self):
        super(CXH, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.a1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256),
        )
        self.a2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
        )
        self.b = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256)
        )
        self.d = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.25)
        )
        self.f1 = nn.Linear(64, 64)
        self.active = nn.ReLU(inplace=True)
        self.f2 = nn.Linear(64, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stem = self.stem(x)
        a1 = self.a1(stem)
        a2 = self.a2(stem)
        bf = a1+a2
        b = self.b(bf)
        c1 = self.c1(b)
        df = c1+b
        d = self.d(df)
        d = torch.flatten(d, 1)
        d = self.f1(d)
        d = self.active(d)
        d = self.f2(d)
        return d


class CXH_Squeeze_Excitation(Module):
    # RELU
    def __init__(self):
        super(CXH_Squeeze_Excitation, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # se
            SqueezeAndExcitation(64, 16, activation_fun='relu'),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # se
            SqueezeAndExcitation(64, 16, activation_fun='relu')
        )
        self.a1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256),
            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.a2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.b = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),

            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.d = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeAndExcitation(256, 64, activation_fun='relu'),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SqueezeAndExcitation(256, 64, activation_fun='relu'),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            SqueezeAndExcitation(64, 16, activation_fun='relu'),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.25)
        )
        self.f1 = nn.Linear(64, 64)
        self.active = nn.ReLU(inplace=True)
        self.f2 = nn.Linear(64, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stem = self.stem(x)
        a1 = self.a1(stem)
        a2 = self.a2(stem)
        bf = a1+a2
        b = self.b(bf)
        c1 = self.c1(b)
        df = c1+b
        d = self.d(df)
        d = torch.flatten(d, 1)
        d = self.f1(d)
        d = self.active(d)
        d = self.f2(d)
        return d



class CXH_SE_SWISH(Module):
    # RELU
    def __init__(self):
        super(CXH_SE_SWISH, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),

            # se
            SqueezeAndExcitation(64, 16, activation_fun='relu'),

            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),

            # se
            SqueezeAndExcitation(64, 16, activation_fun='relu')
        )
        self.a1 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(256),
            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.a2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.b = nn.Sequential(
            nn.Hardswish(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),

            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.c1 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),

            SqueezeAndExcitation(256, 64, activation_fun='relu')
        )
        self.d = nn.Sequential(
            nn.Hardswish(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
            SqueezeAndExcitation(256, 64, activation_fun='relu'),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.Hardswish(inplace=True),
            SqueezeAndExcitation(256, 64, activation_fun='relu'),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.Hardswish(inplace=True),
            SqueezeAndExcitation(64, 16, activation_fun='relu'),

            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.AdaptiveAvgPool2d(1),
            nn.Dropout(p=0.25)
        )
        self.f1 = nn.Linear(64, 64)
        self.active = nn.Hardswish(inplace=True)
        self.f2 = nn.Linear(64, 10)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stem = self.stem(x)
        a1 = self.a1(stem)
        a2 = self.a2(stem)
        bf = a1+a2
        b = self.b(bf)
        c1 = self.c1(b)
        df = c1+b
        d = self.d(df)
        d = torch.flatten(d, 1)
        d = self.f1(d)
        d = self.active(d)
        d = self.f2(d)
        return d

if __name__ == '__main__':
    net = CXH()
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
    # g = make_dot(y)
    # g.render('EENA', view=True)
