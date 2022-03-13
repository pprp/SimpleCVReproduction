# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import random
from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from matplotlib import cm, projections
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from autorf.components import *
from autorf.components import ReceptiveFieldAttention
from autorf.operations import *
from autorf.spaces import spatial_spaces
from Smodules import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

Genotype = namedtuple("Genotype", "normal normal_concat")

PRIMITIVES = spatial_spaces['hybrid']  # TODO more aligent way

RANDOMGENOTYPE = Genotype(
    normal=[(random.choice(PRIMITIVES), 0),
            (random.choice(PRIMITIVES), random.choice(
                [0, 1])), (random.choice(PRIMITIVES), random.choice([0, 1])),
            (random.choice(PRIMITIVES), random.choice([0, 1, 2])), (random.choice(PRIMITIVES), random.choice([0, 1, 2])), (random.choice(PRIMITIVES), random.choice([0, 1, 2]))], normal_concat=range(0, 4)
)

FORIMAGENET = Genotype(
              normal=[('strippool', 0), ('avg_pool_3x3', 0),
                    ('avg_pool_5x5', 1), ('avg_pool_7x7', 0),
                    ('strippool', 2), ('noise', 1)], normal_concat=range(0, 4))

class RFBFeature(nn.Module):
    def __init__(self):
        super(RFBFeature, self).__init__()
        layers = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),

            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=1),

            # nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
        ]
        self.stem = nn.Sequential(*layers)
        # self.Norm = SPP(256,256)
        # self.Norm = ASPP(256, 256)
        # self.Norm = DCN(256,256)
        # self.Norm = InceptionC(256, 256)
        # self.Norm = StripPool(256, 256)
        # self.Norm = ReceptiveFieldAttention(256, genotype=RANDOMGENOTYPE)
        # self.Norm = SE(256)
        self.Norm = CBAM(256)
        # self.Norm = None
        # self.Norm = BasicRFB(256, 256, scale = 1.0, visual=2)
        # self.Norm = nn.Conv2d(256, 256, 1, 1, 0)

    def forward(self, x):
        # print(x.shape)
        x = self.stem(x)
        # print("Before Norm: ", x.shape)
        if self.Norm:
            return self.Norm(x)
        return x


class DeformNet(nn.Module):
    def __init__(self):
        super(DeformNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.offsets = nn.Conv2d(128, 18, kernel_size=3, padding=1)
        self.conv4 = DeformConv2D(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

    def forward(self, x):
        self.feature_maps = []
        # convs
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)

        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        # x = self.pool2(x)

        # x = F.relu(self.conv3(x))
        # x = self.bn3(x)
        # x = self.pool3(x)

        # deformable convolution
        offsets = self.offsets(x)
        x = self.conv4(x, offsets)

        return x


NAME = "TEST"

net = RFBFeature()  # build the net
# net = DeformNet()


def weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


def weight_init_random(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.normal_(m.state_dict()[key])
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0


# net.Norm.apply(weights_init) #initial
net.apply(weight_init_random)
input_shape = [32, 32, 3]

imgt = Image.open(r"./cat.jpg", mode="r")
imgt = imgt.resize((input_shape[0], input_shape[1]), Image.ANTIALIAS)
trans = transforms.Compose([transforms.ToTensor(),
                            transforms.Resize(32),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

x = trans(imgt)

x = x.unsqueeze(0)

x = torch.randn(1, 3, 32, 32)
x = Variable(x, requires_grad=True)  # input
out = net(x)  # output

Zero_grad = torch.Tensor(1, 256, 16, 16).zero_()  # Zero_grad

Zero_grad[0][128][8][8] = 1  # set the middle pixel to 1.0

out.backward(Zero_grad)  # backward

z = x.grad.data.cpu().numpy()  # get input graident

z = np.sum(np.abs(z), axis=1)

# z = np.mean(z,axis=1) #calculate mean by channels
# # z = np.array(z).mean(0).
# z /= z.max()
# z += (np.abs(z) > 0) * 0.2


# convert to 0-255
z = z * 255 / np.max(z)
z = np.uint8(z)
z = z[0, :, :]

print(z.shape)

# img = Image.fromarray(z) #convert to image
plt.imshow(z)
plt.savefig(f"{NAME}_{random.randint(0,100)}.png", dpi=200)
