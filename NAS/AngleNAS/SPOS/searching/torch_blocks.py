import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math
from config import config

blocks_keys = config.blocks_keys

blocks_dict = {
    'mobilenet_3x3_ratio_3':lambda inp, oup, stride : InvertedResidual(inp, oup, 3, 1, stride, 3),
    'mobilenet_3x3_ratio_6':lambda inp, oup, stride : InvertedResidual(inp, oup, 3, 1, stride, 6),
    'mobilenet_5x5_ratio_3':lambda inp, oup, stride : InvertedResidual(inp, oup, 5, 2, stride, 3),
    'mobilenet_5x5_ratio_6':lambda inp, oup, stride : InvertedResidual(inp, oup, 5, 2, stride, 6),
    'mobilenet_7x7_ratio_3':lambda inp, oup, stride : InvertedResidual(inp, oup, 7, 3, stride, 3),
    'mobilenet_7x7_ratio_6':lambda inp, oup, stride : InvertedResidual(inp, oup, 7, 3, stride, 6),
}

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ksize, padding, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and inp == oup
        self.expand_ratio = expand_ratio

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, ksize, stride, padding, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(inp * expand_ratio, inp * expand_ratio, ksize, stride, padding, groups=inp * expand_ratio, bias=False),
                nn.BatchNorm2d(inp * expand_ratio),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)



