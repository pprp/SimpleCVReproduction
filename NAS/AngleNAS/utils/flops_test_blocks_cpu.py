import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from torch.nn import init
import math

op_keys = [
    'PreProcessing',
    'mobilenet_3x3_ratio_3',
    'mobilenet_3x3_ratio_6',
    'mobilenet_5x5_ratio_3',
    'mobilenet_5x5_ratio_6',
    'mobilenet_7x7_ratio_3',
    'mobilenet_7x7_ratio_6',
    'PostProcessing'
    ]

blocks_dict = {
    'PreProcessing':lambda inp, oup, stride : PreProcessing(inp, oup, stride),
    'mobilenet_3x3_ratio_3':lambda inp, oup, stride : InvertedResidual(inp, oup, 3, 1, stride, 3),
    'mobilenet_3x3_ratio_6':lambda inp, oup, stride : InvertedResidual(inp, oup, 3, 1, stride, 6),
    'mobilenet_5x5_ratio_3':lambda inp, oup, stride : InvertedResidual(inp, oup, 5, 2, stride, 3),
    'mobilenet_5x5_ratio_6':lambda inp, oup, stride : InvertedResidual(inp, oup, 5, 2, stride, 6),
    'mobilenet_7x7_ratio_3':lambda inp, oup, stride : InvertedResidual(inp, oup, 7, 3, stride, 3),
    'mobilenet_7x7_ratio_6':lambda inp, oup, stride : InvertedResidual(inp, oup, 7, 3, stride, 6),
    'PostProcessing':lambda inp, oup, stride : PostProcessing(inp, oup, stride)
}

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class PreProcessing(nn.Module):
    def __init__(self, inp, oup, stride, width_mult=1.):
        super(PreProcessing, self).__init__()
        self.conv_bn = conv_bn(inp, oup, stride)
        self.MBConv_ratio_1 = InvertedResidual(oup, int(16*width_mult), 3, 1, 1, 1)
    
    def forward(self, x, rngs=None):
        x = self.conv_bn(x)
        output = self.MBConv_ratio_1(x)
        return output

class PostProcessing(nn.Module):
    def __init__(self, inp, oup, stride, input_size=224, n_class=1000):
        super(PostProcessing, self).__init__()
        self.conv_1x1_bn = conv_1x1_bn(inp, oup)
        self.avgpool = nn.AvgPool2d(input_size//32)
        self.oup = oup
        self.classifier = nn.Linear(oup, n_class)
    
    def forward(self, x, rngs=None):
        x = self.conv_1x1_bn(x)
        x = self.avgpool(x)
        x = x.view(-1, self.oup)
        output = self.classifier(x)
        return output

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, ksize, padding, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride

        self.use_res_connect = self.stride == 1 and inp == oup
        self.inp = inp
        self.oup = oup
        self.type='MBConv_{}_{}_{}'.format(expand_ratio, ksize, oup)
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

    def forward(self, x, rngs=None):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)