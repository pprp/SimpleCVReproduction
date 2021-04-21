'''
Decode model from alpha
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from utils.compute_flops import *
from nets.se_module import SELayer
import argparse
from pdb import set_trace as br

__all__ = ['resnet_decode']


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class DShortCut(nn.Module):
    def __init__(self, cin, cout, has_avg, has_BN, affine=True):
        super(DShortCut, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False)
        if has_avg:
          self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
          self.avg = None

        if has_BN:
          self.bn = nn.BatchNorm2d(cout, affine=affine)
        else:
          self.bn = None

    def forward(self, x):
        if self.avg:
          out = self.avg(x)
        else:
          out = x

        out = self.conv(out)
        if self.bn:
          out = self.bn(out)
        return out


def ChannelWiseInterV2(inputs, oC, downsample=False):
    assert inputs.dim() == 4, 'invalid dimension : {:}'.format(inputs.size())
    batch, C, H, W = inputs.size()

    if downsample:
      inputs = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)(inputs)
      H, W = inputs.shape[2], inputs.shape[3]
    if C == oC:
      return inputs
    else:
      return nn.functional.adaptive_avg_pool3d(inputs, (oC,H,W))


class BasicBlock(nn.Module):
    # expansion = 1

    def __init__(self, in_planes, cfg, stride=1, affine=True, \
                 se=False, se_reduction=-1):
        """ Args:
          in_planes: an int, the input chanels;
          cfg: a list of int, the mid and output channels;
          se: whether use SE module or not
          se_reduction: the mid width for se module
        """
        super(BasicBlock, self).__init__()
        assert len(cfg) == 2, 'wrong cfg length'
        mid_planes, planes = cfg[0], cfg[1]
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes, affine=affine)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(mid_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine)
        self.relu2 = nn.ReLU()

        self.se = se
        self.se_reduction = se_reduction
        if self.se:
            assert se_reduction > 0, "Must specify se reduction > 0"
            self.se_module = SELayer(planes, se_reduction)

        if stride == 2:
            self.shortcut = DShortCut(in_planes, planes, has_avg=True, has_BN=False)
        elif in_planes != planes:
            self.shortcut = DShortCut(in_planes, planes, has_avg=False, has_BN=True)
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.se:
          out = self.se_module(out)

        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNetDecode(nn.Module):
    """
    Re-build the resnet based on the cfg obtained from alpha.
    """
    def __init__(self, block, cfg, num_classes=10, affine=True, se=False, se_reduction=-1):
        super(ResNetDecode, self).__init__()
        # self.in_planes = 16
        self.cfg = cfg
        self.affine = affine
        self.se = se
        self.se_reduction = se_reduction

        self.conv1 = nn.Conv2d(3, self.cfg[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.cfg[0], affine=self.affine)
        self.relu1 = nn.ReLU()
        num_blocks = [(len(self.cfg) - 1) // 2 // 3] * 3   # [3] * 3

        count = 1
        self.layer1 = self._make_layer(block, self.cfg[count-1:count+num_blocks[0]*2-1], self.cfg[count:count+num_blocks[0]*2], stride=1)
        count += num_blocks[0]*2
        self.layer2 = self._make_layer(block, self.cfg[count-1:count+num_blocks[1]*2-1], self.cfg[count:count+num_blocks[1]*2], stride=2)
        count += num_blocks[1]*2
        self.layer3 = self._make_layer(block, self.cfg[count-1:count+num_blocks[2]*2-1], self.cfg[count:count+num_blocks[2]*2], stride=2)
        count += num_blocks[2]*2
        assert count == len(self.cfg)
        self.linear = nn.Linear(self.cfg[-1], num_classes)
        self.apply(_weights_init)

    def _make_layer(self, block, planes, cfg, stride):
        num_block = len(cfg) // 2
        strides = [stride] + [1]*(num_block-1)
        layers = nn.ModuleList()
        count = 0
        for idx, stride in enumerate(strides):
            layers.append(block(planes[count], cfg[count:count+2], stride, affine=self.affine, se=self.se, se_reduction=self.se_reduction))
            count += 2
        assert count == len(cfg), 'cfg and block num mismatch'
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        avgpool = nn.AvgPool2d(out.size()[3])
        out = avgpool(out)

        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def resnet_decode(cfg, num_classes, se=False, se_reduction=-1):
  return ResNetDecode(BasicBlock, cfg, num_classes=num_classes, se=se, se_reduction=se_reduction)
