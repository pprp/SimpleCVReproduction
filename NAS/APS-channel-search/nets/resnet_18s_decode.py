import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from nets.se_module import SELayer
import os
from pdb import set_trace as br

__all__ = ['ResNet18s_Decode', 'resnet12_decode', 'resnet14_decode', 'resnet18_decode', 'resnet50_decode']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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


class BasicBlock(nn.Module):

    def __init__(self, inplanes, cfg, stride=1, se=False, se_reduction=-1):
        super(BasicBlock, self).__init__()
        assert len(cfg) == 2, 'wrong cfg length!'
        mid_planes, planes = cfg[0], cfg[1]

        self.conv1 = conv3x3(inplanes, mid_planes, stride)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(mid_planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True) # added by me
        self.stride = stride

        if stride == 2:
            self.shortcut = DShortCut(inplanes, planes, has_avg=True, has_BN=False)
        elif inplanes != planes:
            self.shortcut = DShortCut(inplanes, planes, has_avg=False, has_BN=True)
        else:
            self.shortcut = nn.Sequential()

        self.se = se
        self.se_reduction = se_reduction
        if self.se:
            assert se_reduction > 0, "Must specify se reduction > 0"
            self.se_module = SELayer(planes, se_reduction)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.se:
            out = self.se_module(out)

        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, inplanes, cfg, stride=1, se=False, se_reduction=-1):
        # NOTE: while no expansion=4 here, make sure it is multiplied in cfgs
        super(Bottleneck, self).__init__()
        assert len(cfg) == 3, 'wrong cfg length'
        assert cfg[0] == cfg[1], 'dw channels are not equal'
        mid_planes, planes = cfg[0], cfg[-1]

        self.conv1 = conv1x1(inplanes, mid_planes)
        self.bn1 = nn.BatchNorm2d(mid_planes)

        self.conv2 = conv3x3(mid_planes, mid_planes, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)

        self.conv3 = conv1x1(mid_planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

        if stride == 2:
            self.shortcut = DShortCut(inplanes, planes, has_avg=True, has_BN=False)
        elif inplanes != planes:
            self.shortcut = DShortCut(inplanes, planes, has_avg=False, has_BN=True)
        else:
            self.shortcut = nn.Sequential()

        self.se = se
        self.se_reduction = se_reduction
        if self.se:
            assert se_reduction > 0, "Must specify se reduction > 0"
            self.se_module = SELayer(planes, se_reduction)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.se:
            out = self.se_module(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18s_Decode(nn.Module):
    def __init__(self, block, num_blocks, cfgs, num_classes=1000, se=False, se_reduction=-1, zero_init_residual=False):
        super(ResNet18s_Decode, self).__init__()
        self.cfgs = cfgs
        self.num_blocks = num_blocks # [3,4,6,3]
        self.block_layer_num = 2 if block == BasicBlock else 3
        assert len(self.cfgs) == self.block_layer_num*sum(self.num_blocks) + 1, 'cfg length and num_blocks do not match'
        self.se = se
        self.se_reduction = se_reduction

        self.conv1 = nn.Conv2d(3, cfgs[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(cfgs[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        count = 1
        self.layer1 = self._make_layer(block, self.cfgs[count-1 : count+num_blocks[0]*self.block_layer_num-1], \
            self.cfgs[count : count+num_blocks[0]*self.block_layer_num], stride=1)
        count += num_blocks[0]*self.block_layer_num

        self.layer2 = self._make_layer(block, self.cfgs[count-1 : count+num_blocks[1]*self.block_layer_num-1], \
            self.cfgs[count : count+num_blocks[1]*self.block_layer_num], stride=2)
        count += num_blocks[1]*self.block_layer_num

        self.layer3 = self._make_layer(block, self.cfgs[count-1 : count+num_blocks[2]*self.block_layer_num-1], \
            self.cfgs[count : count+num_blocks[2]*self.block_layer_num], stride=2)
        count += num_blocks[2]*self.block_layer_num

        self.layer4 = self._make_layer(block, self.cfgs[count-1 : count+num_blocks[3]*self.block_layer_num-1], \
            self.cfgs[count : count+num_blocks[3]*self.block_layer_num], stride=2)
        count += num_blocks[3]*self.block_layer_num
        assert count == len(cfgs)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.cfgs[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, inplanes, cfgs, stride):
        assert len(cfgs) % self.block_layer_num == 0, 'cfgs must be dividable by block_layer_num'
        print('Out channels:', cfgs)
        num_block = len(cfgs) // self.block_layer_num
        strides = [stride] + [1]*(num_block-1)
        layers = nn.ModuleList()
        count = 0
        for idx, stride in enumerate(strides):
            layers.append(block(inplanes[count], cfgs[count:count+self.block_layer_num], stride, \
                se=self.se, se_reduction=self.se_reduction))
            count += self.block_layer_num
        assert count == len(cfgs)
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet12_decode(cfg, num_classes, se=False, se_reduction=-1):
    """Constructs a ResNet-18 cfg configured model.
    """
    return ResNet18s_Decode(BasicBlock, [1,1,1,2], cfg, num_classes=num_classes, se=se, se_reduction=se_reduction)


def resnet14_decode(cfg, num_classes, se=False, se_reduction=-1):
    """Constructs a ResNet-18 cfg configured model.
    """
    return ResNet18s_Decode(BasicBlock, [1,1,2,2], cfg, num_classes=num_classes, se=se, se_reduction=se_reduction)


def resnet18_decode(cfg, num_classes, se=False, se_reduction=-1):
    """Constructs a ResNet-18 cfg configured model.
    """
    return ResNet18s_Decode(BasicBlock, [2,2,2,2], cfg, num_classes=num_classes, se=se, se_reduction=se_reduction)


def resnet50_decode(cfg, num_classes, se=False, se_reduction=-1):
    """Constructs a ResNet-18 cfg configured model.
    """
    return ResNet18s_Decode(Bottleneck, [3,4,6,3], cfg, num_classes=num_classes, se=se, se_reduction=se_reduction)

