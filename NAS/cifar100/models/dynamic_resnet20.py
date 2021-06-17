from __future__ import absolute_import, division

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from .modules.dynamic_modules import (DynamicConv2d, DynamicBatchNorm2d,
                                      DynamicLinear)

__all__ = ['dynamic_resnet20']


def get_configs():
    model_config = {}

    for i in range(0, 7):
        model_config[i] = [4*(i+1) for i in range(4)]
    for i in range(7, 13):
        model_config[i] = [4*(i+1) for i in range(8)]
    for i in range(13, 20):
        model_config[i] = [4*(i+1) for i in range(16)]

    return model_config


class DynamicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel_list, mid_channel_list, out_channel_list, stride=1):
        super(DynamicBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', DynamicConv2d(max(in_channel_list), max(
                    mid_channel_list), kernel_size=3, stride=stride)),
                ('bn', DynamicBatchNorm2d(max(mid_channel_list))),
                ('relu', nn.ReLU(inplace=True))
            ]))

        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv', DynamicConv2d(max(mid_channel_list), max(
                    out_channel_list), kernel_size=3, stride=1)),
                ('bn', DynamicBatchNorm2d(max(out_channel_list)))
            ]))

        self.downsample = nn.Sequential(
            OrderedDict([
                ('conv', DynamicConv2d(max(in_channel_list), max(
                    out_channel_list), stride=stride)),
                ('bn', DynamicBatchNorm2d(max(out_channel_list)))
            ]))

        self.active_mid_channel = max(mid_channel_list)
        self.active_out_channel = max(out_channel_list)

    def forward(self, x, amc=None, aoc=None):
        # amc: conv1 output channel
        # aoc: conv2 output channel
        if amc is None:
            amc = self.active_mid_channel
        if aoc is None:
            aoc = self.active_out_channel

        self.conv1.conv.active_out_channel = amc
        self.conv2.conv.active_out_channel = aoc

        self.downsample.conv.active_out_channel = aoc

        # print("amc:", amc, "aoc:", aoc)

        out = self.conv1(x)
        out = self.conv2(out)

        # print("shape: ", "x:", x.shape,  "out:", out.shape,
        #       "shortcut:", self.downsample(x).shape)

        out += self.downsample(x)
        out = F.relu(out)
        return out

    def re_organize_middle_weight(self):
        # conv2 -> conv1
        importance = torch.sum(
            torch.abs(self.conv2.conv.conv.weight.data), dim=(0, 2, 3))
        sorted_importance, sorted_idx = torch.sort(
            importance, dim=0, descending=True)
        self.conv2.conv.conv.weight.data = torch.index_select(
            self.conv2.conv.conv.weight.data, 1, sorted_idx)
        self.adjust_bn_according_to_idx(self.conv1.bn.bn, sorted_idx)
        self.conv1.conv.conv.weight.data = torch.index_select(
            self.conv1.conv.conv.weight.data, 0, sorted_idx)
        # print("re org done")
        return None

    def adjust_bn_according_to_idx(self, bn, idx):
        bn.weight.data = torch.index_select(bn.weight.data, 0, idx)
        bn.bias.data = torch.index_select(bn.bias.data, 0, idx)
        if type(bn) in [nn.BatchNorm1d, nn.BatchNorm2d]:
            bn.running_mean.data = torch.index_select(
                bn.running_mean.data, 0, idx)
            bn.running_var.data = torch.index_select(
                bn.running_var.data, 0, idx)
        return None


class DynamicLayer(nn.Module):
    def __init__(self, block, num_blocks, stride):
        super(DynamicLayer, self).__init__()
        self.mc = get_configs()

        global IDX
        strides = [stride] + [1]*(num_blocks-1)

        self.layer1 = block(
            self.mc[IDX-1], self.mc[IDX], self.mc[IDX+1], stride=strides[0])
        IDX += 2
        self.layer2 = block(
            self.mc[IDX-1], self.mc[IDX], self.mc[IDX+1], stride=strides[1])
        IDX += 2
        self.layer3 = block(
            self.mc[IDX-1], self.mc[IDX], self.mc[IDX+1], stride=strides[2])
        IDX += 2

    def forward(self, x, config=None):
        if config is None:
            # layer config
            config = [16, 16, 16, 16, 16, 16, 16]

        # print("block1")
        out = self.layer1(x, amc=config[1], aoc=config[2])
        # print("block2")
        out = self.layer2(out, amc=config[3], aoc=config[4])
        # print("block3")
        out = self.layer3(out, amc=config[5], aoc=config[6])
        return out


class DynamicResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(DynamicResNet, self).__init__()

        global IDX
        IDX = 0

        self.mc = get_configs()

        self.first_conv = nn.Sequential(OrderedDict([
            ('conv', DynamicConv2d(3, max(self.mc[IDX]), kernel_size=3)),
            ('bn', DynamicBatchNorm2d(max(self.mc[IDX]))),
            ('relu', nn.ReLU(inplace=True))
        ]))

        IDX += 1

        self.block1 = DynamicLayer(block, num_blocks[0], stride=1)
        self.block2 = DynamicLayer(block, num_blocks[1], stride=2)
        self.block3 = DynamicLayer(block, num_blocks[2], stride=2)

        self.classifier = DynamicLinear(max(self.mc[IDX]), num_classes)
        self._initialize_weights()

    def forward(self, x, config=None):
        # print("config:", config)
        if config is None:
            config = [16, 16, 16, 16, 16, 16, 16, 32, 32,
                      32, 32, 32, 32, 64, 64, 64, 64, 64, 64]

        cfg_first = config[0]
        cfg_layer1 = config[0:7]
        cfg_layer2 = config[6:13]
        cfg_layer3 = config[12:19]

        # print("first conv")
        out = self.first_conv.conv(x, cfg_first)
        out = self.first_conv.bn(out)
        out = self.first_conv.relu(out)
        # print("layer1")
        out = self.block1(out, cfg_layer1)
        # print("layer2")
        out = self.block2(out, cfg_layer2)
        # print("layer3")
        out = self.block3(out, cfg_layer3)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        # print("end")
        return self.classifier(out)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # print("init conv2d")
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # print("init bn")
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                # nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                # print("init linear")
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        # for m in self.modules():
        #     if isinstance(m, DynamicBlock):
        #         # type: ignore[arg-type]
        #         nn.init.constant_(m.conv2.bn.bn.weight, 0)


def dynamic_resnet20():
    return DynamicResNet(DynamicBlock, [3, 3, 3])


# arc_config = [4, 12, 4, 4, 16, 8, 4, 12, 32,
#               24, 16, 8, 8, 24, 60, 12, 64, 64, 52, 60]
# arc_config = [i+1 for i in range(20)]
# model = dynamic_resnet20()
# print(model)
# a = torch.randn(3, 3, 32, 32)
# print(model(a, arc_config[:-1]).shape)
# m1 = dynamic_resnet20()
# print(m1)
# print(m1.first_conv.conv.conv.weight.shape)
