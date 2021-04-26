import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from collections import OrderedDict

# from model.slimmable_modules import SlimmableConv2d, SlimmableLinear, SwitchableBatchNorm2d
from independent_modules import SlimmableConv2d, SwitchableBatchNorm2d, SwitchableLinear

# arc_representation = "4-12-4-4-16-8-4-12-32-24-16-8-8-24-60-12-64-64-52-60"
arc_representation = [4, 12, 4, 4, 16, 8, 4, 12, 32,
                      24, 16, 8, 8, 24, 60, 12, 64, 64, 52, 60]
# "16-8-16-16-8-12-12-20-12-4-12-32-32-24-48-8-52-16-12-36"
max_arc_rep = "16-16-16-16-16-16-16-32-32-32-32-32-32-64-64-64-64-64-64-64"


def get_configs():
    model_config = {}

    for i in range(0, 7):
        model_config[i] = [4*(i+1) for i in range(4)]
    for i in range(7, 13):
        model_config[i] = [4*(i+1) for i in range(8)]
    for i in range(13, 20):
        model_config[i] = [4*(i+1) for i in range(16)]

    return model_config


class IndependentBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel_list, mid_channel_list, out_channel_list, stride=1):
        super(IndependentBlock, self).__init__()

        self.conv1 = nn.Sequential(
            OrderedDict([
                ('conv', SlimmableConv2d(in_channel_list,
                                         mid_channel_list,
                                         kernel_size=3,
                                         groups_list=[
                                             2 for _ in out_channel_list],
                                         stride=stride)),
                ('bn', SwitchableBatchNorm2d(mid_channel_list)),
                ('relu', nn.ReLU(inplace=True))
            ]))

        self.conv2 = nn.Sequential(
            OrderedDict([
                ('conv', SlimmableConv2d(mid_channel_list,
                                         out_channel_list,
                                         kernel_size=3,
                                         groups_list=[
                                             2 for _ in out_channel_list],
                                         stride=1)),
                ('bn', SwitchableBatchNorm2d(out_channel_list))
            ]))

        self.downsample = nn.Sequential(
            OrderedDict([
                ('conv', SlimmableConv2d(in_channel_list,
                                         out_channel_list,
                                         stride=stride,
                                         kernel_size=1)),
                ('bn', SwitchableBatchNorm2d(out_channel_list))
            ])
        )

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

        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.conv2(out)

        out += residual
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


class IndependentLayer(nn.Module):
    def __init__(self, block, num_blocks, stride):
        super(IndependentLayer, self).__init__()
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

        print(config)

        # print("block1")
        out = self.layer1(x, amc=config[1], aoc=config[2])
        # print("block2")
        out = self.layer2(out, amc=config[3], aoc=config[4])
        # print("block3")
        out = self.layer3(out, amc=config[5], aoc=config[6])
        return out


class IndependentResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(IndependentResNet, self).__init__()

        global IDX
        IDX = 0

        self.mc = get_configs()

        self.first_conv = nn.Sequential(OrderedDict([
            ('conv', SlimmableConv2d([3],
                                     self.mc[IDX],
                                     groups_list=[1],
                                     kernel_size=3)),
            ('bn', SwitchableBatchNorm2d(self.mc[IDX])),
            ('relu', nn.ReLU(inplace=True))
        ]))

        IDX += 1

        self.block1 = IndependentLayer(block, num_blocks[0], stride=1)
        self.block2 = IndependentLayer(block, num_blocks[1], stride=2)
        self.block3 = IndependentLayer(block, num_blocks[2], stride=2)

        self.classifier = SwitchableLinear(self.mc[IDX], num_classes)
        self._initialize_weights()

    def forward(self, x, config=None):
        print("!!", config)
        if config is None:
            config = [16, 16, 16, 16, 16, 16, 16, 32, 32,
                      32, 32, 32, 32, 64, 64, 64, 64, 64, 64]

        cfg_first = config[0]
        cfg_layer1 = config[0:7]
        cfg_layer2 = config[6:13]
        cfg_layer3 = config[12:19]

        print("first conv")
        out = self.first_conv.conv(x, cfg_first)
        out = self.first_conv.bn(out)
        out = self.first_conv.relu(out)
        print("layer1")
        out = self.block1(out, cfg_layer1)
        print("layer2")
        out = self.block2(out, cfg_layer2)
        print("layer3")
        out = self.block3(out, cfg_layer3)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        return self.classifier(out)

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def Independent_resnet20():
    return IndependentResNet(IndependentBlock, [3, 3, 3])


if __name__ == "__main__":
    model = Independent_resnet20()

    input = torch.zeros(16, 3, 32, 32)

    output = model(input, [8, 8, 8, 8, 8, 8, 8, 8, 8,
                           8, 8, 8, 8, 8, 8, 8, 8, 8, 8])
