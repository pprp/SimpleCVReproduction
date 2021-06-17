import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from prettytable import PrettyTable


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, in_choose_list=None, out_choose_list=None, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list

        # 构建选择的index list
        self.in_choose_list = [
            i+1 for i in range(len(in_features_list))] if in_choose_list is None else in_choose_list
        self.out_choose_list = [
            i+1 for i in range(len(out_features_list))] if out_choose_list is None else out_choose_list

        self.in_choice = max(self.in_choose_list)
        self.out_choice = max(self.out_choose_list)

        # self.width_mult = max(self.width_mult_list)

    def forward(self, input):

        in_idx = self.in_choose_list.index(self.in_choice)
        out_idx = self.out_choose_list.index(self.out_choice)

        self.in_features = self.in_features_list[in_idx]
        self.out_features = self.out_features_list[out_idx]

        weight = self.weight[:self.out_features, :self.in_features]

        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class SwitchableBatchNorm2d(nn.Module):
    # num_features_list: [16, 32, 48, 64]
    # 与out_channels_list相一致
    def __init__(self, num_features_list, out_choose_list=None):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)  # 4
        # self.width_mult_list = width_mult_list

        self.out_choose_list = [
            i+1 for i in range(len(num_features_list))] if out_choose_list is None else out_choose_list

        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))  # 分别有多个bn与其对应

        self.bn = nn.ModuleList(bns)  # 其中包含4个bn

        self.out_choice = max(self.out_choose_list)
    def forward(self, input):
        # idx = self.width_mult_list.index(self.width_mult)
        idx = self.out_choose_list.index(self.out_choice)
        y = self.bn[idx](input)
        return y



class SlimmableConv2d(nn.Conv2d):
    # in_channels_list: [3,3,3,3]
    # out_channels_list: [16, 32, 48, 64]
    # kernel_size: 3
    # stride: 1
    # padding: 3
    # bias=False
    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 kernel_size,
                 in_choose_list=None,
                 out_choose_list=None,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups_list=[1],
                 bias=True):
        super(SlimmableConv2d, self).__init__(max(in_channels_list),
                                              max(out_channels_list),
                                              kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=max(groups_list),
                                              bias=bias)

        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list

        self.in_choose_list = [
            i+1 for i in range(len(in_channels_list))] if in_choose_list is None else in_choose_list
        self.out_choose_list = [
            i+1 for i in range(len(out_channels_list))] if out_choose_list is None else out_choose_list

        # self.width_mult_list = width_mult_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        # self.width_mult = max(self.width_mult_list)

        self.in_choice = max(self.in_choose_list)
        self.out_choice = max(self.out_choose_list)
        # 这里必须选用最大的channel数目作为共享的对象

    def forward(self, input):
        # idx = self.width_mult_list.index(self.width_mult)  # 判定到底选择哪个作为index
        in_idx = self.in_choose_list.index(self.in_choice)
        out_idx = self.out_choose_list.index(self.out_choice)

        self.in_channels = self.in_channels_list[in_idx]
        self.out_channels = self.out_channels_list[out_idx]  # 找到对应的in和out

        self.groups = self.groups_list[in_idx]  # 组卷积

        weight = self.weight[:self.out_channels, :self.in_channels, :, :]

        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding,
                                 self.dilation, self.groups)
        return y
