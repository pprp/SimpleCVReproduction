import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from prettytable import PrettyTable


def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(
        kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class SwitchableLinear(nn.Module):
    def __init__(self, in_choose_list=None, bias=True, num_classes=100):
        super(SwitchableLinear, self).__init__()
        self.in_choose_list = in_choose_list  # [4,8,12,16]

        lnlst = []
        for i in range(len(self.in_choose_list)):
            lnlst.append(
                nn.Linear(self.in_choose_list[i], num_classes, bias=bias))

        self.lnlst = nn.ModuleList(lnlst)

    def forward(self, input, channel=None):
        if channel is None:
            channel = input.size(1)
        idx = self.in_choose_list.index(channel)

        return self.lnlst[idx](input)


class SwitchableBatchNorm2d(nn.Module):
    def __init__(self, in_choose_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.in_choose_list = in_choose_list

        bns = []
        for i in range(len(self.in_choose_list)):
            bns.append(nn.BatchNorm2d(self.in_choose_list[i]))  # 分别有多个bn与其对应
        self.bn = nn.ModuleList(bns)  # 其中包含4个bn

    def forward(self, input, width=None):
        if width is None:
            width = input.size(1)

        idx = self.in_choose_list.index(width)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    def __init__(self,
                 in_choose_list,
                 out_choose_list,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups_list=[1],
                 bias=True):
        super(SlimmableConv2d, self).__init__(max(in_choose_list),
                                              max(out_choose_list),
                                              kernel_size,
                                              stride=stride,
                                              padding=get_same_padding(
                                                  kernel_size),
                                              dilation=dilation,
                                              groups=max(groups_list),
                                              bias=bias)

        print("INITRIAL: ", max(groups_list), "weight:", self.weight.shape,
              "in channel:", max(in_choose_list), "out channel:", max(out_choose_list))

        self.padding = get_same_padding(kernel_size)
        self.in_choose_list = in_choose_list
        self.out_choose_list = out_choose_list

        self.groups_list = groups_list

        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_choose_list))]

        self.active_out_channel = max(out_choose_list)

    def forward(self, input, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel

        in_channel = input.size(1)

        in_idx = self.in_choose_list.index(in_channel)

        self.groups = self.groups_list[in_idx]  # 组卷积

        weight = self.weight[:out_channel, :in_channel, :, :]

        if self.bias is not None:
            bias = self.bias[:out_channel]
        else:
            bias = self.bias

        print("weight:", weight.shape,  "groups:", self.groups)

        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding,
                                 self.dilation, self.groups)
        return y


# m1 = SlimmableConv2d([4, 8, 12], [4, 4, 8], 3, groups_list=[4])
# # m2 = SwitchableLinear([4, 8, 34])
# # m3 = SwitchableBatchNorm2d([4, 4, 8])

# # a = torch.randn(4,4,8,8)

# # b = torch.randn(4, 4)

# # print(m1(a,4,8).shape)
# # print(m2(b,4).shape)
# # print(m3(a,4).shape)

# print(m1.weight.shape)
