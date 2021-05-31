import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.tensorboard.writer import SummaryWriter

SuperNetSetting = [
    [4, 8, 12, 16],  # 1
    [4, 8, 12, 16],  # 2
    [4, 8, 12, 16],  # 3
    [4, 8, 12, 16],  # 4
    [4, 8, 12, 16],  # 5
    [4, 8, 12, 16],  # 6
    [4, 8, 12, 16],  # 7
    [4, 8, 12, 16, 20, 24, 28, 32],  # 8
    [4, 8, 12, 16, 20, 24, 28, 32],  # 9
    [4, 8, 12, 16, 20, 24, 28, 32],  # 10
    [4, 8, 12, 16, 20, 24, 28, 32],  # 11
    [4, 8, 12, 16, 20, 24, 28, 32],  # 12
    [4, 8, 12, 16, 20, 24, 28, 32],  # 13
    [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],  # 14
    [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],  # 15
    [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],  # 16
    [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],  # 17
    [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],  # 18
    [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],  # 19
    [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64],  # fc
]

TrackRunningStats = False


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


class SampleConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.layer_id = layer_id
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(
            out_planes, affine=affine, track_running_stats=TrackRunningStats)

        self.register_buffer('masks', torch.zeros(
            [len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())  # 4, 16, 1, 1

        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i][:channel] = 1

    def forward(self, x, weight, lenth=None):
        '''
        weight 设置一个 one hot [0 1 0 0] 相当于
        '''
        out = self.bn(self.conv(x))
        mixed_masks = 0

        if lenth is None:  # supernet forward
            for w, mask in zip(weight, self.masks):
                # weight存储的信息是, 某一层信息 alpha [0,1,0,0]
                # mask存储的信息也是一层的，每层包括4/8/16个mask
                mixed_masks += w * mask
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            mixed_masks += self.masks[index]

        out = out.cuda()
        return out * mixed_masks


# model = SampleConvBN(0, 8, 16, 1, 1, 1, 1)
# input = torch.zeros(64, 8, 32, 32)
# output = model(input, [1, 0, 0, 0], 8)
# print(output.shape)


class MaskedConv2dBN(nn.Module):
    def __init__(self, layer_id, max_in_channels, max_out_channels, kernel_size=1, stride=1, affine=True):
        super(MaskedConv2dBN, self).__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.layer_id = layer_id

        self.conv = nn.Conv2d(
            self.max_in_channels,
            self.max_out_channels,
            self.kernel_size,
            stride=self.stride,
            bias=False,
            padding=get_same_padding(self.kernel_size)
        )
        self.bn = nn.BatchNorm2d(
            max_out_channels, affine=affine, track_running_stats=False)

        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]),
                                                   SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i][:channel] = 1

        self.active_out_channel = self.max_out_channels

    def forward(self, x, out_channel=None):
        if out_channel is None:
            out_channel = self.active_out_channel

        mixed_masks = 0

        # set mask based on out_channel
        index = SuperNetSetting[self.layer_id].index(out_channel)
        mixed_masks += self.masks[index]
        
        # forward
        out = self.bn(self.conv(x))

        return out * mixed_masks


# model = MaskedConv2dBN(8, 8)
# input = torch.zeros(64, 8, 32, 32)
# output = model(input, 8)
# print(output.shape)
