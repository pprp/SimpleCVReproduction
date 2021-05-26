import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import random
import json
import os
from itertools import combinations

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20']

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

LenList = [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]
MaskRepeat = 1  # used in RandomMixChannelConvBN and SampleRandomConvBN
ProbRatio = 1.  # used in 'sample_flops_uniform' and 'sample_flops_fair'
R = 1  # used in SampleLocalFreeConvBN
MultiConvBNNum = 2  # used in SampleMultiConvBN
TrackFile = "Track1_final_archs.json"  # used in 'sample_trackarch'
LocalSepPortion = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # used in SampleLocalSepMask SampleLocalSepAdd
TrackRunningStats = False  # used in BN
SameShortCut = False  # used in BasicBlock


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class MixChannelConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(MixChannelConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.layer_id = layer_id
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats)

        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i][:channel] = 1

    def forward(self, x, weight, lenth=None):
        out = self.bn(self.conv(x))
        mixed_masks = 0
        if lenth is None:  # supernet forward
            for w, mask in zip(weight, self.masks):
                mixed_masks += w * mask
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            for i, (w, mask) in enumerate(zip(weight, self.masks)):
                if i != index:
                    mixed_masks += w * mask
        return out * mixed_masks


class RandomMaskMixChannelConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(RandomMaskMixChannelConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats)
        global MaskRepeat
        self.mask_repeat = MaskRepeat
        self.mask_lenth = out_planes

        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i] = self.generate_random_mask(channel).cuda()

    def generate_random_mask(self, one_counts):
        import random

        mask = torch.zeros([self.mask_lenth, 1, 1])
        for i in range(self.mask_repeat):
            indice = list(range(self.mask_lenth))
            random.shuffle(indice)
            indice = indice[:one_counts]
            for j in indice:
                mask[j] += 1
        mask /= self.mask_repeat
        return mask

    def forward(self, x, weight, lenth=None):
        out = self.bn(self.conv(x))
        mixed_masks = 0
        if lenth is None:  # supernet forward
            for w, mask in zip(weight, self.masks):
                mixed_masks += w * mask
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            for i, (w, mask) in enumerate(zip(weight, self.masks)):
                if i != index:
                    mixed_masks += w * mask
        return out * mixed_masks


class SampleConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.layer_id = layer_id
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats)

        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i][:channel] = 1

    def forward(self, x, weight, lenth=None):
        out = self.bn(self.conv(x))
        mixed_masks = 0
        if lenth is None:  # supernet forward
            for w, mask in zip(weight, self.masks):
                mixed_masks += w * mask
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            mixed_masks += self.masks[index]
        return out * mixed_masks


class SampleRandomConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleRandomConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats)
        global MaskRepeat
        self.mask_repeat = MaskRepeat
        self.mask_lenth = out_planes

        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i] = self.generate_random_mask(channel).cuda()

    def generate_random_mask(self, one_counts):
        import random

        mask = torch.zeros([self.mask_lenth, 1, 1])
        for i in range(self.mask_repeat):
            indice = list(range(self.mask_lenth))
            random.shuffle(indice)
            indice = indice[:one_counts]
            for j in indice:
                mask[j] += 1
        mask /= self.mask_repeat
        return mask

    def forward(self, x, weight, lenth=None):
        out = self.bn(self.conv(x))
        mixed_masks = 0
        if lenth is None:  # supernet forward
            for w, mask in zip(weight, self.masks):
                mixed_masks += w * mask
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            mixed_masks += self.masks[index]
        return out * mixed_masks


class SampleLocalSepAddConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleLocalSepAddConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global LocalSepPortion
        global TrackRunningStats

        self.layer_id = layer_id
        self.out_planes = out_planes
        self.sep_num = int(1 / LocalSepPortion[layer_id])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(self.sep_num):
            self.convs.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.bns.append(nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats))
        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i][:channel] = 1

    def forward(self, x, weight, lenth=None):
        out = 0
        if lenth is None:  # supernet forward
            for i, w in enumerate(weight):
                if w != 0:
                    for conv, bn in zip(self.convs, self.bns):
                        tmp = bn(conv(x))
                        tmp = self.masks[i] * tmp
                        out += w * tmp
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            for conv, bn in zip(self.convs, self.bns):
                tmp = bn(conv(x))
                tmp = self.masks[index] * tmp
                out += tmp
        return out


class SampleLocalSepMaskConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleLocalSepMaskConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global LocalSepPortion
        global TrackRunningStats

        self.layer_id = layer_id
        self.out_planes = out_planes
        self.sep_num = int(1 / LocalSepPortion[layer_id])
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        for _ in range(self.sep_num):
            self.convs.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.bns.append(nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats))
        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i][:channel] = 1

    def get_index(self, i):
        total_lenth = len(SuperNetSetting[self.layer_id])
        local_lenth = total_lenth // self.sep_num
        return i // local_lenth

    def forward(self, x, weight, lenth=None):
        out = 0
        if lenth is None:  # supernet forward
            for i, w in enumerate(weight):
                if w != 0:
                    tmp = self.bns[self.get_index(i)](self.convs[self.get_index(i)](x))
                    tmp = self.masks[i] * tmp
                    out += w * tmp
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            tmp = self.bns[self.get_index(index)](self.convs[self.get_index(index)](x))
            tmp = self.masks[index] * tmp
            out += tmp
        return out


class SampleSepMaskConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleSepMaskConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.layer_id = layer_id

        self.out_planes = out_planes
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for channel in SuperNetSetting[layer_id]:
            self.convs.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.bns.append(nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats))
        self.register_buffer('masks', torch.zeros([len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for i, channel in enumerate(SuperNetSetting[layer_id]):
            self.masks[i][:channel] = 1

    def forward(self, x, weight, lenth=None):
        out = 0
        if lenth is None:  # supernet forward
            for i, w in enumerate(weight):
                if w != 0:
                    tmp = self.bns[i](self.convs[i](x))
                    tmp = self.masks[i] * tmp
                    out += w * tmp
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            tmp = self.bns[index](self.convs[index](x))
            tmp = self.masks[index] * tmp
            out += tmp
        return out


class SampleSepProjectConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleSepProjectConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.layer_id = layer_id

        self.out_planes = out_planes
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.projects = nn.ModuleList()
        for channel in SuperNetSetting[layer_id]:
            self.convs.append(nn.Conv2d(in_planes, channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
            self.bns.append(nn.BatchNorm2d(channel, affine=affine, track_running_stats=TrackRunningStats))
            self.projects.append(nn.Sequential(nn.Conv2d(channel, out_planes, kernel_size=1, stride=1, padding=0, bias=False), nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats)))

    def forward(self, x, weight, lenth=None):
        out = 0
        if lenth is None:  # supernet forward
            for i, w in enumerate(weight):
                if w != 0:
                    tmp = self.projects[i](self.bns[i](self.convs[i](x)))
                    out += w * tmp
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            tmp = self.projects[index](self.bns[index](self.convs[index](x)))
            out += tmp
        return out


class SampleLocalFreeConvBN(nn.Module):
    def __init__(self, layer_id, in_planes, out_planes, kernel_size, stride, padding, bias, affine=False):
        super(SampleLocalFreeConvBN, self).__init__()
        assert out_planes == SuperNetSetting[layer_id][-1]
        global TrackRunningStats
        self.layer_id = layer_id
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, affine=affine, track_running_stats=TrackRunningStats)

        global R
        num_masks = len(list(combinations(list(range(R * 2 + 1)), R + 1)))
        len_bin = SuperNetSetting[layer_id][0]
        self.register_buffer('masks', torch.zeros([num_masks, len(SuperNetSetting[layer_id]), SuperNetSetting[layer_id][-1], 1, 1]).cuda())
        for k, comb in enumerate(combinations(list(range(R * 2 + 1)), R + 1)):
            for i, channel in enumerate(SuperNetSetting[layer_id]):
                max_C = int(SuperNetSetting[layer_id][-1] / len_bin)
                C = int(channel / len_bin)
                base_end = C - R - 1
                free = []
                for part in comb:
                    f = C - R + part
                    if f > max_C:
                        j = max_C
                        while j in free and j > 1:
                            j -= 1
                        f = j
                    free.append(f)
                if len(free) > C:
                    free = free[-C:]
                if base_end > 0:
                    self.masks[k][i][:base_end * len_bin] = 1
                for f in free:
                    if f > 0 and f <= max_C:
                        self.masks[k][i][(f - 1) * len_bin:f * len_bin] = 1
                assert self.masks[k][i].sum() == channel
        self.mask_index = 0

    def forward(self, x, weight, lenth=None):
        out = self.bn(self.conv(x))
        mixed_masks = 0
        if lenth is None:  # supernet forward
            for w, mask in zip(weight, self.masks[self.mask_index]):
                mixed_masks += w * mask
        else:  # subnet forward
            assert lenth in SuperNetSetting[self.layer_id]
            index = SuperNetSetting[self.layer_id].index(lenth)
            mixed_masks += self.masks[self.mask_index][index]
        return out * mixed_masks

    def set_mask_index(self, i):
        self.mask_index = i


def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, len_list, stride=1, affine=False, convbn_type=MixChannelConvBN, drop_path_rate=0.):
        super(BasicBlock, self).__init__()
        self.len_list = len_list
        self.drop_path_rate = drop_path_rate
        self.stride = stride

        global IND, TrackRunningStats, SameShortCut

        self.same_shortcut = SameShortCut

        self.convbn1 = convbn_type(IND, self.len_list[IND - 1], self.len_list[IND], kernel_size=3, stride=stride, padding=1, bias=False, affine=affine)
        self.convbn2 = convbn_type(IND + 1, self.len_list[IND], self.len_list[IND + 1], kernel_size=3, stride=1, padding=1, bias=False, affine=affine)

        self.shortcut = nn.Sequential()
        if SameShortCut:
            self.shortcut = convbn_type(IND + 1, self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0, bias=False, affine=affine)
        else:
            if stride == 2:
                self.shortcut = nn.Sequential(nn.Conv2d(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0, bias=False), nn.BatchNorm2d(self.len_list[IND + 1], track_running_stats=TrackRunningStats))
            elif stride == 1 and (self.len_list[IND - 1] != self.len_list[IND + 1]):
                self.shortcut = nn.Sequential(nn.Conv2d(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0, bias=False), nn.BatchNorm2d(self.len_list[IND + 1], track_running_stats=TrackRunningStats))
        IND += 2

    def forward(self, x, weight0, weight1, weight2, lenth0=None, lenth1=None, lenth2=None):
        out = F.relu(self.convbn1(x, weight1, lenth1))
        out = self.convbn2(out, weight2, lenth2)
        if self.drop_path_rate > 0. and self.stride != 2:
            x = drop_path(x, self.drop_path_rate, self.training)
        if self.same_shortcut:
            out += self.shortcut(x, weight2, lenth2)
            # if self.stride == 1 and (torch.equal(weight0, weight2) or (lenth0 != None and lenth0 == lenth2)):
            #     out += x
            # else:
            #     out += self.shortcut(x, weight2, lenth2)
        else:
            out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, affine=False, convbn_type=MixChannelConvBN, alpha_type='mix', drop_path_rate=0., dropout=0., num_classes=100):
        super(ResNet, self).__init__()
        self.len_list = LenList
        self.affine = affine
        self.convbn_type = convbn_type
        self.alpha_type = alpha_type
        self.drop_path_rate = drop_path_rate
        self.dropout = dropout

        global IND
        IND = 0

        self.convbn1 = convbn_type(IND, 3, self.len_list[IND], kernel_size=3, stride=1, padding=1, bias=False)
        IND += 1
        self.layer1 = self._make_layer(self.len_list, block, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(self.len_list, block, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(self.len_list, block, num_blocks[2], stride=2)
        self.linear = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(self.len_list[-2], num_classes))
        # self.linear = nn.Linear(self.len_list[-2], num_classes)

        self.apply(_weights_init)

        self.alpha_init()

    def alpha_init(self):
        global ProbRatio
        if self.alpha_type == 'mix':
            self.alpha1 = Variable(1e-3 * torch.randn(7, 4).cuda(), requires_grad=True)
            self.alpha2 = Variable(1e-3 * torch.randn(6, 8).cuda(), requires_grad=True)
            self.alpha3 = Variable(1e-3 * torch.randn(6, 16).cuda(), requires_grad=True)

            self.alpha = [self.alpha1, self.alpha2, self.alpha3]
        elif self.alpha_type == 'sample_fair':
            self.register_buffer('counts1', torch.zeros(7, 4).cuda())
            self.register_buffer('counts2', torch.zeros(6, 8).cuda())
            self.register_buffer('counts3', torch.zeros(6, 16).cuda())
        elif self.alpha_type == 'sample_flops_uniform':
            l1 = ProbRatio * (np.array([4., 8., 12., 16.]) - 4) + 4
            l2 = ProbRatio * (np.array([4., 8., 12., 16., 20., 24., 28., 32.]) - 4) + 4
            l3 = ProbRatio * (np.array([4., 8., 12., 16., 20., 24., 28., 32., 36., 40., 44., 48., 52., 56., 60., 64.]) - 4) + 4
            self.prob1 = l1 / l1.sum()
            self.prob2 = l2 / l2.sum()
            self.prob3 = l3 / l3.sum()
        elif self.alpha_type == 'sample_flops_fair':
            self.delta1 = 4 / (ProbRatio * (np.array(SuperNetSetting[0:7]) - 4) + 4)
            self.delta2 = 4 / (ProbRatio * (np.array(SuperNetSetting[7:13]) - 4) + 4)
            self.delta3 = 4 / (ProbRatio * (np.array(SuperNetSetting[13:19]) - 4) + 4)
            self.register_buffer('counts1', torch.Tensor(7 * [4 * [0]]).cuda())
            self.register_buffer('counts2', torch.Tensor(6 * [8 * [0]]).cuda())
            self.register_buffer('counts3', torch.Tensor(6 * [16 * [0]]).cuda())
        elif self.alpha_type == 'sample_sandwich':
            self.alpha_sandwich_type = 'random'
        elif self.alpha_type == 'sample_trackarch':
            global TrackFile
            assert os.path.exists(TrackFile)
            with open(TrackFile, 'r') as f:
                arch_info = json.load(f)
            self.trackarchs = []
            self.trackindex = 0
            for arch_i in arch_info:
                arch = arch_info[arch_i]['arch'].split('-')
                acc = arch_info[arch_i]['acc']
                trackarch = []
                for c in arch:
                    trackarch.append(int(c) // 4 - 1)
                self.trackarchs.append({'arch': trackarch, 'acc': acc})
            random.shuffle(self.trackarchs)

    def alpha_hold(self):
        if not hasattr(self, 'pre_alphas'):
            self.pre_alphas = []
        self.tohold = True

    def alpha_pop(self):
        self.topop = True

    def alpha_cal(self):
        global ProbRatio
        global SuperNetSetting
        if hasattr(self, 'topop') and self.topop:
            alpha1, alpha2, alpha3 = self.pre_alphas[0]
            del self.pre_alphas[0]
            self.topop = False
            return alpha1, alpha2, alpha3

        if 'mix' == self.alpha_type:
            alpha1 = F.softmax(self.alpha1, dim=-1)
            alpha2 = F.softmax(self.alpha2, dim=-1)
            alpha3 = F.softmax(self.alpha3, dim=-1)
        elif 'sample_uniform' == self.alpha_type:
            with torch.no_grad():
                alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(np.random.randint(0, 4, size=7)).view(-1, 1), value=1).cuda())
                alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(np.random.randint(0, 8, size=6)).view(-1, 1), value=1).cuda())
                alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(np.random.randint(0, 16, size=6)).view(-1, 1), value=1).cuda())
        elif 'sample_fair' == self.alpha_type:
            with torch.no_grad():
                pos1 = torch.argmin(self.counts1 + 0.01 * torch.randn(self.counts1.size()).cuda(), dim=1, keepdim=True)
                pos2 = torch.argmin(self.counts2 + 0.01 * torch.randn(self.counts2.size()).cuda(), dim=1, keepdim=True)
                pos3 = torch.argmin(self.counts3 + 0.01 * torch.randn(self.counts3.size()).cuda(), dim=1, keepdim=True)
                alpha1 = Variable(torch.zeros(7, 4).cuda().scatter_(dim=1, index=pos1, value=1))
                alpha2 = Variable(torch.zeros(6, 8).cuda().scatter_(dim=1, index=pos2, value=1))
                alpha3 = Variable(torch.zeros(6, 16).cuda().scatter_(dim=1, index=pos3, value=1))
                self.counts1.add_(alpha1)
                self.counts2.add_(alpha2)
                self.counts3.add_(alpha3)
        elif 'sample_flops_uniform' == self.alpha_type:
            with torch.no_grad():
                alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(np.random.choice(list(range(4)), size=7, p=self.prob1.ravel())).view(-1, 1), value=1).cuda())
                alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(np.random.choice(list(range(8)), size=6, p=self.prob2.ravel())).view(-1, 1), value=1).cuda())
                alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(np.random.choice(list(range(16)), size=6, p=self.prob3.ravel())).view(-1, 1), value=1).cuda())
        elif 'sample_flops_fair' == self.alpha_type:
            with torch.no_grad():
                pos1 = torch.argmin(self.counts1 + 0.01 * torch.randn(self.counts1.size()).cuda(), dim=1, keepdim=True)
                pos2 = torch.argmin(self.counts2 + 0.01 * torch.randn(self.counts2.size()).cuda(), dim=1, keepdim=True)
                pos3 = torch.argmin(self.counts3 + 0.01 * torch.randn(self.counts3.size()).cuda(), dim=1, keepdim=True)
                alpha1 = Variable(torch.zeros(7, 4).cuda().scatter_(dim=1, index=pos1, value=1))
                alpha2 = Variable(torch.zeros(6, 8).cuda().scatter_(dim=1, index=pos2, value=1))
                alpha3 = Variable(torch.zeros(6, 16).cuda().scatter_(dim=1, index=pos3, value=1))
                self.counts1.add_(alpha1 * torch.Tensor(self.delta1).cuda())
                self.counts2.add_(alpha2 * torch.Tensor(self.delta2).cuda())
                self.counts3.add_(alpha3 * torch.Tensor(self.delta3).cuda())
        elif 'sample_sandwich' == self.alpha_type:
            with torch.no_grad():
                assert self.alpha_sandwich_type in ['min', 'max', 'random']
                if self.alpha_sandwich_type == 'random':
                    alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(np.random.randint(0, 4, size=7)).view(-1, 1), value=1).cuda())
                    alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(np.random.randint(0, 8, size=6)).view(-1, 1), value=1).cuda())
                    alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(np.random.randint(0, 16, size=6)).view(-1, 1), value=1).cuda())
                elif self.alpha_sandwich_type == 'min':
                    alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(np.array(7 * [0])).view(-1, 1), value=1).cuda())
                    alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(np.array(6 * [0])).view(-1, 1), value=1).cuda())
                    alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(np.array(6 * [0])).view(-1, 1), value=1).cuda())
                elif self.alpha_sandwich_type == 'max':
                    alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(np.array(7 * [3])).view(-1, 1), value=1).cuda())
                    alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(np.array(6 * [7])).view(-1, 1), value=1).cuda())
                    alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(np.array(6 * [15])).view(-1, 1), value=1).cuda())
        elif 'sample_trackarch' == self.alpha_type:
            with torch.no_grad():
                alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(self.trackarchs[self.trackindex]['arch'][0:7]).view(-1, 1), value=1).cuda())
                alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(self.trackarchs[self.trackindex]['arch'][7:13]).view(-1, 1), value=1).cuda())
                alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(self.trackarchs[self.trackindex]['arch'][13:19]).view(-1, 1), value=1).cuda())
                self.trackindex += 1
                if self.trackindex == len(self.trackarchs):
                    random.shuffle(self.trackarchs)
                    self.trackindex = 0

        if hasattr(self, 'tohold') and self.tohold:
            self.pre_alphas.append([alpha1, alpha2, alpha3])
            self.tohold = False

        return alpha1, alpha2, alpha3

    def _make_layer(self, len_list, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(len_list, stride, self.affine, self.convbn_type, self.drop_path_rate))

        return nn.Sequential(*layers)

    def forward(self, x, lenth_list=None):
        if lenth_list is None:  # supernet forward
            lenth_list = [None] * len(LenList)
        else:  # subnet forward
            assert len(lenth_list) == len(LenList)
        k = 0
        alpha1, alpha2, alpha3 = self.alpha_cal()
        out = F.relu(self.convbn1(x, alpha1[0], lenth_list[k]))
        k += 1
        for i, layer in enumerate(self.layer1):
            out = layer(out, alpha1[2 * i], alpha1[1 + 2 * i], alpha1[1 + 2 * i + 1], lenth_list[k - 1], lenth_list[k], lenth_list[k + 1])
            k += 2
        for i, layer in enumerate(self.layer2):
            out = layer(out, alpha2[2 * i] if i > 0 else alpha1[-1], alpha2[2 * i], alpha2[2 * i + 1], lenth_list[k - 1], lenth_list[k], lenth_list[k + 1])
            k += 2
        for i, layer in enumerate(self.layer3):
            out = layer(out, alpha3[2 * i] if i > 0 else alpha2[-1], alpha3[2 * i], alpha3[2 * i + 1], lenth_list[k - 1], lenth_list[k], lenth_list[k + 1])
            k += 2
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def min_min(self, x, target, criterion):
        global R
        if self.convbn_type == SampleLocalFreeConvBN:
            with torch.no_grad():
                min_loss = 99999
                min_index = 0
                for k, _ in enumerate(combinations(list(range(R * 2 + 1)), R + 1)):
                    for m in self.modules():
                        if isinstance(m, SampleLocalFreeConvBN):
                            m.set_mask_index(k)
                    loss = criterion(self.forward(x), target)
                    if loss < min_loss:
                        min_index = k
                        min_loss = loss
                for m in self.modules():
                    if isinstance(m, SampleLocalFreeConvBN):
                        m.set_mask_index(min_index)

    def set_drop_path_rate(self, drop_path_rate):
        for m in self.modules():
            if hasattr(m, 'drop_path_rate'):
                m.drop_path_rate = drop_path_rate


def set_localsep_portion(localsep_layers, localsep_portion):
    global LocalSepPortion
    if localsep_layers is None:
        return
    elif localsep_layers == 'all':
        for i in range(len(LocalSepPortion)):
            LocalSepPortion[i] = localsep_portion
        print('LocalSepPortion: ', LocalSepPortion)
    else:
        try:
            layers = localsep_layers.split(',')
            for layer in layers:
                LocalSepPortion[int(layer)] = localsep_portion
            print(LocalSepPortion)
        except:
            raise Exception("Localsep_layers format is NOT true: {}".format(localsep_layers))


def resnet20(
    affine=False,
    convbn_type='mix_channel',
    mask_repeat=1,
    alpha_type='mix',
    prob_ratio=1.,
    r=1,
    localsep_layers=None,
    localsep_portion=1,
    track_file='Track1_final_archs.json',
    drop_path_rate=0.,
    dropout=0.,
    same_shortcut=False,
    track_running_stats=False,
):
    convbn_dict = {
        'mix_channel': MixChannelConvBN,
        'random_mix_channel': RandomMaskMixChannelConvBN,
        'sample_channel': SampleConvBN,
        'sample_random_channel': SampleRandomConvBN,
        'sample_sepmask_channel': SampleSepMaskConvBN,
        'sample_sepproject_channel': SampleSepProjectConvBN,
        'sample_localfree_channel': SampleLocalFreeConvBN,
        'sample_localsepmask_channel': SampleLocalSepMaskConvBN,
        'sample_localsepadd_channel': SampleLocalSepAddConvBN,
    }
    assert convbn_type in convbn_dict
    global MaskRepeat
    MaskRepeat = mask_repeat
    global ProbRatio
    ProbRatio = prob_ratio
    global R
    R = int(r)
    global SameShortCut
    SameShortCut = same_shortcut
    global TrackRunningStats
    TrackRunningStats = track_running_stats
    global TrackFile
    TrackFile = track_file
    if convbn_type == 'sample_localsepmask_channel' or convbn_type == 'sample_localsepadd_channel':
        set_localsep_portion(localsep_layers, localsep_portion)
    alpha_dict = {
        'mix': ['mix_channel', 'random_mix_channel'],
        'sample_uniform': ['sample_channel', 'sample_random_channel', 'sample_sepmask_channel', 'sample_sepproject_channel', 'sample_localfree_channel', 'sample_localsepmask_channel', 'sample_localsepadd_channel'],
        'sample_fair': ['sample_channel', 'sample_random_channel', 'sample_sepmask_channel', 'sample_sepproject_channel', 'sample_localfree_channel', 'sample_localsepmask_channel', 'sample_localsepadd_channel'],
        'sample_flops_uniform': ['sample_channel', 'sample_random_channel', 'sample_sepmask_channel', 'sample_sepproject_channel', 'sample_localfree_channel', 'sample_localsepmask_channel', 'sample_localsepadd_channel'],
        'sample_flops_fair': ['sample_channel', 'sample_random_channel', 'sample_sepmask_channel', 'sample_sepproject_channel', 'sample_localfree_channel', 'sample_localsepmask_channel', 'sample_localsepadd_channel'],
        'sample_sandwich': ['sample_channel', 'sample_random_channel', 'sample_sepmask_channel', 'sample_sepproject_channel', 'sample_localfree_channel', 'sample_localsepmask_channel', 'sample_localsepadd_channel'],
        'sample_trackarch': ['sample_channel', 'sample_random_channel', 'sample_sepmask_channel', 'sample_sepproject_channel', 'sample_localfree_channel', 'sample_localsepmask_channel', 'sample_localsepadd_channel'],
    }
    assert alpha_type in alpha_dict and convbn_type in alpha_dict[alpha_type]
    return ResNet(BasicBlock, [3, 3, 3], affine=affine, convbn_type=convbn_dict[convbn_type], alpha_type=alpha_type, drop_path_rate=drop_path_rate, dropout=dropout)


def test():
    inputs = torch.rand(2, 3, 32, 32).cuda()
    net = resnet20().cuda()
    print(net)
    print(net(inputs).size())


if __name__ == "__main__":
    test()