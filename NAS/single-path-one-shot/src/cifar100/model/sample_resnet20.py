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

__all__ = ['sample_resnet20']

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

LenList = [16, 16, 16, 16, 16, 16, 16, 32, 32,
           32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]
MaskRepeat = 1  # used in RandomMixChannelConvBN and SampleRandomConvBN
ProbRatio = 1.  # used in 'sample_flops_uniform' and 'sample_flops_fair'
R = 1  # used in SampleLocalFreeConvBN
MultiConvBNNum = 2  # used in SampleMultiConvBN
# used in 'sample_trackarch'
TrackFile = "Track1_Submit//files//Track1_final_archs.json"
# used in SampleLocalSepMask SampleLocalSepAdd
LocalSepPortion = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
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
    # work with diff dim tensors, not just 2D ConvNets
    shape = (x.shape[0], ) + (1, ) * (x.ndim - 1)
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


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
        return out * mixed_masks


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, len_list, stride=1, affine=False, convbn_type=SampleConvBN, drop_path_rate=0.):
        super(BasicBlock, self).__init__()
        self.len_list = len_list
        self.drop_path_rate = drop_path_rate
        self.stride = stride

        global IND, TrackRunningStats, SameShortCut

        self.same_shortcut = SameShortCut

        self.convbn1 = convbn_type(IND, self.len_list[IND - 1], self.len_list[IND],
                                   kernel_size=3, stride=stride, padding=1, bias=False, affine=affine)
        self.convbn2 = convbn_type(
            IND + 1, self.len_list[IND], self.len_list[IND + 1], kernel_size=3, stride=1, padding=1, bias=False, affine=affine)

        self.shortcut = nn.Sequential()

        if SameShortCut:
            self.shortcut = convbn_type(
                IND + 1, self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride, padding=0, bias=False, affine=affine)
        else:
            if stride == 2:
                self.shortcut = nn.Sequential(nn.Conv2d(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride,
                                                        padding=0, bias=False), nn.BatchNorm2d(self.len_list[IND + 1], track_running_stats=TrackRunningStats))
            elif stride == 1 and (self.len_list[IND - 1] != self.len_list[IND + 1]):
                self.shortcut = nn.Sequential(nn.Conv2d(self.len_list[IND - 1], self.len_list[IND + 1], kernel_size=1, stride=stride,
                                                        padding=0, bias=False), nn.BatchNorm2d(self.len_list[IND + 1], track_running_stats=TrackRunningStats))
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
    def __init__(self, block, num_blocks, affine=False, convbn_type=SampleConvBN, alpha_type='mix', drop_path_rate=0., dropout=0., num_classes=100):
        super(ResNet, self).__init__()
        self.len_list = LenList
        self.affine = affine
        self.convbn_type = convbn_type
        self.alpha_type = alpha_type
        self.drop_path_rate = drop_path_rate
        self.dropout = dropout

        global IND
        IND = 0

        self.convbn1 = convbn_type(
            IND, 3, self.len_list[IND], kernel_size=3, stride=1, padding=1, bias=False)
        IND += 1
        self.layer1 = self._make_layer(
            self.len_list, block, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(
            self.len_list, block, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(
            self.len_list, block, num_blocks[2], stride=2)

        self.linear = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.len_list[-2], num_classes))
        # self.linear = nn.Linear(self.len_list[-2], num_classes)

        self.apply(_weights_init)

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
                alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(
                    np.random.randint(0, 4, size=7)).view(-1, 1), value=1).cuda())
                alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(
                    np.random.randint(0, 8, size=6)).view(-1, 1), value=1).cuda())
                alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(
                    np.random.randint(0, 16, size=6)).view(-1, 1), value=1).cuda())
        elif 'sample_fair' == self.alpha_type:
            with torch.no_grad():
                pos1 = torch.argmin(
                    self.counts1 + 0.01 * torch.randn(self.counts1.size()).cuda(), dim=1, keepdim=True)
                pos2 = torch.argmin(
                    self.counts2 + 0.01 * torch.randn(self.counts2.size()).cuda(), dim=1, keepdim=True)
                pos3 = torch.argmin(
                    self.counts3 + 0.01 * torch.randn(self.counts3.size()).cuda(), dim=1, keepdim=True)
                alpha1 = Variable(torch.zeros(7, 4).cuda().scatter_(
                    dim=1, index=pos1, value=1))
                alpha2 = Variable(torch.zeros(6, 8).cuda().scatter_(
                    dim=1, index=pos2, value=1))
                alpha3 = Variable(torch.zeros(6, 16).cuda().scatter_(
                    dim=1, index=pos3, value=1))
                self.counts1.add_(alpha1)
                self.counts2.add_(alpha2)
                self.counts3.add_(alpha3)
        elif 'sample_flops_uniform' == self.alpha_type:
            with torch.no_grad():
                alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(
                    np.random.choice(list(range(4)), size=7, p=self.prob1.ravel())).view(-1, 1), value=1).cuda())
                alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(
                    np.random.choice(list(range(8)), size=6, p=self.prob2.ravel())).view(-1, 1), value=1).cuda())
                alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(
                    np.random.choice(list(range(16)), size=6, p=self.prob3.ravel())).view(-1, 1), value=1).cuda())
        elif 'sample_flops_fair' == self.alpha_type:
            with torch.no_grad():
                pos1 = torch.argmin(
                    self.counts1 + 0.01 * torch.randn(self.counts1.size()).cuda(), dim=1, keepdim=True)
                pos2 = torch.argmin(
                    self.counts2 + 0.01 * torch.randn(self.counts2.size()).cuda(), dim=1, keepdim=True)
                pos3 = torch.argmin(
                    self.counts3 + 0.01 * torch.randn(self.counts3.size()).cuda(), dim=1, keepdim=True)
                alpha1 = Variable(torch.zeros(7, 4).cuda().scatter_(
                    dim=1, index=pos1, value=1))
                alpha2 = Variable(torch.zeros(6, 8).cuda().scatter_(
                    dim=1, index=pos2, value=1))
                alpha3 = Variable(torch.zeros(6, 16).cuda().scatter_(
                    dim=1, index=pos3, value=1))
                self.counts1.add_(alpha1 * torch.Tensor(self.delta1).cuda())
                self.counts2.add_(alpha2 * torch.Tensor(self.delta2).cuda())
                self.counts3.add_(alpha3 * torch.Tensor(self.delta3).cuda())
        elif 'sample_sandwich' == self.alpha_type:
            with torch.no_grad():
                assert self.alpha_sandwich_type in ['min', 'max', 'random']
                if self.alpha_sandwich_type == 'random':
                    alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(
                        np.random.randint(0, 4, size=7)).view(-1, 1), value=1).cuda())
                    alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(
                        np.random.randint(0, 8, size=6)).view(-1, 1), value=1).cuda())
                    alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(
                        np.random.randint(0, 16, size=6)).view(-1, 1), value=1).cuda())
                elif self.alpha_sandwich_type == 'min':
                    alpha1 = Variable(torch.zeros(7, 4).scatter_(
                        dim=1, index=torch.LongTensor(np.array(7 * [0])).view(-1, 1), value=1).cuda())
                    alpha2 = Variable(torch.zeros(6, 8).scatter_(
                        dim=1, index=torch.LongTensor(np.array(6 * [0])).view(-1, 1), value=1).cuda())
                    alpha3 = Variable(torch.zeros(6, 16).scatter_(
                        dim=1, index=torch.LongTensor(np.array(6 * [0])).view(-1, 1), value=1).cuda())
                elif self.alpha_sandwich_type == 'max':
                    alpha1 = Variable(torch.zeros(7, 4).scatter_(
                        dim=1, index=torch.LongTensor(np.array(7 * [3])).view(-1, 1), value=1).cuda())
                    alpha2 = Variable(torch.zeros(6, 8).scatter_(
                        dim=1, index=torch.LongTensor(np.array(6 * [7])).view(-1, 1), value=1).cuda())
                    alpha3 = Variable(torch.zeros(6, 16).scatter_(
                        dim=1, index=torch.LongTensor(np.array(6 * [15])).view(-1, 1), value=1).cuda())
        elif 'sample_trackarch' == self.alpha_type:
            with torch.no_grad():
                alpha1 = Variable(torch.zeros(7, 4).scatter_(dim=1, index=torch.LongTensor(
                    self.trackarchs[self.trackindex]['arch'][0:7]).view(-1, 1), value=1).cuda())
                alpha2 = Variable(torch.zeros(6, 8).scatter_(dim=1, index=torch.LongTensor(
                    self.trackarchs[self.trackindex]['arch'][7:13]).view(-1, 1), value=1).cuda())
                alpha3 = Variable(torch.zeros(6, 16).scatter_(dim=1, index=torch.LongTensor(
                    self.trackarchs[self.trackindex]['arch'][13:19]).view(-1, 1), value=1).cuda())
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
            layers.append(block(len_list, stride, self.affine,
                                self.convbn_type, self.drop_path_rate))

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
            out = layer(out, alpha1[2 * i], alpha1[1 + 2 * i], alpha1[1 +
                                                                      2 * i + 1], lenth_list[k - 1], lenth_list[k], lenth_list[k + 1])
            k += 2

        for i, layer in enumerate(self.layer2):
            out = layer(out, alpha2[2 * i] if i > 0 else alpha1[-1], alpha2[2 * i],
                        alpha2[2 * i + 1], lenth_list[k - 1], lenth_list[k], lenth_list[k + 1])
            k += 2

        for i, layer in enumerate(self.layer3):
            out = layer(out, alpha3[2 * i] if i > 0 else alpha2[-1], alpha3[2 * i],
                        alpha3[2 * i + 1], lenth_list[k - 1], lenth_list[k], lenth_list[k + 1])
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
            raise Exception(
                "Localsep_layers format is NOT true: {}".format(localsep_layers))


def sample_resnet20(
    affine=True,
    convbn_type='sample_channel',
    mask_repeat=1,
    alpha_type='sample_uniform',
    prob_ratio=1.,
    r=1,
    localsep_layers=None,
    localsep_portion=1,
    track_file='files/benchmark.json',
    drop_path_rate=0.,
    dropout=0.,
    same_shortcut=True,
    track_running_stats=False,
):
    convbn_dict = {
        'sample_channel': SampleConvBN,
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
        'sample_uniform': ['sample_channel', 'sample_random_channel', 'sample_sepmask_channel', 'sample_sepproject_channel', 'sample_localfree_channel', 'sample_localsepmask_channel', 'sample_localsepadd_channel'],
    }
    assert alpha_type in alpha_dict and convbn_type in alpha_dict[alpha_type]
    return ResNet(BasicBlock, [3, 3, 3], affine=affine, convbn_type=convbn_dict[convbn_type],
                  alpha_type=alpha_type, drop_path_rate=drop_path_rate, dropout=dropout)


def test():
    inputs = torch.rand(2, 3, 32, 32).cuda()
    net = sample_resnet20().cuda()
    print(net)
    print(net(inputs).size())


if __name__ == "__main__":
    test()
