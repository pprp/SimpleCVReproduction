import os
import pdb
import sys
import warnings
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .operations import CBAM
from .spaces import OPS

Genotype = namedtuple("Genotype", "normal normal_concat")


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class SE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    # ch_in, ch_out, kernel, stride, padding, groups
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p),
                              groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            # suppress torch 1.9.0 max_pool2d() warning
            warnings.simplefilter('ignore')
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def conv1x1(in_channels, out_channels, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride, bias=False
    )


def conv3x3(in_channels, out_channels, stride=1, padding=1, dilation=1):
    """3x3 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


def conv7x7(in_channels, out_channels, stride=1, padding=3, dilation=1):
    """7x7 convolution"""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=stride,
        padding=padding,
        dilation=dilation,
        bias=False,
    )


class ReceptiveFieldAttention(nn.Module):
    # 2752 params
    '''
        receptive field attention module 
        choose:
            se: True or False 
            conv3x3: use 3x3 or 1x3 conv to fuse feature after rf module 
    '''

    def __init__(self, C, steps=3, reduction=4, se=True, genotype=None):
        super(ReceptiveFieldAttention, self).__init__()
        assert genotype is not None
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se
        self.C_in = C
        self.conv3x3 = False
        self.reduction = reduction

        self.genotype = genotype
        op_names, indices = zip(*self.genotype.normal)
        concat = genotype.normal_concat

        self.bottle = nn.Conv2d(C, C // self.reduction, kernel_size=1,
                                stride=1, padding=0, bias=False)

        self.conv1x1 = nn.Conv2d(
            C // self.reduction * self._steps, C, kernel_size=1, stride=1, padding=0, bias=False)

        if self._se:
            self.se = SE(self.C_in, reduction=reduction)

        if self.conv3x3:
            self.conv3x3 = nn.Conv2d(
                C // self.reduction * self._steps, C, kernel_size=3, stride=1, padding=1, bias=False)

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C // self.reduction, 1, True)
            self._ops += [op]

        self.indices = indices

    def forward(self, x):
        t = self.bottle(x)

        states = [t]
        offset = 0

        total_step = (1+self._steps) * self._steps // 2

        for i in range(total_step):
            h = states[self.indices[i]]
            ops = self._ops[i]
            s = ops(h)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)

        if self.conv3x3:
            node_out = self.conv3x3(node_out)
        else:
            node_out = self.conv1x1(node_out)

        # shortcut
        node_out = node_out + x

        if self._se:
            node_out = self.se(node_out)

        return node_out


class ReceptiveFieldSelfAttention(nn.Module):
    '''
        params: 7504 
        receptive field self attention module (add FFN module)
        choose:
            se: True or False 
            conv3x3: use 3x3 or 1x3 conv to fuse feature after rf module 

    '''

    def __init__(self, C, steps=3, reduction=4, se=True, genotype=None, drop_prob=0., mlp_ratio=8):
        super(ReceptiveFieldSelfAttention, self).__init__()
        assert genotype is not None
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se
        self.C_in = C
        self.conv3x3 = False
        self.reduction = reduction 
        self.norm1 = nn.BatchNorm2d(C)

        self.genotype = genotype
        op_names, indices = zip(*self.genotype.normal)
        concat = genotype.normal_concat

        self.bottle = nn.Conv2d(C, C // self.reduction, kernel_size=1,
                                stride=1, padding=0, bias=False)

        self.conv1x1 = nn.Conv2d(
            C // self.reduction * self._steps, C, kernel_size=1, stride=1, padding=0, bias=False)
        self.drop_path = DropPath(
            drop_prob) if drop_prob > 0. else nn.Identity()
        self.norm2 = nn.BatchNorm2d(C)

        mlp_hidden_dim = int(C // mlp_ratio)

        self.mlp = CMlp(
            in_features=C, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)

        if self._se:
            self.se = SE(self.C_in, reduction=16)

        if self.conv3x3:
            self.conv3x3 = nn.Conv2d(
                C // self.reduction * self._steps, C, kernel_size=3, stride=1, padding=1, bias=False)

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](C // self.reduction, 1, True)
            self._ops += [op]

        self.indices = indices

    def forward(self, x):
        t = self.bottle(x)

        states = [t]
        offset = 0

        total_step = (1+self._steps) * self._steps // 2

        for i in range(total_step):
            h = states[self.indices[i]]
            ops = self._ops[i]
            # print(total_step, h.shape)
            s = ops(h)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)

        if self.conv3x3:
            node_out = self.conv3x3(node_out)
        else:
            node_out = self.conv1x1(node_out)

        # shortcut
        node_out = node_out + x

        node_out = self.norm1(node_out)

        if self._se:
            node_out = self.se(node_out)

        # mlp part
        node_out = node_out + self.drop_path(self.mlp(self.norm2(node_out)))

        return node_out


class RFConvNeXtAttention(nn.Module):
    '''
        receptive field self attention module ConvNext Style (add FFN module)
        choose:
            se: True or False 
            conv3x3: use 3x3 or 1x3 conv to fuse feature after rf module 

    '''

    def __init__(self, C, steps=3, reduction=8, se=False, genotype=None, drop_prob=0., mlp_ratio=4):
        super(RFConvNeXtAttention, self).__init__()
        assert genotype is not None
        self._ops = nn.ModuleList()
        self._C = C
        self._steps = steps
        self._stride = 1
        self._se = se
        self.C_in = C
        self.conv3x3 = False
        self.reduction = reduction
        # self.norm1 = nn.BatchNorm2d(C)
        self.norm1 = nn.LayerNorm(C, eps=1e-6)

        self.genotype = genotype
        op_names, indices = zip(*self.genotype.normal)
        concat = genotype.normal_concat

        self.bottle = nn.Conv2d(C, C // self.reduction, kernel_size=1,
                                stride=1, padding=0, bias=False)

        self.drop_path = DropPath(
            drop_prob) if drop_prob > 0. else nn.Identity()
        
        self.norm2 = nn.BatchNorm2d(C)

        mlp_hidden_dim = int(C // mlp_ratio)

        # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Linear(C, mlp_hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_hidden_dim, C)

        if self._se:
            self.se = SE(self.C_in, reduction=self.reduction)

        if self.conv3x3:
            self.conv3x3 = nn.Conv2d(
                C // self.reduction * self._steps, C, kernel_size=3, stride=1, padding=1, bias=False)
        else:
            self.conv1x1 = nn.Conv2d(
                C // self.reduction * self._steps, C, kernel_size=1, stride=1, padding=0, bias=False)

        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](self.C_in // self.reduction, 1, True)
            self._ops += [op]

        self.indices = indices

    def forward(self, x):
        t = self.bottle(x)

        states = [t]

        total_step = (1+self._steps) * self._steps // 2

        for i in range(total_step):
            h = states[self.indices[i]]
            ops = self._ops[i]
            # print(total_step, h.shape)
            s = ops(h)
            states.append(s)

        # concate all released nodes
        node_out = torch.cat(states[-self._steps:], dim=1)

        if self.conv3x3:
            node_out = self.conv3x3(node_out)
        else:
            node_out = self.conv1x1(node_out)

        node_out = node_out.permute(0, 2, 3, 1)  # N C H W -> N H W C

        # shortcut
        # node_out = node_out + x

        node_out = self.norm1(node_out)
        node_out = self.pwconv1(node_out)
        node_out = self.act(node_out)
        node_out = self.pwconv2(node_out)

        node_out = node_out.permute(0, 3, 1, 2)

        if self._se:
            node_out = self.se(node_out)

        # mlp part
        node_out = node_out + self.drop_path(node_out)

        return node_out


class CifarRFBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step, genotype=None):
        super(CifarRFBasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.genotype = genotype
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = ReceptiveFieldAttention(
            planes, genotype=self.genotype)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = out + residual
        out = self.relu(out)

        return out


class Attention(nn.Module):
    def __init__(self, step, C, genotype):
        super(Attention, self).__init__()
        self._steps = step
        self._C = C
        self._ops = nn.ModuleList()
        self.C_in = self._C // 4
        self.C_out = self._C
        self.width = 4
        self.se = SE(self.C_in, reduction=2)  # 8
        self.se2 = SE(self.C_in * 4, reduction=2)  # 8
        self.channel_back = nn.Sequential(
            nn.Conv2d(
                self.C_in * 5, self._C, kernel_size=1, padding=0, groups=1, bias=False
            ),
            nn.BatchNorm2d(self._C),
            nn.ReLU(inplace=False),
            nn.Conv2d(self._C, self._C, kernel_size=1,
                      padding=0, groups=1, bias=False),
            nn.BatchNorm2d(self._C),
        )
        self.genotype = genotype
        op_names, indices = zip(*genotype.normal)
        concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat)

    def _compile(self, C, op_names, indices, concat):
        assert len(op_names) == len(indices)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            op = OPS[name](self.C_in, 1, True)
            self._ops += [op]
        self.indices = indices

    def forward(self, x):
        states = [x]
        C_num = x.shape[1]
        length = C_num // 4
        spx = torch.split(x, length, 1)
        spx_sum = sum(spx)
        spx_sum = self.se(spx_sum)
        states[0] = spx[0]
        h01 = states[self.indices[0]]
        op01 = self._ops[0]
        h01_out = op01(h01)
        s = h01_out
        states += [s]

        states[0] = spx[1]
        h02 = states[self.indices[1]]
        h12 = states[self.indices[2]]
        op02 = self._ops[1]
        op12 = self._ops[2]
        h02_out = op02(h02)
        h12_out = op12(h12)
        s = h02_out + h12_out
        states += [s]

        states[0] = spx[2]
        h03 = states[self.indices[3]]
        h13 = states[self.indices[4]]
        h23 = states[self.indices[5]]
        op03 = self._ops[3]
        op13 = self._ops[4]
        op23 = self._ops[5]
        h03_out = op03(h03)
        h13_out = op13(h13)
        h23_out = op23(h23)
        s = h03_out + h13_out + h23_out
        states += [s]

        states[0] = spx[3]
        h04 = states[self.indices[6]]
        h14 = states[self.indices[7]]
        h24 = states[self.indices[8]]
        h34 = states[self.indices[9]]

        op04 = self._ops[6]
        op14 = self._ops[7]
        op24 = self._ops[8]
        op34 = self._ops[9]

        h04_out = op04(h04)
        h14_out = op14(h14)
        h24_out = op24(h24)
        h34_out = op34(h34)
        s = h04_out + h14_out + h24_out + h34_out
        states += [s]

        node_concat = torch.cat(states[-4:], dim=1)
        node_concat = torch.cat((node_concat, spx_sum), dim=1)
        attention_out = self.channel_back(node_concat) + x
        attention_out = self.se2(attention_out)
        return attention_out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, step=0, genotype=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarRFConvNeXtBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step=3, genotype=None):
        super(CifarRFConvNeXtBasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = RFConvNeXtAttention(planes, genotype=genotype)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = out + residual
        out = self.relu(out)

        return out


class CifarRFSABasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step=3, genotype=None):
        super(CifarRFSABasicBlock, self).__init__()
        self._steps = step
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride

        self.attention = ReceptiveFieldSelfAttention(planes, genotype=genotype)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = out + residual
        out = self.relu(out)

        return out


class NormalAttentionBasicBlock(nn.Module):
    '''
    SE 
    CBAM 
    SPP 
    '''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, step=0, genotype=None, se=False, cbam=False, spp=False):
        super(NormalAttentionBasicBlock, self).__init__()

        self.se = se
        self.cbam = cbam
        self.spp = spp

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

        if self.se:
            self.se = SE(planes, reduction=16)

        if self.cbam:
            self.cbam = CBAM(planes, reduction_ratio=16)

        if self.spp:
            self.spp = SPP(planes, planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.se:
            out = self.se(out)

        if self.cbam:
            out = self.cbam(out)

        if self.spp:
            out = self.spp(out)

        out += self.shortcut(x)
        out = F.relu(out)
        return out


class CifarAttentionBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride, step, genotype):
        super(CifarAttentionBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.genotype = genotype
        if inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.downsample = lambda x: x
        self.stride = stride
        self._step = step
        # print(f"line 158: planes: {planes}")
        self.attention = Attention(self._step, planes, self.genotype)

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.attention(out)
        out = residual + out
        out = self.relu(out)
        return out


