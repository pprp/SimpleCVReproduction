# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn


class StdConv(nn.Module):
    # 普通的卷积+ bn + relu
    def __init__(self, C_in, C_out):
        super(StdConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(C_in, C_out, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class PoolBranch(nn.Module):
    # pool branch max or avg
    # 设计：conv + pool + bn
    def __init__(self, pool_type, C_in, C_out, kernel_size, stride, padding, affine=False):
        super().__init__()
        self.preproc = StdConv(C_in, C_out)
        self.pool = Pool(pool_type, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        out = self.preproc(x)
        out = self.pool(out)
        out = self.bn(out)
        return out


class SeparableConv(nn.Module):
    # 可分离卷积
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(SeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(C_in, C_in, kernel_size=kernel_size, padding=padding, stride=stride,
                                   groups=C_in, bias=False)
        self.pointwise = nn.Conv2d(C_in, C_out, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class ConvBranch(nn.Module):
    # 卷积分支
    #   - conv 整合channel
    #   - conv 进行卷积操作
    #   - bn+relu
    def __init__(self, C_in, C_out, kernel_size, stride, padding, separable):
        super(ConvBranch, self).__init__()
        self.preproc = StdConv(C_in, C_out)
        if separable:
            self.conv = SeparableConv(
                C_out, C_out, kernel_size, stride, padding)
        else:
            self.conv = nn.Conv2d(C_out, C_out, kernel_size,
                                  stride=stride, padding=padding)
        self.postproc = nn.Sequential(
            nn.BatchNorm2d(C_out, affine=False),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.preproc(x)
        out = self.conv(out)
        out = self.postproc(out)
        return out


class SEConvBranch(nn.Module):
    # FIRST ATTEMPT
    # https://github.com/moskomule/senet.pytorch/blob/master/senet/se_resnet.py
    # Conv + BN + SE
    def __init__(self, c_in, c_out, kernel_size, stride, padding, reduction=16):
        super(SELayer, self).__init__()
        self.preproc = StdConv(c_in, c_out)
        self.conv = nn.Conv2d(
            c_out, c_out, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(c_out, affine=False)

        # SE Part
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.preproc(x)
        x = self.conv(x)
        x = self.bn(x)

        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.avgpool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class FactorizedReduce(nn.Module):
    # 条纹状降维
    def __init__(self, C_in, C_out, affine=False):
        super().__init__()
        self.conv1 = nn.Conv2d(C_in, C_out // 2, 1,
                               stride=2, padding=0, bias=False)
        self.conv2 = nn.Conv2d(C_in, C_out // 2, 1,
                               stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        out = torch.cat([self.conv1(x), self.conv2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class Pool(nn.Module):
    # pool 提供max or avg
    def __init__(self, pool_type, kernel_size, stride, padding):
        super().__init__()
        if pool_type.lower() == 'max':
            self.pool = nn.MaxPool2d(kernel_size, stride, padding)
        elif pool_type.lower() == 'avg':
            self.pool = nn.AvgPool2d(
                kernel_size, stride, padding, count_include_pad=False)
        else:
            raise ValueError()

    def forward(self, x):
        return self.pool(x)


class SepConvBN(nn.Module):
    # relu + 可分离卷积 + bn
    def __init__(self, C_in, C_out, kernel_size, padding):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = SeparableConv(C_in, C_out, kernel_size, 1, padding)
        self.bn = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


if __name__ == "__main__":
    ts = torch.zeros(8, 64, 16, 16)
    model = FactorizedReduce(64, 32)
    print(model.forward(ts).shape)

    # ts2 = torch.zeros(8, 64, 17, 17)
    # model = FactorizedReduce(64, 32)
    # print(model.forward(ts2).shape)
