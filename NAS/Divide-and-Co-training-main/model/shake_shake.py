# -*-coding:utf-8-*-
"""
shake-shake resnet.

Reference:
    [1] https://github.com/BIGBALLON/CIFAR-ZOO

    [2] Xavier Gastaldi. Shake-Shake regularization, 2017, ICLRW.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


__all__ = ['shake_resnet26_2x32d', 'shake_resnet26_2x64d', 'shake_resnet26_2x96d']


class ShakeShake(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x1, x2, training=True):
        if training:
            alpha = torch.cuda.FloatTensor(x1.size(0)).uniform_()
            alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x1)
            # alpha = alpha.reshape(alpha.size(0), 1, 1, 1).expand_as(x1)
        else:
            alpha = 0.5
        return alpha * x1 + (1 - alpha) * x2

    @staticmethod
    def backward(ctx, grad_output):
        beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_()
        beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
        # beta = beta.reshape(beta.size(0), 1, 1, 1).expand_as(grad_output)
        beta = Variable(beta)

        return beta * grad_output, (1 - beta) * grad_output, None


class Shortcut(nn.Module):

    def __init__(self, in_ch, out_ch, stride):
        super(Shortcut, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1,
                               stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1,
                               stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        h = F.relu(x)

        h1 = F.avg_pool2d(h, 1, self.stride)
        h1 = self.conv1(h1)

        h2 = F.avg_pool2d(F.pad(h, (-1, 1, -1, 1)), 1, self.stride)
        h2 = self.conv2(h2)

        h = torch.cat((h1, h2), 1)
        return self.bn(h)


class ShakeBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super(ShakeBlock, self).__init__()
        self.equal_io = in_ch == out_ch
        self.shortcut = self.equal_io and None or Shortcut(
            in_ch, out_ch, stride=stride)

        self.branch1 = self._make_branch(in_ch, out_ch, stride)
        self.branch2 = self._make_branch(in_ch, out_ch, stride)

    def forward(self, x):
        h1 = self.branch1(x)
        h2 = self.branch2(x)
        h = ShakeShake.apply(h1, h2, self.training)
        h0 = x if self.equal_io else self.shortcut(x)
        return h + h0

    def _make_branch(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))


class ShakeResNet(nn.Module):

    def __init__(self, depth, base_width, num_classes=10, dataset='cifar10', split_factor=1):
        super(ShakeResNet, self).__init__()
        n_units = (depth - 2) / 6

        base_width_dict = {96: {1: 96, 2: 64, 4: 48, 8: 32, 16: 24},
                            64: {1: 64, 2: 44, 4: 32, 8: 20},
                            32: {1: 32, 2: 24, 4: 16},
                        }

        base_width = base_width_dict[base_width][split_factor]
        print('INFO:PyTorch: The base width of shake-shake resnet is {}'.format(base_width))

        in_chs = [16, base_width, base_width * 2, base_width * 4]
        self.in_chs = in_chs

        self.conv_1 = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(in_chs[0])

        self.stage_1 = self._make_layer(n_units, in_chs[0], in_chs[1])
        self.stage_2 = self._make_layer(n_units, in_chs[1], in_chs[2], stride=2)
        self.stage_3 = self._make_layer(n_units, in_chs[2], in_chs[3], stride=2)
        self.fc_out = nn.Linear(in_chs[3], num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Initialize paramters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv_1(x)
        out = self.bn_1(out)
        out = self.stage_1(out)
        out = self.stage_2(out)
        out = self.stage_3(out)
        out = F.relu(out)
        # out = F.avg_pool2d(out, 8)
        
        out = self.avgpool(out)
        out = out.view(-1, self.in_chs[3])
        out = self.fc_out(out)

        return out

    def _make_layer(self, n_units, in_ch, out_ch, stride=1):
        layers = []
        for _ in range(int(n_units)):
            layers.append(ShakeBlock(in_ch, out_ch, stride=stride))
            in_ch, stride = out_ch, 1
        return nn.Sequential(*layers)


# Only for CIFAR and SVHN
def shake_resnet26_2x32d(**kwargs):
    return ShakeResNet(depth=26, base_width=32, **kwargs)


def shake_resnet26_2x64d(**kwargs):
    return ShakeResNet(depth=26, base_width=64, **kwargs)


def shake_resnet26_2x96d(**kwargs):
    return ShakeResNet(depth=26, base_width=96, **kwargs)
