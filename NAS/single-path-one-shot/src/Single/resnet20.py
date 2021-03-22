''' from https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32',
           'resnet44', 'resnet56', 'resnet110', 'resnet1202']

model_list = "12-8-6-4-16-16-12-28-28-28-32-4-24-48-20-16-12-64-4-4"
# model_list = "3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3-3"


def make_net():
    mutable_arch = list_to_array(model_list)


def list_to_array(model_list):
    model_array = model_list.split('-')
    model_array = [int(model_array[i]) for i in range(len(model_array))]
    return model_array


def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    # expansion = 1

    def __init__(self, in_planes, medium_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, medium_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(medium_channels)

        self.conv2 = nn.Conv2d(medium_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        option = 'B'

        if stride != 1 or out_channels != in_planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x: F.pad(
                    x[:, :, ::2, ::2], (0, 0, 0, 0, out_channels-in_planes, out_channels-in_planes), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, out_channels,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    # num_blocks [3,3,3]
    def __init__(self, block, mutable_arch, num_classes=100):
        super(ResNet, self).__init__()

        self.mutable_arch = list_to_array(mutable_arch)
        self.idx = 0
        self.conv1 = nn.Conv2d(3, self.mutable_arch[self.idx], kernel_size=3,
                               stride=1, padding=1, bias=False)  # 01

        self.bn1 = nn.BatchNorm2d(self.mutable_arch[self.idx])
        self.in_channels = self.mutable_arch[self.idx]
        self.layer1 = [self._make_layer(
            block, self.mutable_arch[self.idx+1], self.mutable_arch[self.idx+2], stride=1) for _ in range(3)]
        self.layer2 = [
            self._make_layer(block, self.mutable_arch[self.idx + 1], self.mutable_arch[self.idx + 2], stride=2) for _ in
            range(3)]
        self.layer3 = [
            self._make_layer(block, self.mutable_arch[self.idx + 1], self.mutable_arch[self.idx + 2], stride=2) for _ in
            range(3)]
        self.convsum = nn.Sequential(*self.layer1, *self.layer2, *self.layer3)
        self.linear1 = nn.Linear(
            self.mutable_arch[self.idx], self.mutable_arch[self.idx+1])
        self.linear2 = nn.Linear(self.mutable_arch[self.idx+1], num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, medium_channels, out_channels, stride):
        # strides = [stride] + [1]*(num_blocks-1)
        # layers = []
        # layers.append(block(self.in_planes, planes, stride))
        layer = block(self.in_channels, medium_channels, out_channels, stride)
        # self.in_planes = planes * block.expansion
        self.in_channels = out_channels
        self.idx += 2
        # print("test ", self.idx, self.mutable_arch[self.idx])

        return layer

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        # print(out.shape)
        out = self.convsum(out)
        # print(out.shape)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        # print(out.shape)

        return out


def resnet20(mutable_arch):
    return ResNet(BasicBlock, mutable_arch)


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(
        lambda p: p.requires_grad and len(p.data.size()) > 1, net.parameters()))))


if __name__ == "__main__":
    mutable_arch = list_to_array(model_list)
    model = resnet20(mutable_arch)
    # print(model)

    # input = torch.zeros(16, 3, 32, 32)
