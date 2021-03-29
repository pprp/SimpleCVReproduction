import math
import paddle
import paddle.nn as nn
import numpy as np

__all__ = ['ResNet20']

##########   Original_Module   ##########


class Block_Conv1(nn.Layer):
    def __init__(self, in_planes, places, stride=1):
        super(Block_Conv1, self).__init__()
        self.conv1_input_channel = in_planes
        self.output_channel = places

        # defining conv1
        self.conv1 = self.Conv(
            self.conv1_input_channel, self.output_channel, kernel_size=3, stride=stride, padding=1)

    def Conv(self, in_places, places, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2D(in_channels=in_places, out_channels=places,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=False),
            nn.BatchNorm2D(places),
            nn.ReLU())

    def forward(self, x):
        out = self.conv1(x)
        return out


class BasicBolock(nn.Layer):
    def __init__(self, len_list, stride=1, group=1, downsampling=False):
        super(BasicBolock, self).__init__()
        global IND

        self.downsampling = downsampling
        self.adaptive_pooling = False
        self.len_list = len_list

        self.conv1 = self.Conv(
            self.len_list[IND-1], self.len_list[IND], kernel_size=3, stride=stride, padding=1)
        self.conv2 = self.Conv(
            self.len_list[IND], self.len_list[IND+1], kernel_size=3, stride=1, padding=1)

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels=self.len_list[IND-1], out_channels=self.len_list[IND+1],
                          kernel_size=1, stride=stride, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.len_list[IND+1]))
        elif not self.downsampling and (self.len_list[IND-1] != self.len_list[IND+1]):
            self.downsample = nn.Sequential(
                nn.Conv2D(in_channels=self.len_list[IND-1], out_channels=self.len_list[IND+1],
                          kernel_size=1, stride=stride, padding=0, bias_attr=False),
                nn.BatchNorm2D(self.len_list[IND+1]))
            self.downsampling = True
        self.relu = nn.ReLU()
        IND += 2

    def Conv(self, in_places, places, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2D(in_channels=in_places, out_channels=places,
                      kernel_size=kernel_size, stride=stride, padding=padding, bias_attr=False),
            nn.BatchNorm2D(places))

    def forward(self, x):
        residual = x
        x = self.relu(self.conv1(x))
        out = self.conv2(x)
        if self.downsampling:
            residual = self.downsample(residual)
        out += residual
        out = self.relu(out)
        return out


def _calculate_fan_in_and_fan_out(tensor, op):
    op = op.lower()
    valid_modes = ['linear', 'conv']
    if op not in valid_modes:
        raise ValueError(
            "op {} not supported, please use one of {}".format(op, valid_modes))
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError(
            "Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if op == 'linear':
        num_input_fmaps = tensor.shape[0]
        num_output_fmaps = tensor.shape[1]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
    receptive_field_size = 1
    if tensor.dim() > 2:
        receptive_field_size = tensor[0][0].size
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def _calculate_correct_fan(tensor, op, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError(
            "Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor, op)
    return fan_in if mode == 'fan_in' else fan_out


def kaiming_normal_(tensor, op='linear', a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, op, mode)
    gain = math.sqrt(2.0)
    std = gain / math.sqrt(fan)
    # Calculate uniform bounds from standard deviation
    bound = math.sqrt(3.0) * std
    with paddle.no_grad():
        return paddle.assign(paddle.uniform(tensor.shape, min=-bound, max=bound), tensor)


class ResNet(nn.Layer):
    def __init__(self, blocks, len_list, module_type=BasicBolock, num_classes=10, expansion=1):
        super(ResNet, self).__init__()
        self.block = module_type
        self.len_list = len_list
        self.expansion = expansion

        global IND
        IND = 0

        self.conv1 = Block_Conv1(in_planes=3, places=self.len_list[IND])
        IND += 1
        self.layer1 = self.make_layer(
            self.len_list, block=blocks[0], block_type=self.block, stride=1)
        self.layer2 = self.make_layer(
            self.len_list, block=blocks[1], block_type=self.block, stride=2)
        self.layer3 = self.make_layer(
            self.len_list, block=blocks[2], block_type=self.block, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
  
        print('='*10, self.len_list[-2], "+"*10, self.len_list)
        self.fc = nn.Linear(self.len_list[-2], num_classes)

        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight, op='conv',
                                mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2D):
                paddle.assign(paddle.ones(m.weight.shape), m.weight)
                paddle.assign(paddle.zeros(m.bias.shape), m.bias)
            elif isinstance(m, nn.Linear):
                kaiming_normal_(m.weight, op='linear',
                                mode='fan_out', nonlinearity='relu')
                paddle.assign(paddle.zeros(m.bias.shape), m.bias)

    def make_layer(self, len_list, block, block_type, stride):
        layers = []
        layers.append(block_type(len_list, stride, downsampling=True))
        for i in range(1, block):
            layers.append(block_type(len_list))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        x = self.avgpool(out3).flatten(1)
        x = self.fc(x)
        return x


##########   ResNet Model   ##########
# default block type --- BasicBolock for ResNet20;

def ResNet20(CLASS, len_list=None):
    return ResNet([3, 3, 3], len_list=len_list, num_classes=CLASS, module_type=BasicBolock)
