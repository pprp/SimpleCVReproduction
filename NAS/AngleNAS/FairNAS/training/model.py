import torch.nn as nn
import math
from torch_blocks import *

def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

class Select_one_OP(nn.Module):
  def __init__(self, inp, oup, stride):
    super(Select_one_OP, self).__init__()
    self._ops = nn.ModuleList()
    self.input_channel = inp
    self.output_channel = oup
    self.stride = stride
    for idx, key in enumerate(config.blocks_keys):
      op = blocks_dict[key](inp, oup, stride)
      op.idx = idx
      self._ops.append(op)

  def forward(self, x, id):
    return self._ops[id](x)


class Network(nn.Module):
    def __init__(self, rngs, n_class=1000, input_size=224, width_mult=1.):
        super(Network, self).__init__()
        # setting of inverted residual blocks
        self.interverted_residual_setting = [
            # t, c, n, s
            [6, 32, 2, 2],
            [6, 40, 4, 2],
            [6, 80, 4, 2],
            [6, 96, 4, 1],
            [6, 192, 4, 2],
            [6, 320, 1, 1],
        ]
        # building first layer
        input_channel = int(32 * width_mult)
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv_bn = conv_bn(3, input_channel, 2)
        self.MBConv_ratio_1 = InvertedResidual(input_channel, int(16*width_mult), 3, 1, 1, 1)
        input_channel = int(16*width_mult)
        self.features = []
        num = 0
        # building inverted residual blocks
        for t, c, n, s in self.interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                rng = rngs[num]
                num += 1
                if rng < 0:
                    continue
                if i == 0:
                    op = blocks_dict[blocks_keys[rng]](input_channel, output_channel, s)
                    self.features.append(op)
                else:
                    op = blocks_dict[blocks_keys[rng]](input_channel, output_channel, 1)
                    self.features.append(op)
                input_channel = output_channel
        self.features = nn.Sequential(*self.features)
        # building last several layers
        self.conv_1x1_bn = conv_1x1_bn(input_channel, self.last_channel)
        self.avgpool = nn.AvgPool2d(input_size//32)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, n_class),
        )

    def forward(self, x, rngs=None):
        x = self.conv_bn(x)
        x = self.MBConv_ratio_1(x)
        x = self.features(x)
        x = self.conv_1x1_bn(x)
        x = self.avgpool(x)
        x = x.view(-1, self.last_channel)
        x = self.classifier(x)
        return x

    def architecture(self):
        arch = []
        for feat in self.features:
            if feat.stride == 2:
                arch.append('{}(reduce, oup={})'.format(feat.type, feat.oup))
            else:
                arch.append('{}(normal, oup={})'.format(feat.type, feat.oup))
        return arch