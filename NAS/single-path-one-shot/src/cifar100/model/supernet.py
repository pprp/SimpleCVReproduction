import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
# from config.config import SuperNetSetting

from modules.dynamic_ops import *

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


class SuperNet(nn.Module):
    def __init__(self, conv, bn, fc, config=SuperNetSetting):
        super().__init__()
        self.bn_cls = bn
        self.conv_cls = conv
        self.fc_cls = fc
        self.config = config
        self.max_list = [max(x) for x in self.config]
        self.size_list = [len(x) for x in self.config]
        self._build_layers()
        self._init_weights()

    def _init_weights(self):
        m = self.modules()
        classname = m.__class__.__name__
        # print(classname)
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            init.kaiming_uniform_(
                m.weight, mode='fan_out', nonlinearity='relu')

    def _build_layers(self):
        bn, conv, fc = self.bn_cls, self.conv_cls, self.fc_cls

        self.first_conv = conv([3], self.config[0])
        self.first_bn = bn([3], self.config[0])

        for i in range(18):
            stride = 2 if i in [6, 12] else 1
            setattr(
                self, f"conv-{i}", conv(self.config[i], self.config[i+1], stride=stride))
            setattr(self, f"bn-{i}", bn(self.config[i], self.config[i+1]))

            # downsample
            if i % 2 == 0:
                setattr(
                    self, f"conv-{i}-down", conv(self.config[i], self.config[i+2], stride=stride, down=True))
                setattr(self, f"bn-{i}-down",
                        bn(self.config[i], self.config[i+2]))

        self.fc = fc(self.config[18], 100)

    def forward(self, x, arch, sc=True):
        # skip connection for sc
        x = self.first_conv(x, 3, arch[0])
        x = self.first_bn(x, 3, arch[0])
        x = F.relu(x)

        for block in range(3):
            for layer in [0, 2, 4]:
                base = block * 6 + layer
                shortcut = x
                # first
                x = getattr(self, f"conv-{base}")(x, arch[base], arch[base+1])
                x = getattr(self, f"bn-{base}")(x, arch[base], arch[base+1])
                x = F.relu(x)
                # second
                x = getattr(self, f"conv-{base+1}")(x,
                                                    arch[base+1], arch[base+2])
                x = getattr(self, f"bn-{base+1}")(x,
                                                  arch[base+1], arch[base+2])
                # shortcut
                if sc or layer == 0 or arch[base] != arch[base+2]:
                    shortcut = getattr(
                        self, f"conv-{base}-down")(shortcut, arch[base], arch[base+2])
                    shortcut = getattr(
                        self, f"bn-{base}-down")(shortcut, arch[base], arch[base+2])
                x = x + shortcut
                x = F.relu(x)
            x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
            return self.fc(x, arch[18])


if __name__ == "__main__":
    arch = "4-12-4-4-16-8-4-12-32-24-16-8-8-24-60-12-64-64-52-60"
    arch = [int(x) for x in arch.split("-")]
    model = SuperNet(FullConv, FullBN, FullFC)
    input = torch.zeros([4, 3, 32, 32])
    output = model(input, arch)
    print(output.shape)
