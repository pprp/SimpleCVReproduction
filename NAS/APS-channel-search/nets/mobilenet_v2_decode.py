import torch
from torch import nn
from nets.se_module import SELayer
from pdb import set_trace as br

__all__ = ['mobilenet_v2_decode']

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class DShortCut(nn.Module):
    def __init__(self, cin, cout, has_avg, has_BN, affine=True):
        super(DShortCut, self).__init__()
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, stride=1, padding=0, bias=False)
        if has_avg:
          self.avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        else:
          self.avg = None

        if has_BN:
          self.bn = nn.BatchNorm2d(cout, affine=affine)
        else:
          self.bn = None

    def forward(self, x):
        if self.avg:
          out = self.avg(x)
        else:
          out = x

        out = self.conv(out)
        if self.bn:
          out = self.bn(out)
        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, cfg, stride, expand_ratio, use_res, se=False, se_reduction=-1):
        super(InvertedResidual, self).__init__()
        self.inp = inp
        self.cfg = cfg
        self.stride = stride
        self.use_res_connect = use_res
        self.se = se
        self.se_reduction = se_reduction
        assert stride in [1, 2]

        # check cfg
        if expand_ratio != 1:
            assert len(cfg) == 3, "Wrong cfg length when expand_ratio!=1. Expected length 3 (c1-out, c2-out, c3-out), but got %d" % len(cfg)
            assert cfg[0] == cfg[1], "Wrong cfg, dw inc and outc should be equal"
            hidden_dim = cfg[1]
            oup = cfg[2]
        else:
            assert len(cfg) == 2, "Wrong cfg length when expand_ratio==1. Expected length 2 (c2-out, c3-out), but got %d" % len(cfg)
            hidden_dim = cfg[0]
            oup = cfg[1]

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

        if self.use_res_connect:
            if inp == oup:
                self.shortcut = nn.Sequential()
            else:
                self.shortcut = DShortCut(inp, oup, has_avg=False, has_BN=True)
        else:
            self.shortcut = None

        # define the se module
        if self.se:
            assert se_reduction > 0, "Must specify se reduction > 0"
            self.se_module = SELayer(oup, se_reduction)

    def forward(self, x):
        out = self.conv(x)
        if self.se:
            out = self.se_module(out)
        if self.use_res_connect:
            out += self.shortcut(x)
        return out

class MobileNetV2_Decode(nn.Module):
    def __init__(self,
                 cfg,
                 num_classes=1000,
                 se=False,
                 se_reduction=-1,
                 dropout_rate=0.0):
        """
        MobileNet V2 main class

        Args:
            cfg (list of int): the channel number configurations, len(cfg) == 52
            num_classes (int): Number of classes
            se (bool): use squeeze and exication or not
            se_reduction (int): the reduction number of se module
            dropout_rate (float): the dropout ratio

        """
        super(MobileNetV2_Decode, self).__init__()

        assert(len(cfg) == 52), "Error length of cfg, 52 expected but got length of %d" % len(cfg)
        block = InvertedResidual
        input_channel = cfg[0]
        last_channel = cfg[-1]
        self.dropout_rate = dropout_rate
        self.block_layer_num = 3

        inverted_residual_setting = [
            # t, c, n, s
            [1, cfg[1:3], 1, 1],
            [6, cfg[3:3+2*self.block_layer_num], 2, 2],
            [6, cfg[9:9+3*self.block_layer_num], 3, 2],
            [6, cfg[18:18+4*self.block_layer_num], 4, 2],
            [6, cfg[30:30+3*self.block_layer_num], 3, 1],
            [6, cfg[39:39+3*self.block_layer_num], 3, 2],
            [6, cfg[48:48+1*self.block_layer_num], 1, 1],
        ]

        # building first layer
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for stage_idx, (t, c, n, s) in enumerate(inverted_residual_setting):
            for i in range(n):
                stride = s if i == 0 else 1
                use_res = stride == 1 and i != 0
                if stage_idx == 0:
                    block_cfg = c[0:2]
                else:
                    block_cfg = c[i*self.block_layer_num: (i+1)*self.block_layer_num]
                features.append(block(\
                    input_channel, block_cfg, stride, t, use_res, se, se_reduction))

                if stage_idx == 0:
                    input_channel = c[-1]
                else:
                    input_channel = c[(i+1)*self.block_layer_num-1]

        # building last several layers
        assert input_channel == cfg[-2]
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(last_channel, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x):
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1 => must use reshape with x.shape[0]
        x = nn.functional.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)


def mobilenet_v2_decode(cfg, num_classes, se=False, se_reduction=-1, dropout_rate=0.0):
    """
    Constructs a MobileNetV2 architecture from decoding cfg.
    """
    return MobileNetV2_Decode(cfg, num_classes, se=se, se_reduction=se_reduction, dropout_rate=dropout_rate)

