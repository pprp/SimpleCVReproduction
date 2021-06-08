from model.modules.dynamic_ops import *
import torch
import torch.nn as nn
from config.config import SuperNetSetting


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

    def _init_weights():
        pass

    def _build_layers(self):
        bn, conv, fc = self.bn, self.conv, self.fc
        