from nni.nas.pytorch import mutables
import torch.nn as nn
import torch
import torch.nn.functional as F
from nni.nas.pytorch.mutables import LayerChoice, InputChoice
from collections import OrderedDict


class Net(nn.Module):
    def __init__(self, hidden_size):
        super(Net, self).__init__()
        # two options of conv1
        self.conv1 = LayerChoice(OrderedDict([
            ("conv5x5", nn.Conv2d(1, 20, 5, 1)),
            ("conv3x3", nn.Conv2d(1, 20, 3, 1))
        ]), key='first_conv')
        # two options of mid_conv
        self.mid_conv = LayerChoice([
            nn.Conv2d(20, 20, 3, 1, padding=1),
            nn.Conv2d(20, 20, 5, 1, padding=2)
        ], key='mid_conv')
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 10)
        # skip connection over mid_conv
        self.input_switch = InputChoice(n_candidates=2,
                                        n_chosen=1,
                                        key='skip')

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        old_x = x
        x = F.relu(self.mid_conv(x))
        zero_x = torch.zeros_like(old_x)
        skip_x = self.input_switch([zero_x, old_x])
        x = torch.add(x, skip_x)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
