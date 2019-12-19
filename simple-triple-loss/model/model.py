import torch.nn as nn
import torch

from torch.autograd import Variable
from torchvision import models


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=False, relu=False):
        super(ClassBlock, self).__init__()
        # base block
        blocks = []
        blocks.append(nn.BatchNorm1d(input_dim))
        if relu:
            blocks.append(nn.ReLU())
        if dropout:
            blocks.append(nn.Dropout(0.5))
        self.base = nn.Sequential(*blocks)

        # classifier
        self.classifier = nn.Sequential(nn.Linear(input_dim, class_num))

    def forward(self, x):
        x = self.base(x)
        # apply L2 norm
        x_norm = x.norm(p=2, dim=1, keepdim=True) + 1e-8
        f = x.div(x_norm)
        x = self.classifier(f)
        return x, f


class Res18(nn.Module):
    def __init__(self, num_classes):
        super(Res18, self).__init__()
        self.base = models.resnet18(pretrained=False)
        self.base.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(2048,
                                     num_classes,
                                     dropout=False,
                                     relu=False)

    def forward(self, x):
        x = self.base(x)
        x = torch.squeeze(x)
        x, f = self.classifier(x)
        return x, f

    def save(self, filename):
        self.save(filename)
