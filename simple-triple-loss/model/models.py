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


class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.base = models.resnet18(pretrained=False)
        self.base.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = ClassBlock(1000,
                                     num_classes,
                                     dropout=False,
                                     relu=False)

    def forward(self, x):
        x = self.base(x)
        x = torch.squeeze(x)
        x, f = self.classifier(x)
        return x, f

    def save(self, file_path):
        if file_path is not None:
            torch.save(self.state_dict(), file_path)

    def load(self, weight_path):
        if weight_path is not None:
            self.load_state_dict(torch.load(weight_path))
