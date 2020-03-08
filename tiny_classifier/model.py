import torch
import torch.nn as nn
from torch.nn import init


class TinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(12,20, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(20)
        self.relu3 = nn.ReLU(True)

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(20, num_classes)

        # init
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                init.constant_(m.bias.data, 0.0)
            if isinstance(m,nn.Linear):
                m.weight.data.normal_(0,0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.gap(x)
        x = x.view(x.shape[0],-1)
        return self.classifier(x)


class DeepTinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.conv3 = nn.Conv2d(16, 32, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU(True)

        self.conv4 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.ReLU(True)

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(64, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)


class ShallowTinyModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight.data)
                init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)