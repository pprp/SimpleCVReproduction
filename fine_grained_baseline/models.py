import torch
from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = resnet18(pretrained=False)
        self.model.fc = nn.Linear(512, 200)

    def forward(self, x):
        return self.model(x)

class FineGrainedModel(nn.Module):
    def __init__(self):
        super(FineGrainedModel, self).__init__()
        self.base = nn.Sequential(
            resnet18().conv1,
            resnet18().bn1,
            resnet18().relu,
            resnet18().maxpool,
            resnet18().layer1,
            resnet18().layer2,
            resnet18().layer3,
            resnet18().layer4
        )

        self.classifier = nn.Linear(2048 ** 2, 200)

    def forward(self, input):
        # x shape: bs, c, h, w
        x = self.base(input)
        bs = x.size(0)
        x = x.view(bs, 2048, x.size(2)**2)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / 28 ** 2
        x = x.view(bs, -1)
        x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x)+1e-10))
        x = self.classifier(x)
        return x
