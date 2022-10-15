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

        self.classifier = nn.Linear(262144, 200)

    def forward(self, input):
        # x shape: bs, c, h, w
        x = self.base(input)
        BS, D, H, W = x.shape
        x = x.reshape(BS, D, H*W)
        # print(x.shape, torch.transpose(x, 1, 2).shape)
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / (H*W)
        x = x.view(BS, -1)
        x = F.normalize(torch.sign(x) * torch.sqrt(torch.abs(x)+1e-5))
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    input = torch.zeros((4, 3, 416, 416))

    model = FineGrainedModel()

    output = model(input)

    print(output.shape)
