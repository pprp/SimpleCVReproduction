import torch
import torch.nn as nn


class KeyPointModel(nn.Module):
    def __init__(self):
        super(KeyPointModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU(True)
        self.maxpool1 = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(6, 12, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(12)
        self.relu2 = nn.ReLU(True)
        self.maxpool2 = nn.MaxPool2d((2, 2))

        # self.conv3 = nn.Conv2d(12, 20, 3, 1, 1)
        # self.bn3 = nn.BatchNorm2d(20)
        # self.relu3 = nn.ReLU(True)

        self.gap = nn.AdaptiveMaxPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(12, 2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = self.relu3(x)

        x = self.gap(x)
        x = x.view(x.shape[0], -1)
        return self.classifier(x)

if __name__ == "__main__":
    model = KeyPointModel()

    input_tensor = torch.zeros((4, 3, 256, 256))

    output_tensor = model(input_tensor)

    print(output_tensor.shape, output_tensor)

