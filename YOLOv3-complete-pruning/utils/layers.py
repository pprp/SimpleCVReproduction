import torch.nn as nn
import torch

'''
shape对应的数必须是奇数

[spatialmaxpool]
# 52x52 26x26 13x13
from=75, 70, 62
shape=13, 13, 13
out_plane = 128
'''
class SpatialMaxpool(nn.Module):
    def __init__(self, shapes, filters, out_plane=128):
        # shapes: type=list
        # filters: type=list
        super(SpatialMaxpool, self).__init__()

        self.spp1 = nn.MaxPool2d(  # 52
            kernel_size=shapes[0],
            stride=1,
            padding=int((shapes[0] - 1) // 2))
        self.conv1x1_1 = nn.Conv2d(filters[0], out_plane, kernel_size=3,
                                   stride=2,
                                   padding=1)

        self.spp2 = nn.MaxPool2d(  # 26
            kernel_size=shapes[1],
            stride=1,
            padding=int((shapes[1] - 1) // 2))
        self.conv1x1_2 = nn.Conv2d(filters[1], out_plane, kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.spp3 = nn.MaxPool2d(  # 13
            kernel_size=shapes[2],
            stride=1,
            padding=int((shapes[2] - 1) // 2))
        self.conv1x1_3 = nn.Conv2d(filters[2], out_plane, kernel_size=1,
                                   stride=1,
                                   padding=0)

        self.us_spp3 = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x1, x2, x3):
        # 52 26 13
        out1 = self.conv1x1_1(self.spp1(x1))
        out2 = self.conv1x1_2(self.spp2(x2))
        out3 = self.us_spp3(self.conv1x1_3(self.spp3(x3)))
        return out1+out2+out3


'''
并不是常规的se，而是特殊的se
# layer=80
[se]
# attention feature
from=62, -1
reduction=4
out_plane=256# 这个地方要跟上边的值保持一致
'''
class SpecialSE(nn.Module):
    def __init__(self, in_plane, out_plane, reduction=4):
        super(SpecialSE, self).__init__()
        self.out_plane = out_plane
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_plane, in_plane//4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_plane//4, out_plane, bias=False),
            nn.Sigmoid()
        )

    def forward(self, attention, y):
        # apply the attention extracted from x to y
        b, c, _, _ = attention.size()
        attention = self.gap(attention).view(b, c)
        channel_attention = self.fc(attention).view(b, self.out_plane, 1, 1)
        return channel_attention * y


if __name__ == "__main__":
    model=SpatialMaxpool(shapes=[13, 13, 13], filters=[128, 128, 512],out_plane=256)

    x1 = torch.zeros((3, 128, 52, 52))
    x2 = torch.zeros((3, 128, 26, 26))
    x3 = torch.zeros((3, 512, 13, 13))

    print(model(x1,x2,x3).shape)

    # # attention, feature
    # model = SpecialSE(512, 256, reduction=4)

    # x1 = torch.zeros(4, 512, 13, 13)
    # y1 = torch.zeros(4, 256, 26, 26)

    # # attention, feature
    # print(model(x1, y1).shape)
