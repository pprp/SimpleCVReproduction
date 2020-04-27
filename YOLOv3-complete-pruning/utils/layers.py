import torch.nn as nn
import torch

'''
[LSRCEM]

'''

class SpatialMaxpool(nn.Module):
    def __init__(self, shapes, filters, out_plane=128):
        # shapes: type=list
        # filters: type=list
        super(SpatialMaxpool, self).__init__()

        self.spp1 = nn.MaxPool2d(  # 52
            kernel_size=shapes[0],
            stride=2,
            padding=int((shapes[0] - 1) // 2))
        self.conv1x1_1 = nn.Conv2d(filters[0], out_plane, kernel_size=1,
                                 stride=1,
                                 padding=0)


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

        print(out1.shape, out2.shape, out3.shape)
        return out1+out2+out3
    

if __name__ == "__main__":
    model=SpatialMaxpool(shapes=[52, 26, 13], filters=[128, 256, 512],out_plane=256)
    
    x1 = torch.zeros((3, 128, 52, 52))
    x2 = torch.zeros((3, 256, 26, 26))
    x3 = torch.zeros((3, 512, 13, 13))

    print(model(x1,x2,x3).shape)

