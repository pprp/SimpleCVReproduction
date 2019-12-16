import torch
import torch.nn as nn


class NetG(nn.Module):
    '''
    generator
    '''
    def __init__(self, opt):
        super(NetG, self).__init__()
        num_gen_feature = opt.ngf
        # 输入为nz维度的噪声（nz*1*1）
        """
        out = (in-1)*stride-2*padding+kernel_size
        """
        self.base = nn.Sequential(
            nn.ConvTranspose2d(opt.nz,
                               num_gen_feature * 8,
                               4,
                               1,
                               0,
                               bias=False),
            nn.BatchNorm2d(num_gen_feature * 8),
            nn.ReLU(inplace=True),

            # 4->8
            nn.ConvTranspose2d(num_gen_feature * 8,
                               num_gen_feature * 4,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(num_gen_feature * 4),
            nn.ReLU(True),

            # 8-> 16
            nn.ConvTranspose2d(num_gen_feature * 4,
                               num_gen_feature * 2,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(num_gen_feature * 2),
            nn.ReLU(True),

            # 16->32
            nn.ConvTranspose2d(num_gen_feature * 2,
                               num_gen_feature,
                               4,
                               2,
                               1,
                               bias=False),
            nn.BatchNorm2d(num_gen_feature),
            nn.ReLU(True),

            # last out 3 * 96 * 96
            nn.ConvTranspose2d(num_gen_feature, 3, 5, 3, 1, bias=False),
            # tanh 归一化到-1-1, sigmoid归一化 0-1
            nn.Tanh()
        )

    def forward(self, x):
        return self.base(x)


class NetD(nn.Module):
    '''
    discriminator
    '''
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.base = nn.Sequential(
            nn.Conv2d(3, ndf, 5, 3, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 2, 0, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        return self.base(x).view(-1)