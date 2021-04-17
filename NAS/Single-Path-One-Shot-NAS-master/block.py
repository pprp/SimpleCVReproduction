import torch
import torch.nn as nn


def channel_shuffle(x):
    """
        code from https://github.com/megvii-model/SinglePathOneShot/src/Search/blocks.py#L124
    """
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class Choice_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, supernet=True):
        super(Choice_Block, self).__init__()
        padding = kernel // 2
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=kernel, stride=stride, padding=padding,
                      bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw_linear
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=kernel, stride=2, padding=padding,
                          bias=False, groups=self.in_channels),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle(x)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y


class Choice_Block_x(nn.Module):
    def __init__(self, in_channels, out_channels, stride, supernet=True):
        super(Choice_Block_x, self).__init__()
        if supernet:
            self.affine = False
        else:
            self.affine = True
        self.stride = stride
        self.in_channels = in_channels
        self.mid_channels = out_channels // 2
        self.out_channels = out_channels - in_channels

        self.cb_main = nn.Sequential(
            # dw
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=stride,
                      padding=1, bias=False, groups=self.in_channels),
            nn.BatchNorm2d(self.in_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            nn.ReLU(inplace=True),
            # dw
            nn.Conv2d(self.mid_channels, self.mid_channels, kernel_size=3, stride=1,
                      padding=1, bias=False, groups=self.mid_channels),
            nn.BatchNorm2d(self.mid_channels, affine=self.affine),
            # pw
            nn.Conv2d(self.mid_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.out_channels, affine=self.affine),
            nn.ReLU(inplace=True)
        )
        if stride == 2:
            self.cb_proj = nn.Sequential(
                # dw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=2,
                          padding=1, groups=self.in_channels, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                # pw
                nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.in_channels, affine=self.affine),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = channel_shuffle(x)
            y = torch.cat((self.cb_main(x1), x2), 1)
        else:
            y = torch.cat((self.cb_main(x), self.cb_proj(x)), 1)
        return y
