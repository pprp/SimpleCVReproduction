import torch
import torch.nn as nn
import torchvision
from ptsemseg.models.utils import unetConv2, unetUp

class Decoder(nn.Module):
  def __init__(self, in_channels, middle_channels, out_channels):
    super(Decoder, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
    self.conv_relu = nn.Sequential(
        nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
        )
  def forward(self, x1, x2):
    x1 = self.up(x1)
    x1 = torch.cat((x1, x2), dim=1)
    x1 = self.conv_relu(x1)
    return x1

class unet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        self.base_model = torchvision.models.resnet18(False)
        self.base_layers = list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            self.base_layers[1],
            self.base_layers[2])
        self.layer2 = nn.Sequential(*self.base_layers[3:5])
        self.layer3 = self.base_layers[5]
        self.layer4 = self.base_layers[6]
        self.layer5 = self.base_layers[7]
        self.decode4 = Decoder(512, 256+256, 256)
        self.decode3 = Decoder(256, 256+128, 256)
        self.decode2 = Decoder(256, 128+64, 128)
        self.decode1 = Decoder(128, 64+64, 64)
        self.decode0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_classes, 1)

    def forward(self, input):
        # print("----", input.shape)
        e1 = self.layer1(input) # 64,128,128
        e2 = self.layer2(e1) # 64,64,64
        e3 = self.layer3(e2) # 128,32,32
        e4 = self.layer4(e3) # 256,16,16
        f = self.layer5(e4) # 512,8,8
        # print('----', f.shape)
        d4 = self.decode4(f, e4) # 256,16,16
        d3 = self.decode3(d4, e3) # 256,32,32
        d2 = self.decode2(d3, e2) # 128,64,64
        d1 = self.decode1(d2, e1) # 64,128,128
        d0 = self.decode0(d1) # 64,256,256
        # print('--------', d0.shape)
        out = self.conv_last(d0) # 1,256,256
        return out





'''
class unet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=2, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        #filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final

'''

'''
class DoubleConv(nn.Module):               #将两个卷积操作放在一块               
    def __init__(self,in_ch,out_ch):       #in_ch输入信号的通道，out_ch输出信号的通道
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),    #数据归一化
                nn.ReLU(inplace = True),   #激活函数，True，将会改变输入的数据 ，否则不会改变原输入
                nn.Conv2d(out_ch,out_ch,3,padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace = True) 
        )        
    def forward(self, x):
        return self.conv(x)

class unet(nn.Module):
    def __init__(self, in_ch, out_ch, n_classes=2):
        super(unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)        #需要每次把图片的尺寸缩短一半
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        #反卷积
        self.up6 = nn.ConvTranspose2d(1024,512,2,stride=2)
        self.conv6 = DoubleConv(1024,512)
        self.up7 = nn.ConvTranspose2d(512,256,2,stride=2)
        self.conv7 = DoubleConv(512,256)
        self.up8 = nn.ConvTranspose2d(256,128,2,stride=2)
        self.conv8 = DoubleConv(256,128)
        self.up9 = nn.ConvTranspose2d(128,64,2,stride=2)
        self.conv9 = DoubleConv(128,64)
        self.out = nn.Conv2d(64, n_classes, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up6 = self.up6(c5)
        merge6 = torch.cat([up6,c4],dim=1)#按维数1（列）拼接，0是行拼接
        c6 = self.conv6(merge6)
        up7 = self.up7(c6)
        merge7 = torch.cat([up7,c3],dim=1)
        c7 = self.conv7(merge7)
        up8 = self.up8(c7)
        merge8 = torch.cat([up8,c2],dim=1)
        c8 = self.conv8(merge8)
        up9 = self.up9(c8)
        merge9 = torch.cat([up9,c1],dim=1)
        c9 = self.conv9(merge9)
        out = self.out(c9)
        
        return out
'''

