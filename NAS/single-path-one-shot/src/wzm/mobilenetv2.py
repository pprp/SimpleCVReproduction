'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from thop import profile

from cal_complexity import print_model_parm_flops, print_model_parm_nums

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, kernel_size, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        # print('x size before block conv1',x.size())
        out = F.relu(self.bn1(self.conv1(x)))
        # print('x size after block conv1', out.size())
        out = F.relu(self.bn2(self.conv2(out)))
        # print('x size after block conv2', out.size())
        out = self.bn3(self.conv3(out))
        # print('x size after block conv3', out.size())
        out = out + self.shortcut(x) if self.stride==1 and out.size(3)==x.size(3) else out
        return out



class MobileNetLike(nn.Module):

    # (out_planes, stride)
    fixed = [(16,  1),
             (24,  1),  # NOTE: change stride 2 -> 1 for CIFAR10
             (24,  1),
             (32,  2),
             (32,  1),
             (32,  1),
             (64,  2),
             (64,  1),
             (64,  1),
             (64,  1),
             (96,  1),
             (96,  1),
             (96,  1),
             (160, 2),
             (160, 1),
             (160, 1),
             (320, 1)
    ]

    expand_kernel=[
        [3,3],
        [3,5],
        [3,7],
        [6,3],
        [6,5],
        [6,7]
    ]


    def __init__(self, cfg, num_classes=10):
        super(MobileNetLike, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.cfg=cfg
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.block_0= Block(32, 16, expansion=1, kernel_size=3, stride=1)
        self.layers,self.candidate_layers = self._make_layers()
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.linear = nn.Linear(1280, num_classes)
        self._initialize_weights()


    def _make_layers(self):
        layers= []

        candidate_layers=[None for _ in range(16)]
        cur_layer = 1

        for i in range(0, len(self.cfg), 3):
            layer2_index, op1, op2 = self.cfg[i:i+3]
            in_planes=self.fixed[cur_layer-1][0]
            out_planes=self.fixed[cur_layer][0]
            expansion=self.expand_kernel[op1-1][0]
            ker_size=self.expand_kernel[op1-1][1]
            layers.append(Block(in_planes, out_planes, expansion=expansion, kernel_size=ker_size, stride=self.fixed[cur_layer][1]))

            if layer2_index and op2: # make layer2_index>=3
                expansion_2 = self.expand_kernel[op2 - 1][0]
                ker_size_2 = self.expand_kernel[op2 - 1][1]
                candidate_layers[cur_layer-1] = Block(out_planes, self.fixed[layer2_index-2][0], expansion=expansion_2, kernel_size=ker_size_2, stride=1)
            cur_layer += 1
        return nn.Sequential(*layers), nn.Sequential(*candidate_layers)




    def forward(self, x):
        extra_input = [[] for _ in range(16)]
        prepare = F.relu(self.bn1(self.conv1(x)))
        input=self.block_0(prepare)
        cur_step=0
        output=None

        for i in range(0, len(self.cfg), 3):
            layer2_index, op1, op2 = self.cfg[i: i + 3]
            if extra_input[cur_step]!=[]:
                for item in extra_input[cur_step]:
                    if item.size(3)!=input.size(3):
                        pad = nn.ZeroPad2d(padding=(item.size(3) - input.size(3), 0, item.size(3) - input.size(3), 0))
                        input = pad(input)
                    input=input+item
            output=self.layers[cur_step](input)
            if layer2_index and op2!=0:
                extra_feature=self.candidate_layers[cur_step](output)
                extra_input[layer2_index-2].append(extra_feature)
            input=output
            cur_step+=1
        # print('before conv2 out:',output.size())
        out = F.relu(self.bn2(self.conv2(output)))
        # print('after conv2 out:',out.size())
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = self.avgpool(out)
        # print('after avgpool:', out.size())
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == "__main__":
    cfg=[14, 5, 4, 5, 3, 1, 8, 1, 1, 9, 1, 4, 8, 1, 3, 13, 5, 6, 16, 1, 2, 15, 1, 1, 13, 4, 2, 0, 1, 1, 15, 3, 3, 15, 5, 1, 0, 6, 5, 0, 3, 4, 17, 1, 3, 0, 5, 3]
    net = MobileNetLike(cfg)
    x = torch.randn(2, 3, 32, 32)
    print('By Thop')
    flops, params = profile(net, (x,))
    print('flops:', flops, 'params:', params)
    print('\n')
    print('By hand:')
    print_model_parm_flops(net, x)
    print_model_parm_nums(net)



