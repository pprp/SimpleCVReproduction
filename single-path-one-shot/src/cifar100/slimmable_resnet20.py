import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

arc_representation = "4-12-4-4-16-8-4-12-32-24-16-8-8-24-60-12-64-64-52-60"


def get_configs():
    '''
    level_config 保存每个层级可选的通道个数
    model_config 保存模型每个Layer可选的通道个数
    '''
    level_config = {
        "level1": [4, 8, 12, 16],
        "level2": [4, 8, 12, 16, 20, 24, 28, 32],
        "level3": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
    }

    model_config = {}

    for i in range(0, 7):
        model_config[i] = level_config["level1"]
    for i in range(7, 13):
        model_config[i] = level_config["level2"]
    for i in range(13, 20):
        model_config[i] = level_config["level3"]

    return level_config, model_config

# lc,mc = get_configs()

# for i, v in mc.items():
#     print(i+1, v)


class SlimmableLinear(nn.Linear):
    def __init__(self, in_features_list, out_features_list, bias=True):
        super(SlimmableLinear, self).__init__(
            max(in_features_list), max(out_features_list), bias=bias)
        self.in_features_list = in_features_list
        self.out_features_list = out_features_list
        self.width_mult = max(FLAGS.width_mult_list)

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        self.in_features = self.in_features_list[idx]
        self.out_features = self.out_features_list[idx]
        weight = self.weight[:self.out_features, :self.in_features]
        if self.bias is not None:
            bias = self.bias[:self.out_features]
        else:
            bias = self.bias
        return nn.functional.linear(input, weight, bias)


class SwitchableBatchNorm2d(nn.Module):
    # num_features_list: [16, 32, 48, 64]
    # 与out_channels_list相一致
    def __init__(self, num_features_list):
        super(SwitchableBatchNorm2d, self).__init__()
        self.num_features_list = num_features_list
        self.num_features = max(num_features_list)  # 4
        bns = []
        for i in num_features_list:
            bns.append(nn.BatchNorm2d(i))  # 分别有多个bn与其对应

        self.bn = nn.ModuleList(bns)  # 其中包含4个bn

        self.width_mult = max(FLAGS.width_mult_list)  # 4
        self.ignore_model_profiling = True

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)
        y = self.bn[idx](input)
        return y


class SlimmableConv2d(nn.Conv2d):
    # in_channels_list: [3,3,3,3]
    # out_channels_list: [16, 32, 48, 64]
    # kernel_size: 3
    # stride: 1
    # padding: 3
    # bias=False
    '''
    单个convolution
    传给该对象的应该是：
        上一层的可选channel: in_channels_list
        下一层的可选channel: out_channels_list
    '''

    def __init__(self,
                 in_channels_list,
                 out_channels_list,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups_list=[1],
                 bias=True):
        super(SlimmableConv2d, self).__init__(max(in_channels_list),
                                              max(out_channels_list),
                                              kernel_size,
                                              stride=stride,
                                              padding=padding,
                                              dilation=dilation,
                                              groups=max(groups_list),
                                              bias=bias)

        self.in_channels_list = in_channels_list
        self.out_channels_list = out_channels_list
        self.groups_list = groups_list
        if self.groups_list == [1]:
            self.groups_list = [1 for _ in range(len(in_channels_list))]
        self.width_mult = max(FLAGS.width_mult_list)
        # 这里必须选用最大的channel数目作为共享的对象

    def forward(self, input):
        idx = FLAGS.width_mult_list.index(self.width_mult)  # 判定到底选择哪个作为index

        self.in_channels = self.in_channels_list[idx]
        self.out_channels = self.out_channels_list[idx]  # 找到对应的in和out

        self.groups = self.groups_list[idx]  # 组卷积

        weight = self.weight[:self.out_channels, :self.in_channels, :, :]
        if self.bias is not None:
            bias = self.bias[:self.out_channels]
        else:
            bias = self.bias
        y = nn.functional.conv2d(input, weight, bias, self.stride, self.padding,
                                 self.dilation, self.groups)
        return y


class MutableBlock(nn.Module):
    '''
    这是多个层级：
    第一个block特殊处理
    第一个层级：6个block, 可选4 8 12 16
    第二个层级：6个block, 可选4 8 12 16 20 24 28 32
    第三个层级：6个block, 可选4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64
    最后一个层级，是Linear，独立实现，不在这个block实现

    MutableBlock中包括两个SlimmableConv2d + shortcut层
    ----
    params:
        # 由于所有的block都是6个，所以不需要特别指定
        layer_channel_option： 层级可选目标: 
                      eg: [4,8,12,16]
        layer_channel_choice：层级具体选择: 
                      eg: [16, 4, 12, 16, 8, 16, 16, 24]
                      最后一个是多出来的，需要提前知道输出
        idx_list: 当前模型Layer:
                      eg: [1,2]
        stride: 当前模型block设置步长
                      eg: 1 or 2

    eg:
        以4-12-4-4-16-8-4-12-32-24-16-8-8-24-60-12-64-64-52-60为例：
        第一个层级：
            layer_channel_option: [4 8 12 16]
            layer_channel_choice: [4 12 4 4 16 8 4]

        第二个层级：
            layer_channel_option: [4 8 12 16 20 24 28 32]
            layer_channel_choice: [4 12 32 24 16 8 8]

        第三个层级：
            layer_channel_option: [4 8 12 16 20 24 28 32]
            layer_channel_choice: [8 24 60 12 64 64 52]

    '''

    def __init__(self, idx_list, stride=1):
        super(MutableBlock, self).__init__()

        assert stride in [1, 2]

        self.lc, self.mc = get_configs()

        layers = []

        # 两层卷积
        for idx in idx_list:
            # SlimmableConv2d
            layers.append(
                SlimmableConv2d(
                    in_channels_list=self.mc[idx],
                    out_channels_list=self.mc[idx+1],
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False)
            )
            # SlimmableBN
            layers.append(
                SwitchableBatchNorm2d(self.mc[idx+1])
            )
            layers.append(nn.ReLU(inplace=True))

        self.body = nn.Sequential(*layers)

        # self.residual_connection = stride != 1

        # if self.residual_connection:  # 构建shortcut层
        # 由于通道可变，必定存在shortcut
        self.shortcut = nn.Sequential(
            SlimmableConv2d(
                in_channels_list=self.mc[idx_list[0]],
                out_channels_list=self.mc[idx_list[1]+1],
                kernel_size=1,
                stride=stride,
                bias=False),
            SwitchableBatchNorm2d(self.mc[idx_list[1]+1])
        )

        self.post_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.body(x)
        res += self.shortcut(x)
        return self.post_relu(res)


class MutableModel(nn.Module):

    def __init__(self, arc_representation,
                 block,
                 num_blocks,
                 num_classes=100,
                 input_size=32,
                 ):
        super(MutableModel, self).__init__()

        self.lc, self.mc = get_configs()

        self.idx = 0  # 代表当前model的layer层数

        self.true_arc_dict = self.get_true_arc_dict(arc_representation)

        self.first_conv = SlimmableConv2d(
            in_channels_list=[3],  # 第一个不可变
            out_channels_list=self.mc[idx],
            kernel_size=3,
            stride=1,
            padding=1, bias=False
        )

        self.idx += 1  # 增加层

        self.first_bn = SwitchableBatchNorm2d(mc[idx])

        self.layer1 = self._make_layer(MutableBlock, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(MutableBlock, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(MutableBlock, num_blocks[2], stride=2)

        self.mutable_linear = SlimmableLinear(self.mc[self.idx-1],self.mc[self.idx])
        self.last_linear = SlimmableLinear(self.mc[self.idx], [num_classes for _ in range(len(self.mc[self.idx]))])


    def _make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []

        for stride in strides:
            # 层数
            layers.append(block(idx_list=[self.idx,self.idx+1],stride=stride))
            self.idx += 2
        
        return nn.Sequential(*layers)


    def forward(self, x):
        # 第一个layer
        x = F.relu(self.first_bn(self.first_conv))
        # 三个层级
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = F.avg_pool2d(x,x.size()[3])
        x = x.view(x.size(0),-1)
        x = self.mutable_linear(x)
        x = self.last_linear(x)
        return x

    def get_true_arc_dict(self, arc_rep):
        '''
        实际的网络架构
        '''
        arc_list = [int(item) for item in arc_rep.split('-')]
        arc_level1 = arc_list[:6]
        arc_level2 = arc_list[7:13]
        arc_level3 = arc_list[14:19]

        print(arc_level1, arc_level2, arc_level3)

        return {
            "arc_level1": arc_level1,
            "arc_level2": arc_level2,
            "arc_level3": arc_level3
        }


def mutableResNet20():
    return MutableModel(arc_representation,
                        MutableBlock,
                        [3, 3, 3])
