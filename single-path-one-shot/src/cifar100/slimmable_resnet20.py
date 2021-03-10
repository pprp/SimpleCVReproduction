import torch
import torch.nn as nn

arc_representation = "4-12-4-4-16-8-4-12-32-24-16-8-8-24-60-12-64-64-52-60"


class SwitchableBatchNorm2d(nn.Module):
    # num_features_list: [16, 32, 48, 64]
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
    传给该对象的应该是：
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
        idx = FLAGS.width_mult_list.index(self.width_mult)# 判定到底选择哪个作为index

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




def get_arc_list(arc_rep):
    return [int(item) for item in arc_rep.split('-')]


print(get_arc_list(arc_representation))

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()

        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion * planes,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion * planes))


class MutableBlock(nn.Module):
    '''
    这是多个层级：
    第一个block特殊处理
    第一个层级：6个block, 可选4 8 12 16
    第二个层级：6个block, 可选4 8 12 16 20 24 28 32
    第三个层级：6个block, 可选4 8 12 16 20 24 28 32 36 40 44 48 52 56 60 64
    最后一个层级，是Linear，独立实现，不在这个block实现
    ----
    params:
        # 由于所有的block都是6个，所以不需要特别指定
        层级可选目标: layer_channel_option 
                      eg: [4,8,12,16]
        层级具体选择: layer_channel_choice 
                      eg: [16, 4, 12, 16, 8, 16, 16, 24] 
                      最后一个是多出来的，需要提前知道输出

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
    NUM_OF_BLOCK = 6 # 每个层级都是6个block

    def __init__(self, layer_channel_option, layer_channel_choice):
        super(MutableBlock, self).__init__()
        assert len(layer_channel_choice) > 0 and len(layer_channel_option) > 0

        for i in range(NUM_OF_BLOCK):




class MutableModel(nn.Module):

    def __init__(self, num_class=100, input_size=32):
        super(MutableModel, self).__init__()

        self.first_conv = SlimmableConv2d(
            
        )