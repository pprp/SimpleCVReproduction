import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from prettytable import PrettyTable
from modules import SlimmableConv2d, SlimmableLinear, SwitchableBatchNorm2d

arc_representation = "4-12-4-4-16-8-4-12-32-24-16-8-8-24-60-12-64-64-52-60"
# "16-8-16-16-8-12-12-20-12-4-12-32-32-24-48-8-52-16-12-36"
max_arc_rep = "16-16-16-16-16-16-16-32-32-32-32-32-32-64-64-64-64-64-64-64"
# 1 2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20


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
        for i, idx in enumerate(idx_list):
            # SlimmableConv2d
            # print('layer idx:', idx,
            #       "\t two list: [in, out]: ", self.mc[idx], self.mc[idx+1])
            if i == 0:
                # 只有第一个conv stride设置为2
                layers.append(
                    SlimmableConv2d(
                        in_channels_list=self.mc[idx],
                        out_channels_list=self.mc[idx+1],
                        kernel_size=3,
                        stride=stride,
                        padding=1,
                        bias=False)
                )
            else:
                layers.append(
                    SlimmableConv2d(
                        in_channels_list=self.mc[idx],
                        out_channels_list=self.mc[idx+1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=False)
                )
            # SlimmableBN
            layers.append(
                SwitchableBatchNorm2d(self.mc[idx+1])
            )
            if i == 0:
                layers.append(nn.ReLU())

        self.body = nn.Sequential(*layers)
        self.shortcut1 = nn.Sequential()

        self.shortcut2 = nn.Sequential(
            SlimmableConv2d(
                in_channels_list=self.mc[idx_list[0]],
                out_channels_list=self.mc[idx_list[1]+1],
                kernel_size=1,
                stride=stride,
                padding=0,
                bias=False),
            SwitchableBatchNorm2d(self.mc[idx_list[1]+1])
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        res = self.body(x)

        # print(res.shape, x.shape)

        if res.shape[1] != x.shape[1] or res.shape[2] != x.shape[2]:
            res += self.shortcut2(x)
        else:
            res += self.shortcut1(x)
        return self.relu(res)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class MutableModel(nn.Module):

    def __init__(self,
                 arc_representation,
                 block,
                 num_blocks,
                 num_classes=100):
        super(MutableModel, self).__init__()

        self.lc, self.mc = get_configs()

        self.tb = PrettyTable()
        self.tb.field_names = ["op", "in", "out"]

        self.idx = 0  # 代表当前model的layer层数

        self.true_arc_index_list = self.get_true_arc_list(arc_representation)

        self.first_conv = SlimmableConv2d(
            in_channels_list=[3 for _ in range(
                len(self.mc[self.idx]))],  # 第一个不可变
            out_channels_list=self.mc[self.idx],
            kernel_size=3,
            stride=1,
            padding=1, bias=False
        )

        # print("first conv [in, out]:", [3 for _ in range(
        #     len(self.mc[self.idx]))], self.mc[self.idx])

        self.idx += 1  # 增加层

        # print("first bn [out]:", self.mc[self.idx])

        self.first_bn = SwitchableBatchNorm2d(self.mc[self.idx])

        self.layer1 = self._make_layer(MutableBlock, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(MutableBlock, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(MutableBlock, num_blocks[2], stride=2)

        # self.mutable_linear = SlimmableLinear(
        #     self.mc[self.idx-1], self.mc[self.idx])
        # self.last_linear = SlimmableLinear(
        #     self.mc[self.idx], [num_classes for _ in range(len(self.mc[self.idx]))])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = SlimmableLinear(
            self.mc[self.idx-1], [num_classes for _ in range(len(self.mc[self.idx-1]))])

        self._initialize_weights()

    def _make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for stride in strides:
            # 层数
            layers.append(
                block(idx_list=[self.idx, self.idx+1], stride=stride))
            self.idx += 2

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         init.kaiming_normal_(m.weight)
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1.0)
        #         m.bias.data.zero_()
        #     elif isinstance(m, nn.Linear):
        #         init.kaiming_normal_(m.weight)
        #         m.bias.data.zero_()
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, arc):
        self.get_true_arc_list(arc)
        self.apply(self.modify_channel)

        # 第一个layer
        x = F.relu(self.first_bn(self.first_conv(x)))
        # 三个层级
        # print(x.shape)
        x = self.layer1(x)
        # print(x.shape)
        x = self.layer2(x)
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

    def get_true_arc_list(self, arc_rep):
        '''
        实际的网络架构
        '''
        arc_list = [int(item) for item in arc_rep.split('-')]

        first_conv = arc_list[0]
        arc_level1 = arc_list[:7]
        arc_level2 = arc_list[7:13]
        arc_level3 = arc_list[13:20]

        def convert_idx(arc_level, level_name):
            arc_index = []
            for i in arc_level:
                arc_index.append(self.lc[level_name].index(i)+1)
            return arc_index

        true_arc_list = [*convert_idx(arc_level1, "level1"), *convert_idx(
            arc_level2, "level2"), *convert_idx(arc_level3, "level3")]

        print("true_arc_list:", true_arc_list)

        self.slimmableConv2d_in_choice_list = []
        self.slimmableLinear_in_choice_list = []
        self.slimmableLinear_out_choice_list = []
        self.slimmableConv2d_out_choice_list = []
        self.switchableBatchNorm2d_out_choice_list = []

        self.slimmableConv2d_in_choice_list.append(1)

        self.slimmableLinear_in_choice_list.append(true_arc_list[-2])
        self.slimmableLinear_in_choice_list.append(true_arc_list[-1])

        self.slimmableLinear_out_choice_list.append(true_arc_list[-1])
        self.slimmableLinear_out_choice_list.append(1)

        for num, index in enumerate(true_arc_list):
            # if num
            if num % 2 == 0:
                # shortcut node
                if num == 0:
                    # first
                    self.slimmableConv2d_in_choice_list.append(index)
                else:
                    # other node
                    self.slimmableConv2d_in_choice_list.append(
                        true_arc_list[num-2])
                    self.slimmableConv2d_in_choice_list.append(index)
            else:
                # normal node
                self.slimmableConv2d_in_choice_list.append(index)

        for num, index in enumerate(true_arc_list):
            if num % 2 == 0:
                # 第一个
                if num == 0:
                    self.slimmableConv2d_out_choice_list.append(index)

                    self.switchableBatchNorm2d_out_choice_list.append(index)
                else:
                    # 两次
                    self.slimmableConv2d_out_choice_list.append(index)
                    self.slimmableConv2d_out_choice_list.append(index)

                    self.switchableBatchNorm2d_out_choice_list.append(index)
                    self.switchableBatchNorm2d_out_choice_list.append(index)
            else:
                self.slimmableConv2d_out_choice_list.append(index)
                self.switchableBatchNorm2d_out_choice_list.append(index)

        print('^'*100)
        print(self.slimmableConv2d_in_choice_list)
        print(self.slimmableConv2d_out_choice_list)
        print(self.switchableBatchNorm2d_out_choice_list)
        print('^'*100)

        return true_arc_list

    def modify_channel(self, module):
        if isinstance(module, SlimmableConv2d):
            module.in_choice = self.slimmableConv2d_in_choice_list.pop(0)
            module.out_choice = self.slimmableConv2d_out_choice_list.pop(0)
            self.tb.add_row(
                ["SlimmableConv2d", module.in_choice, module.out_choice])
            print("SlimmableConv2d", module.in_choice, module.out_choice)

        if isinstance(module, SwitchableBatchNorm2d):
            module.out_choice = self.switchableBatchNorm2d_out_choice_list.pop(
                0)
            self.tb.add_row(["SwitchableBatchNorm2d", 0, module.out_choice])
            # print("SwitchableBatchNorm2d", module.out_choice)

        if isinstance(module, SlimmableLinear):
            module.in_choice = self.slimmableLinear_in_choice_list.pop(0)
            module.out_choice = self.slimmableLinear_out_choice_list.pop(0)
            self.tb.add_row(
                ["SlimmableLinear", module.in_choice, module.out_choice])
            print("SlimmableLinear", module.in_choice, module.out_choice)

        # print(self.tb)


def mutableResNet20():
    return MutableModel(arc_representation,
                        MutableBlock,
                        [3, 3, 3])


if __name__ == "__main__":
    model = mutableResNet20()

    input = torch.zeros(16, 3, 32, 32)
    output = model(input, arc_representation)

    model.apply(model.modify_channel)

    for i in model.children():
        print(i)
