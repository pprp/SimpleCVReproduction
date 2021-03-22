# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

'''
micro 代表搜索的对象是单元搜索不是全局搜索
macro 代表的才是全局搜索
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch import mutables
from ops import FactorizedReduce, StdConv, SepConvBN, Pool


class AuxiliaryHead(nn.Module):
    # 辅助头 用于接受tensor 然后输出类别
    # relu + avgpool(5x5)+conv(channel)+adaptiveavgpool+linear
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.pooling = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2d(5, 3, 2)
        )
        self.proj = nn.Sequential(
            StdConv(in_channels, 128),
            StdConv(128, 768)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, 10, bias=False)

    def forward(self, x):
        bs = x.size(0)
        x = self.pooling(x)
        x = self.proj(x)
        x = self.avg_pool(x).view(bs, -1)
        x = self.fc(x)
        return x


class Cell(nn.Module):
    # 最基本的单元随机选择sepConv+avgpool+maxpool+Indentity
    def __init__(self, cell_name, prev_labels, channels):
        super().__init__()
        self.input_choice = mutables.InputChoice(choose_from=prev_labels, n_chosen=1, return_mask=True,
                                                 key=cell_name + "_input")
        self.op_choice = mutables.LayerChoice([
            SepConvBN(channels, channels, 3, 1),
            SepConvBN(channels, channels, 5, 2),
            Pool("avg", 3, 1, 1),
            Pool("max", 3, 1, 1),
            nn.Identity()
        ], key=cell_name + "_op")

    def forward(self, prev_layers):
        chosen_input, chosen_mask = self.input_choice(prev_layers) 
        # 从之前的layers输出中选择的input和choisen mask（代表哪个被选中了）
        cell_out = self.op_choice(chosen_input) # 从LayerChoice中选择一个层，得到输出
        return cell_out, chosen_mask


class Node(mutables.MutableScope): 
    # 选择了两个最基本的单元
    def __init__(self, node_name, prev_node_names, channels):
        super().__init__(node_name)
        self.cell_x = Cell(node_name + "_x", prev_node_names, channels)
        self.cell_y = Cell(node_name + "_y", prev_node_names, channels)

    def forward(self, prev_layers):
        out_x, mask_x = self.cell_x(prev_layers)
        out_y, mask_y = self.cell_y(prev_layers)

        #mask_x 和 mask_y都是01串，1代表被选中了
        return out_x + out_y, mask_x | mask_y

class Calibration(nn.Module):
    # 校准： channel数目不相等的时候，用一个卷积来约束一下数目
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.process = None
        if in_channels != out_channels:
            self.process = StdConv(in_channels, out_channels)

    def forward(self, x):
        if self.process is None:
            return x
        return self.process(x)

class ReductionLayer(nn.Module):
    # 相当于Reduction 对于两个tensor进行分别降维，统一输出的channel为out channels
    def __init__(self, in_channels_pp, in_channels_p, out_channels):
        super().__init__()
        self.reduce0 = FactorizedReduce(
            in_channels_pp, out_channels, affine=False)
        self.reduce1 = FactorizedReduce(
            in_channels_p, out_channels, affine=False)

    def forward(self, pprev, prev):
        return self.reduce0(pprev), self.reduce1(prev)

class ENASLayer(nn.Module):
    # ENAS 中的一个层
    def __init__(self, num_nodes, in_channels_pp, in_channels_p, out_channels, reduction):
        '''
        num_nodes: 设置节点个数
        in_channels_pp: 前前层的channel个数
        in_channels_p: 代表前一层channel个数
        out_channel： 代表这一个ENAS Layer统一了channel个数
        reduction: 是一个flag用于判断是否是一个reduction cell or normal cell
        '''
        super().__init__()
        self.preproc0 = Calibration(in_channels_pp, out_channels)
        self.preproc1 = Calibration(in_channels_p, out_channels)
        # in_channel_pp代表从前前层
        # in_channel_p 代表从前一层

        self.num_nodes = num_nodes # 设置节点个数
        
        name_prefix = "reduce" if reduction else "normal"

        self.nodes = nn.ModuleList()
        node_labels = [mutables.InputChoice.NO_KEY,
                       mutables.InputChoice.NO_KEY]
        
        for i in range(num_nodes):
            # 每个节点，都在nodes中append一个Node对象
            node_labels.append("{}_node_{}".format(name_prefix, i))
            self.nodes.append(
                Node(node_labels[-1], node_labels[:-1], out_channels))
        
        self.final_conv_w = nn.Parameter(torch.zeros(
            out_channels, self.num_nodes + 2, out_channels, 1, 1), requires_grad=True)

        self.bn = nn.BatchNorm2d(out_channels, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.final_conv_w)

    def forward(self, pprev, prev):
        # 输入是前一层和前前层
        pprev_, prev_ = self.preproc0(pprev), self.preproc1(prev)
        # pprev_和prev_代表校准channel为channel_out的tensor

        prev_nodes_out = [pprev_, prev_] # 加到一个list中

        nodes_used_mask = torch.zeros(self.num_nodes + 2, dtype=torch.bool, device=prev.device)
        # nodes used mask [num nodes+2] 

        for i in range(self.num_nodes):
            node_out, mask = self.nodes[i](prev_nodes_out) # Node 中的 Cell 会从这两个输入中选择
            
            nodes_used_mask[:mask.size(0)] |= mask.to(node_out.device) # 判断哪里选择了被连接上了
            
            prev_nodes_out.append(node_out) # 添加到list中，这样可以供下一个选择

        unused_nodes = torch.cat([out for used, out in zip(
            nodes_used_mask, prev_nodes_out) if not used], 1) # 若没有被使用，保存到这个变量
        unused_nodes = F.relu(unused_nodes) # 没有被使用说明已经成为了自由变量

        # shape: [out_channels, num_nodes + 2, out_channels, 1, 1]
        conv_weight = self.final_conv_w[:, ~nodes_used_mask, :, :, :] # ~nodes代表取反 -x-1

        conv_weight = conv_weight.view(conv_weight.size(0), -1, 1, 1)
        out = F.conv2d(unused_nodes, conv_weight) # 对所有自由输出结点的值进行处理
        return prev, self.bn(out)


class MicroNetwork(nn.Module):
    def __init__(self, num_layers=2, num_nodes=5, out_channels=24, in_channels=3, num_classes=10,
                 dropout_rate=0.0, use_aux_heads=False):
        super().__init__()
        self.num_layers = num_layers
        self.use_aux_heads = use_aux_heads

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 3, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels * 3)
        )

        pool_distance = self.num_layers // 3
        pool_layers = [pool_distance, 2 * pool_distance + 1]
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        c_pp = c_p = out_channels * 3
        c_cur = out_channels

        for layer_id in range(self.num_layers + 2): # 额外加了两层
            reduction = False
            if layer_id in pool_layers:
                c_cur, reduction = c_p * 2, True
                self.layers.append(ReductionLayer(c_pp, c_p, c_cur))
                c_pp = c_p = c_cur

            self.layers.append(
                ENASLayer(num_nodes, c_pp, c_p, c_cur, reduction))

            if self.use_aux_heads and layer_id == pool_layers[-1] + 1:
                self.layers.append(AuxiliaryHead(c_cur, num_classes))
                
            c_pp, c_p = c_p, c_cur

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dense = nn.Linear(c_cur, num_classes)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        bs = x.size(0)
        prev = cur = self.stem(x)
        aux_logits = None

        for layer in self.layers:
            if isinstance(layer, AuxiliaryHead):
                if self.training:
                    aux_logits = layer(cur)
            else:
                prev, cur = layer(prev, cur)

        cur = self.gap(F.relu(cur)).view(bs, -1)
        cur = self.dropout(cur)
        logits = self.dense(cur)

        if aux_logits is not None:
            return logits, aux_logits
        return logits
