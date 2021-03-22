import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES
from genotypes import Genotype
import math
import numpy as np
from config import config
import copy

class MixedOp(nn.Module):

  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for idx, primitive in enumerate(PRIMITIVES):
      op = OPS[primitive](C, stride, True)
      op.idx = idx
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=True))
      self._ops.append(op)

  def forward(self, x, rng):
    return self._ops[rng](x)


class Cell(nn.Module):

  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    if reduction_prev:
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=True)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=True)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=True)
    self._steps = steps
    self._multiplier = multiplier
    self._C = C
    self.out_C = self._multiplier * C
    self.reduction = reduction

    self._ops = nn.ModuleList()
    self._bns = nn.ModuleList()
    self.time_stamp = 1 

    for i in range(self._steps):
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride)
        self._ops.append(op)

  def forward(self, s0, s1, rngs):
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)
    states = [s0, s1]
    offset = 0
    for i in range(self._steps):
      s = sum(self._ops[offset+j](h, rngs[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)
    return torch.cat(states[-self._multiplier:], dim=1)

class Network_ImageNet(nn.Module):
    def __init__(self, C=48, num_classes=1000, layers=14, steps=4, multiplier=4):
        super(Network_ImageNet, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._steps = steps
        self._multiplier = multiplier

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction = False
        reduction_prev = True
        
        num_reduction = 2
        for i in range(num_reduction+1):
            if i > 0:
                C_curr *= 2
                reduction = True
            cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = False
            self.cells += [cell]
            C_prev_prev, C_prev = multiplier*C_curr, multiplier*C_curr

        for i in range(layers - num_reduction-1):
            self.insert_layer()

        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.affine = True
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, input, rngs):
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            if cell.reduction == True and not s0.shape[1] == s1.shape[1]:
                s0, s1 = s1, cell(s1, s1, rngs[i])
            else:    
                s0, s1 = s1, cell(s0, s1, rngs[i])
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def insert_layer(self):
        reduction_prev = False
        _, insert_idx = self.get_insert_location()
        C_prev_prev, C_prev, C_curr = self._C, self._C, self._C
        
        if insert_idx > 0:
            reduction_prev = self.cells[insert_idx-1].reduction
            C_prev = self.cells[insert_idx-1].out_C
        if insert_idx > 1:
            C_prev_prev = self.cells[insert_idx-2].out_C
        for i in range(insert_idx):
            if self.cells[i].reduction == True:
                C_curr *=2

        cell = Cell(self._steps, self._multiplier, C_prev_prev, C_prev, C_curr, False, reduction_prev).cuda()

        for m in cell.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                m.affine = True
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        for c in self.cells:
            c.time_stamp *= 2
            
        self.cells.insert(insert_idx, cell)

        return insert_idx

    def get_insert_location(self):
        last_insert_idx, insert_idx = -1, -1
        reduce_idxs = []

        for i, cell in enumerate(self.cells):
            if cell.reduction == True:
                reduce_idxs.append(i)
        assert(len(reduce_idxs) == 2)
        
        front = reduce_idxs[0]
        mid = reduce_idxs[1] - reduce_idxs[0] - 1
        back = len(self.cells) - reduce_idxs[1] - 1
        if front == mid and mid == back:
            last_insert_idx, insert_idx = len(self.cells) - 1, reduce_idxs[0]
        elif mid < front:
            last_insert_idx, insert_idx = reduce_idxs[0] - 1, reduce_idxs[1]
        elif back < mid:
            last_insert_idx, insert_idx = reduce_idxs[1] - 1, len(self.cells)
        else:
            assert('wrong insert order!')
        return last_insert_idx, insert_idx

    def architecture(self):
        arch = []
        for cell in self.cells:
            if cell.reduction == True:
                arch.append('reduce')
            else:
                arch.append('normal')
        return arch