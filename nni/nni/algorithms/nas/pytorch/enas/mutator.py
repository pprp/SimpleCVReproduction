# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F

from nni.nas.pytorch.mutator import Mutator
from nni.nas.pytorch.mutables import LayerChoice, InputChoice, MutableScope


class StackedLSTMCell(nn.Module):
    def __init__(self, layers, size, bias):
        super().__init__()
        self.lstm_num_layers = layers
        self.lstm_modules = nn.ModuleList([nn.LSTMCell(size, size, bias=bias)
                                           for _ in range(self.lstm_num_layers)])

    def forward(self, inputs, hidden):
        prev_h, prev_c = hidden
        next_h, next_c = [], []
        for i, m in enumerate(self.lstm_modules):
            curr_h, curr_c = m(inputs, (prev_h[i], prev_c[i]))
            next_c.append(curr_c)
            next_h.append(curr_h)
            # current implementation only supports batch size equals 1,
            # but the algorithm does not necessarily have this limitation
            inputs = curr_h[-1].view(1, -1)
        return next_h, next_c


class EnasMutator(Mutator):
    """
    A mutator that mutates the graph with RL.
    Parameters
    ----------
    model : nn.Module
        PyTorch model.
    lstm_size : int
        Controller LSTM hidden units.
    lstm_num_layers : int
        Number of layers for stacked LSTM.
    tanh_constant : float
        Logits will be equal to ``tanh_constant * tanh(logits)``. Don't use ``tanh`` if this value is ``None``.
    cell_exit_extra_step : bool
        If true, RL controller will perform an extra step at the exit of each MutableScope, dump the hidden state
        and mark it as the hidden state of this MutableScope. This is to align with the original implementation of paper.
    skip_target : float
        Target probability that skipconnect will appear.
    temperature : float
        Temperature constant that divides the logits.
    branch_bias : float
        Manual bias applied to make some operations more likely to be chosen.
        Currently this is implemented with a hardcoded match rule that aligns with original repo.
        If a mutable has a ``reduce`` in its key, all its op choices
        that contains `conv` in their typename will receive a bias of ``+self.branch_bias`` initially; while others
        receive a bias of ``-self.branch_bias``.
    entropy_reduction : str
        Can be one of ``sum`` and ``mean``. How the entropy of multi-input-choice is reduced.
    """

    def __init__(self, model, lstm_size=64, lstm_num_layers=1, tanh_constant=1.5, cell_exit_extra_step=False,
                 skip_target=0.4, temperature=None, branch_bias=0.25, entropy_reduction="sum"):
        super().__init__(model)
        self.lstm_size = lstm_size
        self.lstm_num_layers = lstm_num_layers
        self.tanh_constant = tanh_constant
        self.temperature = temperature
        self.cell_exit_extra_step = cell_exit_extra_step
        self.skip_target = skip_target
        self.branch_bias = branch_bias

        self.lstm = StackedLSTMCell(self.lstm_num_layers, self.lstm_size, False)

        self.attn_anchor = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.attn_query = nn.Linear(self.lstm_size, self.lstm_size, bias=False)
        self.v_attn = nn.Linear(self.lstm_size, 1, bias=False)
        
        self.g_emb = nn.Parameter(torch.randn(1, self.lstm_size) * 0.1) # 随机的初始化向量作为输入
        self.skip_targets = nn.Parameter(torch.tensor([1.0 - self.skip_target, self.skip_target]), requires_grad=False)  # pylint: disable=not-callable
        
        assert entropy_reduction in ["sum", "mean"], "Entropy reduction must be one of sum and mean."
        
        self.entropy_reduction = torch.sum if entropy_reduction == "sum" else torch.mean 
        # 加和或者求平均值

        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction="none")
        self.bias_dict = nn.ParameterDict()
        # key : bias
        # 人工施加的一些偏好，能够让模型偏向于某个选择

        self.max_layer_choice = 0
        for mutable in self.mutables:
            if isinstance(mutable, LayerChoice): # 如果是层选择
                if self.max_layer_choice == 0:
                    self.max_layer_choice = len(mutable)
                    # 层选择的最大个数就是 mutable 可变对象的长度

                assert self.max_layer_choice == len(mutable), \
                    "ENAS mutator requires all layer choice have the same number of candidates."

                # We are judging by keys and module types to add biases to layer choices. Needs refactor.
                if "reduce" in mutable.key:
                    def is_conv(choice):
                        # 判断choice是否是卷积层，给卷积层赋予更高的权重
                        return "conv" in str(type(choice)).lower()
                    bias = torch.tensor([self.branch_bias if is_conv(choice) else -self.branch_bias  # pylint: disable=not-callable
                                         for choice in mutable])
                    
                    # 如果是卷积层，bias提高，如果是其他层，bias降低选择概率。
                    self.bias_dict[mutable.key] = nn.Parameter(bias, requires_grad=False)

        self.embedding = nn.Embedding(self.max_layer_choice + 1, self.lstm_size)
        # 根据选择对应id，返回lstm的hidden state
        self.soft = nn.Linear(self.lstm_size, self.max_layer_choice, bias=False) 
        # 根据hidden state得到其中一个选择

    def sample_search(self):
        # 初始化的时候，在reset函数中进行调用
        self._initialize() # 初始化参数
        self._sample(self.mutables) # 找到可变层、突变层，进行采样
        return self._choices

    def sample_final(self):
        return self.sample_search()

    def _sample(self, tree):
        # 从mutable对象中提供的选项进行采样
        mutable = tree.mutable
        if isinstance(mutable, LayerChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_layer_choice(mutable) 
        # 将key添加到_choices这个字典中，value是选择的层，对应one_hot向量
        elif isinstance(mutable, InputChoice) and mutable.key not in self._choices:
            self._choices[mutable.key] = self._sample_input_choice(mutable)
        # 如果是连接，选择输入的层
        for child in tree.children:
            self._sample(child)
        if isinstance(mutable, MutableScope) and mutable.key not in self._anchors_hid:
            if self.cell_exit_extra_step:
                self._lstm_next_step()
            self._mark_anchor(mutable.key)

    def _initialize(self):
        # 初始化各个参数
        self._choices = dict()
        self._anchors_hid = dict()
        self._inputs = self.g_emb.data # 这是随机初始化的一个tensor，用于作为lstm的输入

        # lstm的参数
        self._c = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        self._h = [torch.zeros((1, self.lstm_size),
                               dtype=self._inputs.dtype,
                               device=self._inputs.device) for _ in range(self.lstm_num_layers)]
        
        self.sample_log_prob = 0
        self.sample_entropy = 0
        self.sample_skip_penalty = 0

    def _lstm_next_step(self):
        self._h, self._c = self.lstm(self._inputs, (self._h, self._c))
        # h : hidden state
        # c : cell state


    def _mark_anchor(self, key):
        # 锚 估计是用于定位连接位置的
        self._anchors_hid[key] = self._h[-1]

    def _sample_layer_choice(self, mutable):
        # 选择 某个层 只需要选一个就可以了
        self._lstm_next_step() # 让_inputs在lstm中进行一次前向传播

        logit = self.soft(self._h[-1]) # linear 从隐藏层embedd得到可选的层的逻辑评分

        if self.temperature is not None:
            logit /= self.temperature # 一个常量

        if self.tanh_constant is not None:
            # tanh_constant * tanh(logits) 用tanh再激活一次（可选）
            logit = self.tanh_constant * torch.tanh(logit)

        if mutable.key in self.bias_dict:
            logit += self.bias_dict[mutable.key] 
            # 对卷积层进行了偏好处理，如果是卷积层，那就在对应的值加上一个0.25，增大被选中的概率
        
        # softmax, view(-1), 
        branch_id = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1) 
        # 依据概率来选下角标，如果数量不为1，选择的多个中没有重复的
         
        log_prob = self.cross_entropy_loss(logit, branch_id) # 交叉熵损失函数

        self.sample_log_prob += self.entropy_reduction(log_prob) # 求和或者求平均
        
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        
        self.sample_entropy += self.entropy_reduction(entropy)

        self._inputs = self.embedding(branch_id) # 得到对应id的embedding, 

        return F.one_hot(branch_id, num_classes=self.max_layer_choice).bool().view(-1) # 将选择变成one_hot向量

    def _sample_input_choice(self, mutable):
        # 选择某个输入，即连接方式
        query, anchors = [], []
        
        for label in mutable.choose_from:
            if label not in self._anchors_hid:
                self._lstm_next_step()
                self._mark_anchor(label)  # empty loop, fill not found
            query.append(self.attn_anchor(self._anchors_hid[label]))
            anchors.append(self._anchors_hid[label])
        query = torch.cat(query, 0)
        query = torch.tanh(query + self.attn_query(self._h[-1]))
        query = self.v_attn(query)
        if self.temperature is not None:
            query /= self.temperature
        if self.tanh_constant is not None:
            query = self.tanh_constant * torch.tanh(query)

        if mutable.n_chosen is None:
            logit = torch.cat([-query, query], 1)  # pylint: disable=invalid-unary-operand-type

            skip = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip_prob = torch.sigmoid(logit)
            kl = torch.sum(skip_prob * torch.log(skip_prob / self.skip_targets))
            self.sample_skip_penalty += kl
            log_prob = self.cross_entropy_loss(logit, skip)
            self._inputs = (torch.matmul(skip.float(), torch.cat(anchors, 0)) / (1. + torch.sum(skip))).unsqueeze(0)
        else:
            assert mutable.n_chosen == 1, "Input choice must select exactly one or any in ENAS."
            logit = query.view(1, -1)
            index = torch.multinomial(F.softmax(logit, dim=-1), 1).view(-1)
            skip = F.one_hot(index, num_classes=mutable.n_candidates).view(-1)
            log_prob = self.cross_entropy_loss(logit, index)
            self._inputs = anchors[index.item()]

        self.sample_log_prob += self.entropy_reduction(log_prob)
        entropy = (log_prob * torch.exp(-log_prob)).detach()  # pylint: disable=invalid-unary-operand-type
        self.sample_entropy += self.entropy_reduction(entropy)
        return skip.bool()