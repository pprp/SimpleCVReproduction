"""
Modified from resnet_20s.py
For Cifar-10 dataset; 3 residual blocks.
TODOs:
  1. BN learnable params?
  2. version A?
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from utils.utils import accuracy
from utils.projection import project
from utils.beam_search import beam_decode
import math
from utils.utils import AverageMeter
import numpy as np
import pickle
from pdb import set_trace as br

VERSION = 'B' # 'A': downsampling for shortcut; 'B': 1x1 conv for shotcut
MAX_WIDTH =168
CANDIDATE_WIDTH = [16, 32, 64, 96]
OVERLAP = 1.0 # when OVERLAP=1, it reduces to partial
DETACH_PQ = True
AUX_BN = True

__all__ = ['resnet20_width', 'resnet32_width', 'resnet44_width', 'resnet56_width', \
    'resnet110_width']

class ArchMaster(nn.Module):
    def __init__(self, num_search_layers, n_ops, controller_type='LSTM', controller_hid=None,
                 controller_temperature=None, controller_tanh_constant=None,
                 controller_op_tanh_reduce=None, max_flops=4.1e7, lstm_num_layers=2, blockwise=False):
        super(ArchMaster, self).__init__()
        self.num_search_layers = num_search_layers
        self.n_ops = n_ops
        self.max_flops = max_flops
        self.controller_type = controller_type
        self.blockwise = blockwise

        if controller_type == 'ENAS':
            self.controller_hid = controller_hid           # 100 by default
            self.attention_hid = self.controller_hid
            self.temperature = controller_temperature
            self.tanh_constant = controller_tanh_constant
            self.op_tanh_reduce = controller_op_tanh_reduce
            self.lstm_num_layers = lstm_num_layers

            self.w_soft = nn.Linear(self.controller_hid, self.n_ops)
            self.lstm = nn.ModuleList()
            for i in range(self.lstm_num_layers):
                self.lstm.append(nn.LSTMCell(self.controller_hid, self.controller_hid))
            self.w_emb = nn.Embedding(self.n_ops + 1, self.controller_hid - self.num_search_layers - 1)
            # self.num_search_layers for one-hot layer encoding, 1 for left flops budget
            self.reset_parameters()
            self.tanh = nn.Tanh()
        else:
            assert False, "unsupported controller_type: %s" % self.controller_type

    def _init_nodes(self):
        prev_c = [torch.zeros(1, self.controller_hid).to(self.w_soft.weight.device) for _ in range(self.lstm_num_layers)]
        prev_h = [torch.zeros(1, self.controller_hid).to(self.w_soft.weight.device) for _ in range(self.lstm_num_layers)]
        # initialize the first two nodes
        inputs = torch.cat(tuple([self.w_emb(torch.LongTensor([self.n_ops]).to(self.w_soft.weight.device)),
                                  torch.zeros((1, self.num_search_layers)).to(self.w_soft.weight.device),
                                  torch.ones((1,1)).to(self.w_soft.weight.device)]), dim=-1)
        # print('inputs now has length:', inputs.size())
        for node_id in range(2):
            next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
            prev_c, prev_h = next_c, next_h
        return inputs, prev_c, prev_h

    def stack_lstm(self, x, prev_c, prev_h):
        next_c, next_h = [], []
        for layer_id, (_c, _h) in enumerate(zip(prev_c, prev_h)):
            inputs = x if layer_id == 0 else next_h[-1]
            curr_c, curr_h = self.lstm[layer_id](inputs, (_c, _h))
            next_c.append(curr_c)
            next_h.append(curr_h)
        return next_c, next_h

    def reset_parameters(self):
        # params initialization
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.w_soft.bias.data.fill_(0)

    def _compute_flops(self, arch_tmp):
        if type(arch_tmp) == list:
            arch_tmp = torch.tensor(arch_tmp)
        full_archs = self.obtain_full_archs(arch_tmp)
        curr_layers = len(full_archs)
        if len(arch_tmp) == self.num_search_layers:
            # NOTE: only involve conv layer, no classfication layer
            assert curr_layers == len(self.table_flops), 'total layers does not match total network conv layers'
        flops_decode = 0
        for layer in range(curr_layers):
            # for flop in self.table_flops[layer]:
            if len(self.table_flops[layer]) == self.n_ops:
                assert layer == 0 or len(self.table_flops) - 1, 'only first and last layer use n_ops candidates'
                flops_decode += self.table_flops[layer][full_archs[layer]]
            else:
                flops_decode += self.table_flops[layer][full_archs[layer]*self.n_ops + full_archs[layer-1]]
        return flops_decode

    def obtain_full_archs(self, archs):
        # convert blockwise archs to that of length without blockwise
        if self.blockwise:
            full_archs = []
            for idx, arch in enumerate(archs):
                if idx == 0:
                    full_archs += [arch]
                else:
                    full_archs += [arch] * 2
            full_archs = torch.stack(full_archs)
        else:
            full_archs = archs
        return full_archs

    def beam_search(self, size=4):
        cand_seq, logits_seq, logP_accum, entropy_accum = beam_decode(self, topk=size)
        return cand_seq, logits_seq, logP_accum, entropy_accum

    def forward(self):
        if self.controller_type == 'ENAS':
            log_prob, entropy = 0, 0
            self.prev_archs = []
            self.logits_list = []
            self.flops_left = self.max_flops
            inputs, prev_c, prev_h = self._init_nodes()
            # arc_seq = [0] * (self.n_nodes * 4)
            for layer_idx in range(self.num_search_layers):
                # body
                next_c, next_h = self.stack_lstm(inputs, prev_c, prev_h)
                prev_c, prev_h = next_c, next_h
                logits = self.w_soft(next_h[-1]).view(-1)
                if self.temperature is not None:
                    logits /= self.temperature
                if self.tanh_constant is not None:
                    op_tanh = self.tanh_constant / self.op_tanh_reduce
                    logits = op_tanh * self.tanh(logits)
                if self.force_uniform:
                    probs = F.softmax(torch.zeros_like(logits), dim=-1)
                else:
                    probs = F.softmax(logits, dim=-1)
                log_probs = torch.log(probs)
                action = probs.multinomial(num_samples=1)
                self.logits_list.append(logits)
                self.prev_archs.append(action)
                curr_log_prob = log_probs.gather(0, action)[0]
                log_prob += curr_log_prob
                curr_ent = -(log_probs * probs).sum()
                entropy += curr_ent

                arch_curr = torch.cat(tuple(self.prev_archs))
                flops_curr = self._compute_flops(arch_curr)
                flops_left = ((self.max_flops - flops_curr) / self.max_flops).view((1, 1)).to(self.w_soft.weight.device)  # tensor
                layer_idx_onehot = torch.tensor(np.eye(self.num_search_layers)[layer_idx].reshape(1, -1).astype(np.float32)).to(self.w_soft.weight.device)
                inputs = torch.cat(tuple([self.w_emb(action), layer_idx_onehot, flops_left]), dim=-1)

            self.logits = torch.stack(tuple(self.logits_list))
            arch = torch.cat(tuple(self.prev_archs))
            assert arch.size(0) == self.num_search_layers, 'arch should be the size of num_search_layers!'
            return arch, log_prob, entropy

        else:
            assert False, "unsupported controller_type: %s" % self.controller_type

    @property
    def device(self):
        return self.w_soft.weight.device


class TProjection(nn.Module):
    def __init__(self, cin_p, cout_p, cin, cout, overlap=OVERLAP, candidate_width=CANDIDATE_WIDTH):
        super(TProjection, self).__init__()
        self.cin_p = cin_p
        self.cout_p = cout_p
        self.cin=cin
        self.cout=cout
        self.overlap = overlap
        self.candidate_width = candidate_width
        self.P = nn.Parameter(torch.Tensor(cout, cout_p))
        self.Q = nn.Parameter(torch.Tensor(cin, cin_p))
        self.reset_parameters()

    def reset_parameters(self):
        # print("Tprojection overlap ratio: %.4f" % self.overlap)
        self.Q = self._init_projection(self.Q)
        self.P = self._init_projection(self.P)

    def _init_projection(self, W):
        # assume that the overlap ratio makes all the candidate inside the max_chann.
        meta_c, curr_c = W.shape
        if meta_c == curr_c == 3:
            # deal with the first projection
            W.data = torch.eye(curr_c)
            return W

        init_W = torch.zeros_like(W)
        ind = self.candidate_width.index(curr_c)
        cum_c = 0
        if ind == 0:
            init_W[:curr_c,:] = torch.eye(curr_c)
        else:
            for id_p in range(ind):
                cum_c += int((1 - self.overlap) * self.candidate_width[id_p])
            init_W[cum_c:cum_c+curr_c, :] = torch.eye(curr_c)
        W.data = init_W
        W.data += torch.randn(W.shape) * 1e-3

        return W


    def forward(self, meta_weights):
        if DETACH_PQ:
            P = self.P.detach()
            Q = self.Q.detach()
        else:
            P, Q = self.P, self.Q
        projected_weights = project(meta_weights, P, Q)
        return projected_weights

    @property
    def device(self):
        return self.P.device


class ShortCutLayer(nn.Module):
    def __init__(self, cin, cout, stride):
        super(ShortCutLayer, self).__init__()
        # NOTE: use a 1x1 conv to match the out dim, even cin=cout
        # make sure this stride keeps consistent with the block stride
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(cout, affine=False)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out


class BasicBlock(nn.Module):
    # expansion = 1
    def __init__(self, args, in_planes, planes, stride=1, max_width=MAX_WIDTH, candidate_width=CANDIDATE_WIDTH, option=VERSION):
        super(BasicBlock, self).__init__()
        self.max_width = max_width
        self.candidate_width = candidate_width
        self.stride = stride
        # in_planes,planes all should be of width: max_width
        # we first try the case where the width of each layer in the one-shot model is kept the same.
        self.conv1_kernel = nn.Parameter(torch.Tensor(self.max_width, in_planes, 3, 3))
        self.conv2_kernel = nn.Parameter(torch.Tensor(planes, self.max_width, 3, 3))
        self.reset_parameters()
        self.args = args

        self.bn1s = nn.ModuleList()
        self.bn2s = nn.ModuleList()
        self.tprojection1s = nn.ModuleList()
        self.tprojection2s = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        self.aux_transform = nn.ModuleList()
        self.aux_classifiers = nn.ModuleList()

        for i, cand_out in enumerate(self.candidate_width):
            for j, cand_in in enumerate(self.candidate_width):
                self.bn1s.append(nn.BatchNorm2d(cand_out, affine=False))
                self.bn2s.append(nn.BatchNorm2d(cand_out, affine=False))

        for i, cand_out in enumerate(self.candidate_width):
            for j, cand_in in enumerate(self.candidate_width):
                self.tprojection1s.append(TProjection(cin_p=cand_in,
                                                      cout_p=cand_out,
                                                      cin=in_planes,
                                                      cout=self.max_width,
                                                      candidate_width=self.candidate_width,
                                                      overlap=self.args.overlap))
                self.tprojection2s.append(TProjection(cin_p=cand_in,
                                                      cout_p=cand_out,
                                                      cin=self.max_width,
                                                      cout=planes,
                                                      candidate_width=self.candidate_width,
                                                      overlap=self.args.overlap))
                # index: i * len(candidate_width) + j
            self.aux_transform.append(nn.Sequential(nn.BatchNorm2d(cand_out, affine=AUX_BN),\
                                                      nn.ReLU(),\
                                                      nn.AdaptiveAvgPool2d(1)))
            self.aux_classifiers.append(nn.Linear(cand_out, 10))

        if option == 'B':
            # option B: all outputs of previous block applied on the output channs
            for i, cand_out in enumerate(self.candidate_width):
                for j, cand_in in enumerate(self.candidate_width):
                    shortcut = ShortCutLayer(cand_in, cand_out, self.stride)
                    self.shortcuts.append(shortcut)
        else:
            # option A: an one by one matching, candidates with the same chann number are skip connected.
            # chann_num weights use the second layer's
            # However, there is no search in this case on the skip connection path.
            raise NotImplementedError

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv1_kernel, a=math.sqrt(5))
        init.kaiming_uniform_(self.conv2_kernel, a=math.sqrt(5))

    def forward(self, x, arch):
        # for enas implementation x is just
        if type(arch) == list:
            assert len(arch) == 3, \
                'arch of a basic block of the model is not preperly defined'
        else:
            assert list(arch.size()) == [3], \
                'arch of a basic block of the model is not preperly defined'
        # arch should be the indexes of each channel number.
        arch1 = arch[0]  # inputs arch
        arch2 = arch[1]  # 1st layer arch
        arch3 = arch[2]  # output arch
        # layer1 forward
        projected_kernel1 = self.tprojection1s[arch2 * len(self.candidate_width) + arch1](self.conv1_kernel)
        # change to F.conv2d form: (cout, cin, k, k)
        h = F.relu(self.bn1s[arch2 * len(self.candidate_width) + arch1](
            F.conv2d(x, projected_kernel1, stride=self.stride, padding=1)))

        projected_kernel2 = self.tprojection2s[arch3 * len(self.candidate_width) + arch2](self.conv2_kernel)
        # aux logits
        if self.args.use_aux:
           tmp_h = F.conv2d(h, projected_kernel2, stride=1, padding=1).detach()
           tmp_h = self.aux_transform[arch3](tmp_h)
           tmp_h = tmp_h.view(tmp_h.size(0), -1)
           aux_logits = self.aux_classifiers[arch3](tmp_h)
        else:
           aux_logits = torch.ones((h.size(0), 10)).to(h.device)
        # layer2 forward
        h = self.bn2s[arch3 * len(self.candidate_width) + arch2](\
            F.conv2d(h, projected_kernel2, stride=1, padding=1))
        h += self.shortcuts[arch3 * len(self.candidate_width) + arch1](x)
        output = F.relu(h)
        return output, aux_logits


    def orthogonal_regularization(self):
        def calc_loss_l2(A):
            tmp = (A.t().mm(A))**2
            loss_orthog = tmp.sum() - torch.diag(tmp).sum()

            norms = tmp.diag()
            loss_norm = torch.abs(norms - torch.ones_like(norms).to(norms.device)).sum()
            return loss_orthog, loss_norm

        def calc_loss_l1(A):
            tmp = (A.t().mm(A)).abs()

            norms = tmp.diag()
            loss_orthog = tmp.sum() - torch.diag(tmp).sum()
            loss_norm = torch.abs(norms - torch.ones_like(norms).to(norms.device)).sum()
            return loss_orthog, loss_norm

        calc_loss = calc_loss_l1 if self.args.ortho_type == 'l1' else calc_loss_l2

        # The first layer
        proj_p_list, proj_q_list = [], []
        for tproj in self.tprojection1s:
            proj_p_list.append(tproj.P)
            proj_q_list.append(tproj.Q)

        # divide list into n_ops groups
        proj_p_groups, proj_q_groups = [], []
        for i in range(len(self.candidate_width)):
            tmp_p_group = torch.cat(proj_p_list[i:len(self.candidate_width)**2:len(self.candidate_width)], dim=1) # K x N, N > K  (168, 200)
            proj_p_groups.append(tmp_p_group)
            # [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
            tmp_q_group = torch.cat(proj_q_list[i*len(self.candidate_width):(i+1)*len(self.candidate_width)], dim=1)
            proj_q_groups.append(tmp_q_group)
            # [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

        loss_p1, loss_q1 = 0, 0
        loss_p_n1, loss_q_n1 = 0, 0
        for p_group, q_group in zip(proj_p_groups, proj_q_groups):
            loss_p_orthog, loss_p_norm = calc_loss(p_group)
            loss_q_orthog, loss_q_norm = calc_loss(q_group)
            loss_p1 += loss_p_orthog
            loss_q1 += loss_q_orthog
            loss_p_n1 += loss_p_norm
            loss_q_n1 += loss_q_norm


        # The second layer
        proj_p_list, proj_q_list = [], []
        for tproj in self.tprojection2s:
            proj_p_list.append(tproj.P)
            proj_q_list.append(tproj.Q)

        # divide list into n_ops groups
        proj_p_groups, proj_q_groups = [], []
        for i in range(len(self.candidate_width)):
            tmp_p_group = torch.cat(proj_p_list[i:len(self.candidate_width)**2:len(self.candidate_width)], dim=1) # K x N, N > K  (168, 200)
            proj_p_groups.append(tmp_p_group)
            tmp_q_group = torch.cat(proj_q_list[i*len(self.candidate_width):(i+1)*len(self.candidate_width)], dim=1)
            proj_q_groups.append(tmp_q_group)

        loss_p2, loss_q2 = 0, 0
        loss_p_n2, loss_q_n2 = 0, 0
        for p_group, q_group in zip(proj_p_groups, proj_q_groups):
            loss_p_orthog, loss_p_norm = calc_loss(p_group)
            loss_q_orthog, loss_q_norm = calc_loss(q_group)
            loss_p2 += loss_p_orthog
            loss_q2 += loss_q_orthog
            loss_p_n2 += loss_p_norm
            loss_q_n2 += loss_q_norm

        return loss_p1 + loss_q1 + loss_p2 + loss_q2, \
               loss_p_n1 + loss_q_n1 + loss_p_n2 + loss_q_n2


class FirstBlock(nn.Module):
    def __init__(self, args, max_width=MAX_WIDTH, candidate_width=CANDIDATE_WIDTH):
        super(FirstBlock, self).__init__()
        self.args = args
        self.max_width = max_width
        self.candidate_width = candidate_width
        self.stride = 1

        self.conv0_kernel = nn.Parameter(torch.Tensor(self.max_width, 3, 3, 3))
        self.reset_parameters()
        self.tprojection0s = nn.ModuleList()
        self.bn0s = nn.ModuleList()

        for i, cand in enumerate(self.candidate_width):
            self.tprojection0s.append(TProjection(cin_p=3,
                                                  cout_p=cand,
                                                  cin=3,
                                                  cout=self.max_width,
                                                  candidate_width=self.candidate_width,
                                                  overlap=self.args.overlap))
            self.bn0s.append(nn.BatchNorm2d(cand, affine=False))

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv0_kernel, a=math.sqrt(5))

    def forward(self, x, arch):
        chann_idx = int(arch)
        projected_kernel0 = self.tprojection0s[chann_idx](self.conv0_kernel)
        h = F.relu(self.bn0s[chann_idx](F.conv2d(x, projected_kernel0, stride=self.stride, padding=1)))
        return h

    def orthogonal_regularization(self):
        proj_p_as, proj_q_as = [], []

        for tproj in self.tprojection0s:
            proj_p_as.extend(torch.split(tproj.P, 1, dim=1))
            proj_q_as.extend(torch.split(tproj.Q, 1, dim=1))
        proj_p_as_T = torch.stack(tuple(proj_p_as), dim=0).squeeze()
        proj_q_as_T = torch.stack(tuple(proj_q_as), dim=0).squeeze()

        if self.args.ortho_type == 'l1':
            orth_p = torch.abs(proj_p_as_T.mm(proj_p_as_T.transpose(1, 0)))
            orth_q = torch.abs(proj_q_as_T.mm(proj_q_as_T.transpose(1, 0)))
        elif self.args.ortho_type == 'l2':
            orth_p = (proj_p_as_T.mm(proj_p_as_T.transpose(1, 0))**2)
            orth_q = (proj_q_as_T.mm(proj_q_as_T.transpose(1, 0))**2)
        else:
            raise ValueError("unknown ortho type")

        orth_reg_p = orth_p.sum() - orth_p.trace()
        orth_reg_q = orth_q.sum() - orth_q.trace()

        norm_reg_p = torch.abs(orth_p.diag()-torch.ones(orth_p.size(0)).to(orth_p.device)).sum()
        norm_reg_q = torch.abs(orth_q.diag()-torch.ones(orth_q.size(0)).to(orth_q.device)).sum()

        return orth_reg_p+orth_reg_q, norm_reg_p+norm_reg_q


class ClassfierBlock(nn.Module):
    def __init__(self, args, max_width=MAX_WIDTH, candidate_width=CANDIDATE_WIDTH, num_classes=10, bias=True):
        super(ClassfierBlock, self).__init__()
        # should we use a single out put classification layer or ? yes, a single one.
        self.candidate_width = candidate_width
        self.Q_k = nn.ParameterList()
        self.avg_pools = nn.ModuleList()
        self.args = args

        self.kernel = nn.Parameter(torch.Tensor(num_classes, max_width))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_classes))
        else:
            self.register_parameter('bias', None)
        for i, cand in enumerate(self.candidate_width):
            self.Q_k.append(nn.Parameter(torch.Tensor(max_width, cand)))

        for i, cand in enumerate(self.candidate_width):
            self.avg_pools.append(nn.AdaptiveAvgPool2d(1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.kernel, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.kernel)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

        for i, cand in enumerate(self.candidate_width):
            init.kaiming_uniform_(self.Q_k[i], a=math.sqrt(5))

    def forward(self, x, arch):
        if DETACH_PQ:
            Q_k = self.Q_k[arch].detach()
        else:
            Q_k = self.Q_k[arch]
        projected_kernels = self.kernel.mm(Q_k)
        # projected_kernels = self.kernel.mm(self.Q_k[arch])
        h = self.avg_pools[arch](x)
        h = h.view(h.size(0), -1)
        out = F.linear(h, projected_kernels, bias=None)
        if self.bias is not None:
            logits = out + self.bias
        else:
            logits = out
        return logits

    def orthogonal_regularization(self):
        proj_q_as = []

        for q in self.Q_k:
            proj_q_as.extend(torch.split(q, 1, dim=1))

        proj_q_as_T = torch.stack(tuple(proj_q_as), dim=0).squeeze()

        if self.args.ortho_type == 'l1':
            orth_q = torch.abs(proj_q_as_T.mm(proj_q_as_T.transpose(1, 0)))
        elif self.args.ortho_type == 'l2':
            orth_q = (proj_q_as_T.mm(proj_q_as_T.transpose(1, 0)))**2
        else:
            raise ValueError("unknown ortho type")

        orth_reg_q = orth_q.sum() - orth_q.trace()
        norm_reg_q = torch.abs(orth_q.diag()-torch.ones(orth_q.size(0)).to(orth_q.device)).sum()
        return orth_reg_q, norm_reg_q


class ResNetChanSearch(nn.Module):
    def __init__(self, block, num_blocks, candidate_width=CANDIDATE_WIDTH, num_classes=10, max_width=MAX_WIDTH, args=None):
        super(ResNetChanSearch, self).__init__()
        self.num_blocks = num_blocks
        self.args = args
        if self.args.rank == 0:
            print("Overlap ratio: %.4f" % self.args.overlap)
        if candidate_width == CANDIDATE_WIDTH:
            self.candidate_width = candidate_width
        else:
            self.candidate_width = [int(v) for v in candidate_width.split(',')] # parse from args

        self.max_width = max_width
        self.entropy_coeff = args.entropy_coeff

        if self.args.blockwise:
            print("Using blockwise channel search")
            self.num_search_layers = sum(num_blocks) + 1
        else:
            self.num_search_layers = 2 * sum(num_blocks) + 1
        self.num_total_layers = sum(num_blocks) * 2 + 2 # all conv layers and the final classification layer
        self.arch_master = ArchMaster(num_search_layers=self.num_search_layers,
                                      n_ops=len(self.candidate_width),
                                      controller_type=args.controller_type,
                                      controller_hid=args.controller_hid,
                                      controller_temperature=args.controller_temperature,
                                      controller_tanh_constant=args.controller_tanh_constant,
                                      controller_op_tanh_reduce=args.controller_op_tanh_reduce,
                                      max_flops=self.args.max_flops,
                                      blockwise=self.args.blockwise)

        # self.arch_parameters = list(self.arch_master.parameters())

        self.layer0 = FirstBlock(self.args, max_width=self.max_width, candidate_width=self.candidate_width)
        self.layers1 = self._make_layer(block, num_blocks[0], stride=1)
        self.layers2 = self._make_layer(block, num_blocks[1], stride=2)
        self.layers3 = self._make_layer(block, num_blocks[2], stride=2)
        self.classifier = ClassfierBlock(self.args, max_width=self.max_width, candidate_width=self.candidate_width, num_classes=num_classes)

    def _make_layer(self, block, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()
        for idx, stride in enumerate(strides):
            layers.append(block(self.args, in_planes=self.max_width, planes=self.max_width, max_width=self.max_width, candidate_width=self.candidate_width, stride=stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        # use alphas as inputs so that alphas can be changed from outside.
        # The last dim of arch is fixed as the largest the maximum candidate value
        archs, archs_logP, archs_entropy = self.arch_master.forward()
        if self.args.blockwise:
            archs_tmp = self.extend_blockwise_archs(archs)
        else:
            archs_tmp = archs

        counts = 0
        aux_logits_list = []
        h = self.layer0(x, archs_tmp[0])
        for i, layer in enumerate(self.layers1):
            h, aux_logits = layer(h, archs_tmp[counts+2*i: counts+2*(i + 1) + 1])
            aux_logits_list += [aux_logits]
        counts += 2 * len(self.layers1)
        for i, layer in enumerate(self.layers2):
            h, aux_logits = layer(h, archs_tmp[counts+2*i: counts+2*(i + 1) + 1])
            aux_logits_list += [aux_logits]
        counts += 2 * len(self.layers2)
        for i, layer in enumerate(self.layers3):
            if i < len(self.layers3) - 1:
                h, aux_logits = layer(h, archs_tmp[counts+2*i: counts+2*(i + 1) + 1])
            else:
                h, aux_logits = layer(h, torch.cat(tuple([archs_tmp[counts+2*i: counts+2*(i + 1)],
                                              torch.tensor([len(self.candidate_width) - 1]).to(h.device)])))
            aux_logits_list += [aux_logits]
        counts += 2 * len(self.layers3)

        logits  = self.classifier(h, len(self.candidate_width) - 1)
        counts += 1
        assert counts == self.num_total_layers - 1, "counts index mismatch!"
        return logits, aux_logits_list, archs_logP, archs_entropy, archs

    def orthogonal_reg_loss(self):
        loss_orthognal, loss_norm = 0, 0
        l_orthg, l_norm = self.layer0.orthogonal_regularization()
        loss_orthognal += l_orthg
        loss_norm += l_norm
        for i, layer in enumerate(self.layers1):
            l_orthg, l_norm = layer.orthogonal_regularization()
            loss_orthognal += l_orthg
            loss_norm += l_norm
        for i, layer in enumerate(self.layers1):
            l_orthg, l_norm = layer.orthogonal_regularization()
            loss_orthognal += l_orthg
            loss_norm += l_norm
        for i, layer in enumerate(self.layers3):
            l_orthg, l_norm = layer.orthogonal_regularization()
            loss_orthognal += l_orthg
            loss_norm += l_norm
        l_orthg, l_norm = self.classifier.orthogonal_regularization()
        loss_orthognal += l_orthg
        loss_norm += l_norm
        return loss_orthognal, loss_norm


    def test_forward(self, x, archs_tmp):
        # test the accs of each cand channels
        if self.args.blockwise:
            archs_tmp = self.extend_blockwise_archs(archs_tmp)

        counts = 0
        aux_logits_list = []
        h = self.layer0(x, archs_tmp[0])
        for i, layer in enumerate(self.layers1):
            aux_logits_curr = []
            for j in range(len(self.candidate_width)):
                jh, aux_logits = layer(h, torch.cat((archs_tmp[counts+2*i: counts+2*(i + 1)], torch.tensor([j]).to(h.device))))
                aux_logits_curr.append(aux_logits)
                if j == archs_tmp[counts+2*(i+1)]:
                    h_next = jh
            aux_logits_list += [aux_logits_curr]
            h = h_next
        counts += 2 * len(self.layers1)

        for i, layer in enumerate(self.layers2):
            aux_logits_curr = []
            for j in range(len(self.candidate_width)):
                jh, aux_logits = layer(h, torch.cat((archs_tmp[counts+2*i: counts+2*(i + 1)], torch.tensor([j]).to(h.device))))
                aux_logits_curr.append(aux_logits)
                if j == archs_tmp[counts+2*(i+1)]:
                    h_next = jh
            aux_logits_list += [aux_logits_curr]
            h = h_next
        counts += 2 * len(self.layers2)

        for i, layer in enumerate(self.layers3):
            aux_logits_curr = []
            for j in range(len(self.candidate_width)):
                if i < len(self.layers3) - 1:
                    jh, aux_logits = layer(h, torch.cat((archs_tmp[counts+2*i: counts+2*(i + 1)], torch.tensor([j]).to(h.device))))
                else:
                    jh, aux_logits = layer(h, torch.cat(tuple([archs_tmp[counts+2*i: counts+2*(i + 1)],
                                                  torch.tensor([len(self.candidate_width) - 1]).to(h.device)])))
                aux_logits_curr.append(aux_logits)
                if j == archs_tmp[counts+2*(i+1)]:
                    h_next = jh
            aux_logits_list += [aux_logits_curr]
            h = h_next
        counts += 2 * len(self.layers3)

        logits  = self.classifier(h, len(self.candidate_width) - 1)
        counts += 1
        assert counts == self.num_total_layers - 1, "counts index mismatch!"
        return logits, aux_logits_list

    def test_cand_arch(self, test_loader):
        def updt_aux_accs(aux_logits_list, test_target, aux_top1s):
            for i, aux_logits in enumerate(aux_logits_list):
                for j, logits in enumerate(aux_logits):
                    acc = accuracy(logits, test_target)[0]
                    aux_top1s[i][j].update(acc.item(), test_target.size(0))

        # if self.args.controller_type = 'SAMPLE':
        self._device = list(self.arch_master.parameters())[0].device
        # self._device = self.arch_master.w_soft.weight.device
        top1 = AverageMeter('candidates_top1', ':3.3f')
        aux_top1s = [[AverageMeter('layer:%d-cand%d'%(l, c), ':3.3f') for c in self.candidate_width] \
            for l in range(sum(self.num_blocks))]
        # losses = AverageMeter('candidates_losses', ':3.3f')
        # determine the stem arch
        arch_cand, arch_logP, arch_entropy = self.arch_master.forward()
        self.logits = self.arch_master.logits.data
        arch_info, discrepancy = self.get_arch_info(self.logits)

        for step, (test_input, test_target) in enumerate(test_loader):
            test_input = test_input.to(self._device)
            test_target = test_target.to(self._device)
            n = test_input.size(0)
            logits, aux_logits_list = self.test_forward(test_input, arch_cand)
            acc = accuracy(logits, test_target)[0]
            updt_aux_accs(aux_logits_list, test_target, aux_top1s)
            top1.update(acc.item(), n)

        # change aux_top1s to list of list of float
        aux_top1s_avg = [[v.avg for v in x] for x in aux_top1s]
        return top1.avg, aux_top1s_avg, arch_cand, arch_logP, arch_entropy, arch_info, discrepancy

    def get_arch_info(self, logits):
        # must be called after self.test_cand_arch is called
        string = "for width, there are {:} attention probabilities.".format(self.num_search_layers)
        discrepancy = []
        with torch.no_grad():
            for i, att in enumerate(logits):
                prob = nn.functional.softmax(att, dim=0)
                prob = prob.cpu()
                selc = prob.argmax().item()
                prob = prob.tolist()
                prob = ['{:.3f}'.format(x) for x in prob]
                xstring = '{:03d}/{:03d}-th : {:}'.format(i, self.num_search_layers, ' '.join(prob))
                logt = ['{:.3f}'.format(x) for x in att.cpu().tolist()]
                xstring += '  ||  {:52s}'.format(' '.join(logt))
                prob = sorted([float(x) for x in prob])
                disc = prob[-1] - prob[-2]
                xstring += '  || dis={:.2f} || select={:}/{:}'.format(disc, selc, len(prob))
                discrepancy.append(disc)   # difference between the most likely cand and the second likely cand
                string += '\n{:}'.format(xstring)
        return string, discrepancy


    def _loss_arch(self, x, y, baseline=None):
        # _loss_arch use a new arch, for we are training a LSTM controller.
        logits, aux_logits_list, archs_logP, archs_entropy, arch_tmp = self.forward(x)
        acc = accuracy(logits, y)[0] / 100.0
        # TODO: consider aux logit acc into reward?
        if self.args.flops:
            flops = self.arch_master._compute_flops(arch_tmp)
            if flops <= self.args.max_flops:
                reward_raw = acc * ((flops / self.args.max_flops) ** self.args.flops_coeff[0])
            else:
                reward_raw = acc * ((flops / self.args.max_flops) ** self.args.flops_coeff[1])
        else:
            reward_raw = acc
        reward = reward_raw - baseline if baseline else reward_raw
        policy_loss = -archs_logP * reward - (self.entropy_coeff * archs_entropy)
        return policy_loss, reward_raw, archs_entropy

    def extend_blockwise_archs(self, archs):
        extended_archs = []
        for layer_id, arch in enumerate(archs):
            if layer_id == 0:
                extended_archs += [arch]
            else:
                extended_archs += [arch]*2
        extended_archs = torch.tensor(extended_archs).to(archs.device)
        return extended_archs

    def arch_parameters(self):
        return self.arch_master.parameters()

    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch' not in k:
                yield v

    def named_arch_parameters(self):
        return self.arch_master.named_parameters()


    def named_model_parameters(self):
        return [(k, v) for k, v in self.named_parameters() if 'arch' not in k]


    def named_projection_parameters(self):
        for k, v in self.named_model_parameters():
            if 'projection' in k or 'Q_k' in k:
                yield k, v

    def projection_parameters(self):
        for k, v in self.named_projection_parameters():
            yield v


    def named_meta_parameters(self):
        for k, v in self.named_model_parameters():
            if 'projection' not in k and 'Q_k' not in k and 'aux' not in k:
                yield k, v

    def meta_parameters(self):
        for k, v in self.named_meta_parameters():
            yield v


def resnet20_width(num_classes, candidate_width=CANDIDATE_WIDTH, max_width=MAX_WIDTH, args=None):
  return ResNetChanSearch(BasicBlock, \
      [3, 3, 3], \
      num_classes=num_classes, \
      candidate_width=candidate_width, \
      max_width=max_width, args=args)

def resnet32_width(num_classes, candidate_width=CANDIDATE_WIDTH, max_width=MAX_WIDTH):
  return ResNetChanSearch(BasicBlock, \
      [5, 5, 5], \
      num_classes=num_classes, \
      candidate_width=candidate_width, \
      max_width=max_width)

def resnet44_width(num_classes, candidate_width=CANDIDATE_WIDTH, max_width=MAX_WIDTH):
  return ResNetChanSearch(BasicBlock, \
      [7, 7, 7], \
      num_classes=num_classes, \
      candidate_width=candidate_width, \
      max_width=max_width)

def resnet56_width(num_classes, candidate_width=CANDIDATE_WIDTH, max_width=MAX_WIDTH, args=None):
  return ResNetChanSearch(BasicBlock, \
      [9, 9, 9], \
      num_classes=num_classes, \
      candidate_width=candidate_width, \
      max_width=max_width, args=args)

def resnet110_width(num_classes, candidate_width=CANDIDATE_WIDTH, max_width=MAX_WIDTH):
  return ResNetChanSearch(BasicBlock, \
      [18, 18, 18], \
      num_classes=num_classes, \
      candidate_width=candidate_width, \
      max_width=max_width)


