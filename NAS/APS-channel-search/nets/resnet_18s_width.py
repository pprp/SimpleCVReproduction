import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
import torch.nn.init as init
from utils.projection import project
from utils.beam_search import beam_decode
from utils.utils import AverageMeter, accuracy
from utils.dist_utils import *
import torch.distributed as dist
import math
import os
from pdb import set_trace as br
import numpy as np
import pickle

__all__ = ['resnet18_width', 'resnet50_width']

# NOTE: for ilsvrc12 pretrained models.
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

MAX_WIDTH = 224 # 80 # 224
BASE_WIDTH = [32, 48, 64, 80]
# BASE_WIDTH = [16,24,32,40,48,64]
MULTIPLIER = 2
OVERLAP = 1.0 # when OVERLAP=1, it reduces to partial
DETACH_PQ = True
AFFINE = False
MULTIPLY_ADDS = False

class ArchMaster(nn.Module):
    def __init__(self, num_search_layers, block_layer_num, num_blocks, n_ops, controller_type='ENAS',
                 controller_hid=None, controller_temperature=None, controller_tanh_constant=None,
                 controller_op_tanh_reduce=None, max_flops=None, lstm_num_layers=2,
                 blockwise=False):
        super(ArchMaster, self).__init__()
        self.num_search_layers = num_search_layers
        self.block_layer_num = block_layer_num
        self.num_blocks = num_blocks
        self.n_ops = n_ops
        self.max_flops = max_flops
        self.controller_type = controller_type
        self.force_uniform = None
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
        # inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.w_soft.weight.device))   # use the last embed vector
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
            if layer == 0:
                flops_decode += self.table_flops[layer][full_archs[layer]]
                # print(layer, full_archs[layer], self.table_flops[layer][full_archs[layer]])
            else:
                flops_decode += self.table_flops[layer][full_archs[layer] * self.n_ops + full_archs[layer - 1]]
                # print(layer, full_archs[layer] * self.n_ops + full_archs[layer - 1], self.table_flops[layer][full_archs[layer] * self.n_ops + full_archs[layer - 1]])

        return flops_decode

    def obtain_full_archs(self, archs):
        # convert blockwise archs to that of length without blockwise
        idx = 0
        if self.blockwise:
            full_archs = [archs[0]]
            idx += 1
            for num in self.num_blocks:
                for block_id in range(num+1):
                    # stop iteration until the current archs is used out.
                    if idx > len(archs) -1:
                        break

                    if block_id == 0:
                        full_archs += [archs[idx]]*(self.block_layer_num-1)
                    elif block_id == 1:
                        block_out_idx = idx
                        full_archs += [archs[block_out_idx]]
                    else:
                        full_archs += [archs[idx]]*(self.block_layer_num-1) + [archs[block_out_idx]]
                    idx += 1
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
                flops_left = ((self.max_flops - flops_curr) / self.max_flops).view((1, 1)).to(self.device)  # tensor
                # print('now we have flops_left:', flops_left)
                layer_idx_onehot = torch.tensor(np.eye(self.num_search_layers)[layer_idx].reshape(1, -1).astype(np.float32)).to(self.w_soft.weight.device)
                # print('layer_idx_one has shape:', flops_left.size())
                inputs = torch.cat(tuple([self.w_emb(action), layer_idx_onehot, flops_left]), dim=-1)

            self.logits = torch.stack(tuple(self.logits_list))
            # self.logits_stats = 0.98 * self.logits_stats.data + 0.02 * self.logits.data
            arch = torch.cat(tuple(self.prev_archs))
            assert arch.size(0) == self.num_search_layers, 'arch should be the size of num_search_layers!'
            return arch, log_prob, entropy

        else:
            assert False, "unsupported controller_type: %s" % self.controller_type

    @property
    def device(self):
        return self.w_soft.weight.device


class TProjection(nn.Module):
    def __init__(self, cin_p, cout_p, cin, cout, overlap=OVERLAP,
                 candidate_width_p=BASE_WIDTH, candidate_width_q=BASE_WIDTH):
        super(TProjection, self).__init__()
        self.cin_p = cin_p
        self.cout_p = cout_p
        self.cin=cin
        self.cout=cout
        self.overlap = overlap
        self.candidate_width_p = candidate_width_p
        self.candidate_width_q = candidate_width_q
        self.P = nn.Parameter(torch.Tensor(self.cout, self.cout_p))
        self.Q = nn.Parameter(torch.Tensor(self.cin, self.cin_p))
        self.reset_parameters()

    def __repr__(self):
        return 'nn.Module. TProjection. Cin: %d; Cout: %d' % (self.cin_p, self.cout_p)

    def reset_parameters(self):
        # print("Tprojection overlap ratio: %.4f" % self.overlap)
        self.Q = self._init_projection(self.Q, self.candidate_width_q) # in-channel
        self.P = self._init_projection(self.P, self.candidate_width_p) # out-channel

    def _init_projection(self, W, candidate_width):
        # assume that the overlap ratio makes all the candidate inside the max_chann.
        meta_c, curr_c = W.shape
        if meta_c == curr_c == 3:
            W.data = torch.eye(curr_c) # do nothing for the first projection
            return W

        init_W = torch.zeros_like(W)
        ind = candidate_width.index(curr_c)
        cum_c = 0
        if ind == 0:
            init_W[:curr_c,:] = torch.eye(curr_c)
        else:
            for id_p in range(ind):
                cum_c += int((1 - self.overlap) * candidate_width[id_p])
            init_W[cum_c:cum_c+curr_c, :] = torch.eye(curr_c)
        W.data = init_W
        W.data += torch.randn(W.shape) * 1e-2
        return W

    def forward(self, meta_weights):
        if DETACH_PQ:
            P = self.P.detach()
            Q = self.Q.detach()
        else:
            P, Q = self.P, self.Q
        projected_weights = project(meta_weights, P, Q)
        return projected_weights

    def __repr__(self):
        return 'nn.Module. TProjection. Cin: %d; Cout: %d' % (self.cin_p, self.cout_p)

    @property
    def device(self):
        return self.P.device


class ShortCutLayer(nn.Module):
    def __init__(self, cin, cout, stride):
        super(ShortCutLayer, self).__init__()
        # NOTE: use a 1x1 conv to match the out dim, even cin=cout
        # make sure this stride keeps consistent with the block stride
        self.cin, self.cout = cin, cout
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(cout, affine=AFFINE)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out

    def __repr__(self):
        return "nn.Module: ShortCutLayer. Cin: %d; Cout: %d" % (self.cin, self.cout)


class ProjConv2d(nn.Conv2d):
    def __init__(self, meta_weights, cin_p, cout_p, cin, cout, kernel_size, \
                 candidate_width_p, candidate_width_q, overlap, layer=None, **kwargs):
        """
        Args:
          is_expand: a bool, indicating whether this conv needs to be expande (x4);
          candidate_width_p: list of candidate width for P;
          candidate_width_q: list of candidate width for Q;
        """

        super(ProjConv2d, self).__init__(cin_p, cout_p, kernel_size, **kwargs)
        self.tprojection = TProjection(cin_p, cout_p, cin, cout, \
            candidate_width_p=candidate_width_p, \
            candidate_width_q=candidate_width_q, \
            overlap=overlap)  # projection params initialized inside

        self.weight = None # remove the weights since we dynamically compute the projected weights in forward()
        self.layer = layer
        self.meta_weights = meta_weights

    def forward(self, x):
        self.proj_weight = self.tprojection(self.meta_weights)
        return F.conv2d(x, self.proj_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class ProjLinear(nn.Linear):
    def __init__(self, meta_weights, fin_p, fout_p, fin, candidate_width, layer=None, overlap=OVERLAP, **kwargs):
        super(ProjLinear, self).__init__(fin_p, fout_p, **kwargs)
        self.weight = None # over-ride weights to None
        self.meta_weights = meta_weights
        self.candidate_width = candidate_width
        self.fin, self.fin_p = fin, fin_p
        self.Q_k = nn.Parameter(torch.Tensor(fin, fin_p))
        self.overlap = overlap
        self.reset_Q()
        self.layer = layer

    def reset_Q(self):
        # NOTE: initialize Q in a similar way to t Projection
        # init.kaiming_uniform_(self.Q_k, a=math.sqrt(5))
        ind = self.candidate_width.index(self.fin_p)
        init_Q = torch.zeros_like(self.Q_k)
        cum_c = 0
        if ind == 0:
            init_Q[:self.fin_p,:] = torch.eye(self.fin_p)
        else:
            for id_p in range(ind):
                cum_c += int((1 - self.overlap) * self.candidate_width[id_p])
            init_Q[cum_c:cum_c+self.fin_p] = torch.eye(self.fin_p)
        self.Q_k.data = init_Q

    def forward(self, x):
        Q_k = self.Q_k.detach() if DETACH_PQ else self.Q_k
        self.proj_weight = self.meta_weights.mm(Q_k)
        return F.linear(x, self.proj_weight, self.bias)


class BasicBlock(nn.Module):

    def __init__(self, args, in_planes, planes, stride=1, \
                 candidate_width=BASE_WIDTH, multiplier=MULTIPLIER, layer_idx=None):
        """
        Args:
          args: block configuration;
          in_planes: an int, in width max; in_planes = 0.5 * planes if stride == 2
          planes: an int, out width max;
          candidate_width: a list of int, the candidate width, remain consistent for both conv;

        NOTE: inside the block, we should properly feed the correct cin_p,
              cout_p and candidate_width to ProjConv2d and TProjection.
        """
        super(BasicBlock, self).__init__()

        if stride == 2:
            assert in_planes == int(planes / multiplier)
        else:
            assert in_planes == planes

        self.args = args
        self.candidate_width = candidate_width
        self.num_cand = len(candidate_width)
        self.stride = stride
        self.multiplier = multiplier
        self.layer_idx = layer_idx if layer_idx else -2

        self.conv1_kernel = nn.Parameter(torch.Tensor(planes, in_planes, 3, 3))
        self.conv2_kernel = nn.Parameter(torch.Tensor(planes, planes, 3, 3))
        self.reset_parameters()

        self.conv1s, self.conv2s = nn.ModuleList(), nn.ModuleList()
        self.bn1s = nn.ModuleList()
        self.bn2s = nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        for i, cand_out in enumerate(self.candidate_width):
            for j, cand_in in enumerate(self.candidate_width):
                self.bn1s.append(nn.BatchNorm2d(cand_out, affine=AFFINE))
                self.bn2s.append(nn.BatchNorm2d(cand_out, affine=AFFINE))

        candidate_width_1q = [int(v/self.multiplier) for v in self.candidate_width] \
            if stride == 2 else self.candidate_width

        for i, cand_out in enumerate(self.candidate_width):
            for j, cand_in in enumerate(self.candidate_width):
                cand_in_1 = cand_in if stride == 1 else int(cand_in/self.multiplier)
                self.conv1s.append(ProjConv2d(self.conv1_kernel,\
                                              cin_p=cand_in_1, \
                                              cout_p=cand_out, \
                                              cin=in_planes, \
                                              cout=planes, \
                                              kernel_size=3, \
                                              candidate_width_p=self.candidate_width, \
                                              candidate_width_q=candidate_width_1q, \
                                              overlap=self.args.overlap, \
                                              layer=self.layer_idx, \
                                              stride=stride, \
                                              bias=False, \
                                              padding=1))

                self.conv2s.append(ProjConv2d(self.conv2_kernel,\
                                              cin_p=cand_in, \
                                              cout_p=cand_out, \
                                              cin=planes, \
                                              cout=planes, \
                                              kernel_size=3, \
                                              candidate_width_p=self.candidate_width, \
                                              candidate_width_q=self.candidate_width, \
                                              overlap=self.args.overlap, \
                                              layer=self.layer_idx+1, \
                                              stride=1, \
                                              bias=False, \
                                              padding=1))

        # add shortcut layers. 1x1 pointwise alignment
        for i, cand_out in enumerate(self.candidate_width):
            for j, cand_in in enumerate(candidate_width_1q):
                shortcut = ShortCutLayer(cand_in, cand_out, self.stride)
                self.shortcuts.append(shortcut)

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv1_kernel, a=math.sqrt(5))
        init.kaiming_uniform_(self.conv2_kernel, a=math.sqrt(5))

    def forward(self, x, archs):
        archs = archs.int().tolist()
        list_fn = lambda x: [x] if type(x)==int else x
        arch1 = list_fn(archs[0])  # inputs chan
        arch2 = list_fn(archs[1])  # 1st layer chan
        arch3 = list_fn(archs[2])  # 2nd layer chan
        shots = len(arch1)
        h_ws = []

        # layer1 forward
        for i, cand_out in enumerate(arch2):
            h_wis = []
            for j, cand_in in enumerate(arch1):
                ind = cand_out * self.num_cand + cand_in
                h = self.conv1s[ind](x[j])
                h = self.bn1s[ind](h)
                h_wis.append(h)
            h_wi_aggr = sum(h_wis)/shots
            h_wi_aggr = F.relu(h_wi_aggr)     # aggregation before activation
            h_ws.append(h_wi_aggr)

        # layer2 forward
        outputs = []
        for i, cand_out in enumerate(arch3):
            h_wis = []
            for j, (cand_in, cand_in_prev) in enumerate(zip(arch2, arch1)):
                ind = cand_out * self.num_cand + cand_in
                h = self.conv2s[ind](h_ws[j])
                h = self.bn2s[ind](h)

                ind_prev = cand_out * self.num_cand + cand_in_prev
                h += self.shortcuts[ind_prev](x[j])  # shortcuts moved inside aggregation
                h_wis.append(h)     # aggregation before activation

            # NOTE: alpha aggregated over in-channels
            h_wi_aggr = sum(h_wis)/shots
            h_wi_aggr = F.relu(h_wi_aggr)
            outputs.append(h_wi_aggr)

        return outputs

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
        for conv in self.conv1s:
            proj_p_list.append(conv.tprojection.P)
            proj_q_list.append(conv.tprojection.Q)

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
        for conv in self.conv2s:
            proj_p_list.append(conv.tprojection.P)
            proj_q_list.append(conv.tprojection.Q)

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

class Bottleneck(nn.Module):

    def __init__(self, args, in_planes, planes, stride=1, candidate_width=BASE_WIDTH, \
               multiplier=MULTIPLIER, layer_idx=None):
      """ NOTE: in_planes: in width max; planes: mid width max;
          in_planes = self.expansion * planes / 2 if stride == 2.
      Args:
        args: block configurations
        in_planes: an int, block input max channel width;
        planes: an int, block mid conv layer max channel width;
        candidate_width: a list of int, candidate width for the MID CONV of this block,
                         need to be modified for the first/last conv;
        multiplier: a float, the channel width multiplier for search;
        layer_idx: an int, the layer index;

      NOTE: inside the block, we should properly feed the correct cin_p,
            cout_p and candidate_width to ProjConv2d and TProjection.
      """

      super(Bottleneck, self).__init__()

      self.expansion = 4 # this is to follow bottleneck structure
      self.args = args
      self.multiplier = multiplier

      if stride == 2 and layer_idx != 1:
          assert in_planes == self.expansion * int(planes/self.multiplier) # divide multiplier since feature map size is strided by 2.
      elif layer_idx == 1:
          assert in_planes == planes # input from the first block.
      else:
          assert in_planes == planes * self.expansion
      out_planes = self.expansion * planes

      self.candidate_width = candidate_width
      self.num_cand = len(candidate_width)
      self.stride = stride
      self.layer_idx = layer_idx if layer_idx else -3

      self.conv1_kernel = nn.Parameter(torch.Tensor(planes, in_planes, 1, 1))
      self.conv2_kernel = nn.Parameter(torch.Tensor(planes, planes, 3, 3))
      self.conv3_kernel = nn.Parameter(torch.Tensor(out_planes, planes, 1, 1))
      self.reset_parameters()

      self.conv1s, self.conv2s , self.conv3s = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
      self.bn1s, self.bn2s, self.bn3s = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
      self.shortcuts = nn.ModuleList()

      for i, cand_out in enumerate(self.candidate_width):
          for j, cand_in in enumerate(self.candidate_width):
              self.bn1s.append(nn.BatchNorm2d(cand_out, affine=AFFINE))
              self.bn2s.append(nn.BatchNorm2d(cand_out, affine=AFFINE))
              self.bn3s.append(nn.BatchNorm2d(self.expansion*cand_out, affine=AFFINE))

      # determine candidate_width for the first cin and last cout
      if stride == 1 and layer_idx != 1:
          candidate_width_1q = [int(v*self.expansion) for v in self.candidate_width]
      elif stride == 2 and layer_idx != 1:
          candidate_width_1q = [int(v/self.multiplier)*self.expansion for v in self.candidate_width]
      elif stride == 1 and layer_idx == 1:
          candidate_width_1q = self.candidate_width
      elif layer_idx is None:
          pass
      else:
          raise ValueError("Wrong config. stride: %d, layer_idx: %d" % (stride, layer_idx))
      candidate_width_3p = [int(v*self.expansion) for v in self.candidate_width]

      for i, cand_out in enumerate(self.candidate_width):
          for j, cand_in in enumerate(self.candidate_width):
              if layer_idx != 1:
                  cand_in_1 = cand_in*self.expansion if stride == 1 \
                      else int(cand_in/self.multiplier)*self.expansion
              else:
                  # equal to the conv1 output channels
                  cand_in_1 = cand_in

              self.conv1s.append(ProjConv2d(self.conv1_kernel,\
                                            cin_p=cand_in_1, \
                                            cout_p=cand_out, \
                                            cin=in_planes, \
                                            cout=planes, \
                                            kernel_size=1, \
                                            candidate_width_p=self.candidate_width, \
                                            candidate_width_q=candidate_width_1q, \
                                            overlap=self.args.overlap, \
                                            layer=self.layer_idx, \
                                            stride=1, \
                                            bias=False))

              self.conv2s.append(ProjConv2d(self.conv2_kernel,\
                                            cin_p=cand_in, \
                                            cout_p=cand_out, \
                                            cin=planes, \
                                            cout=planes, \
                                            kernel_size=3, \
                                            candidate_width_p=self.candidate_width, \
                                            candidate_width_q=self.candidate_width, \
                                            overlap=self.args.overlap, \
                                            layer=self.layer_idx+1, \
                                            stride=self.stride, \
                                            bias=False, \
                                            padding=1))

              cand_out_1 = cand_out * self.expansion
              self.conv3s.append(ProjConv2d(self.conv3_kernel, \
                                            cin_p=cand_in, \
                                            cout_p=cand_out_1, \
                                            cin=planes, \
                                            cout=out_planes, \
                                            kernel_size=1, \
                                            candidate_width_p=candidate_width_3p, \
                                            candidate_width_q=self.candidate_width, \
                                            overlap=self.args.overlap, \
                                            layer=self.layer_idx+2, \
                                            stride=1, \
                                            bias=False))

              shortcut = ShortCutLayer(cand_in_1, cand_out_1, self.stride)
              self.shortcuts.append(shortcut)

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv1_kernel, a=math.sqrt(5))
        init.kaiming_uniform_(self.conv2_kernel, a=math.sqrt(5))
        init.kaiming_uniform_(self.conv3_kernel, a=math.sqrt(5))

    def forward(self, x, archs):
        assert len(archs) == 4, 'error length for archs.'
        # The arch sample shoud be called outside.
        archs = archs.int().tolist()  # require arch to be list of list
        list_fn = lambda x: [x] if type(x)==int else x
        arch1 = list_fn(archs[0])  # inputs chan
        arch2 = list_fn(archs[1])  # 1st layer chan
        arch3 = list_fn(archs[2])  # 2nd layer chan
        arch4 = list_fn(archs[3])  # output chan
        shots = len(arch1)

        # layer1 forward
        h_1s = []
        for i, cand_out in enumerate(arch2):
            h_wis = []
            for j, cand_in in enumerate(arch1):
                ind = cand_out * self.num_cand + cand_in
                h = self.conv1s[ind](x[j])
                h = self.bn1s[ind](h)
                h_wis.append(h)
            h_wi_aggr = sum(h_wis)/shots
            h_wi_aggr = F.relu(h_wi_aggr)     # aggregation before activation
            h_1s.append(h_wi_aggr)

        # layer2 forward
        h_2s = []
        for i, cand_out in enumerate(arch3):
            h_wis = []
            for j, (cand_in, cand_in_prev) in enumerate(zip(arch2, arch1)):
                ind = cand_out * self.num_cand + cand_in
                h = self.conv2s[ind](h_1s[j])
                h = self.bn2s[ind](h)
                h_wis.append(h)     # aggregation before activation
            h_wi_aggr = sum(h_wis)/shots
            h_wi_aggr = F.relu(h_wi_aggr)     # aggregation before activation
            h_2s.append(h_wi_aggr)

        # layer3 forward
        outputs = []
        for i, cand_out in enumerate(arch4):
            h_wis = []
            for j, (cand_in, cand_in_prev) in enumerate(zip(arch3, arch1)):
                ind = cand_out * self.num_cand + cand_in
                h = self.conv3s[ind](h_2s[j])
                h = self.bn3s[ind](h)
                ind_prev = cand_out * self.num_cand + cand_in_prev
                h += self.shortcuts[ind_prev](x[j])  # shortcuts moved inside aggregation
                h_wis.append(h)     # aggregation before activation

            # NOTE: alpha aggregated over in-channels
            h_wi_aggr = sum(h_wis)/shots
            h_wi_aggr = F.relu(h_wi_aggr)
            outputs.append(h_wi_aggr)

        return outputs

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
        for conv in self.conv1s:
            proj_p_list.append(conv.tprojection.P)
            proj_q_list.append(conv.tprojection.Q)

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
        for conv in self.conv2s:
            proj_p_list.append(conv.tprojection.P)
            proj_q_list.append(conv.tprojection.Q)

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

        # The last layer
        proj_p_list, proj_q_list = [], []
        for conv in self.conv3s:
            proj_p_list.append(conv.tprojection.P)
            proj_q_list.append(conv.tprojection.Q)

        # divide list into n_ops groups
        proj_p_groups, proj_q_groups = [], []
        for i in range(len(self.candidate_width)):
            tmp_p_group = torch.cat(proj_p_list[i:len(self.candidate_width)**2:len(self.candidate_width)], dim=1) # K x N, N > K  (168, 200)
            proj_p_groups.append(tmp_p_group)
            tmp_q_group = torch.cat(proj_q_list[i*len(self.candidate_width):(i+1)*len(self.candidate_width)], dim=1)
            proj_q_groups.append(tmp_q_group)

        loss_p3, loss_q3 = 0, 0
        loss_p_n3, loss_q_n3 = 0, 0
        for p_group, q_group in zip(proj_p_groups, proj_q_groups):
            loss_p_orthog, loss_p_norm = calc_loss(p_group)
            loss_q_orthog, loss_q_norm = calc_loss(q_group)
            loss_p3 += loss_p_orthog
            loss_q3 += loss_q_orthog
            loss_p_n3 += loss_p_norm
            loss_q_n3 += loss_q_norm

        return loss_p1 + loss_q1 + loss_p2 + loss_q2 + loss_p3 + loss_q3, \
               loss_p_n1 + loss_q_n1 + loss_p_n2 + loss_q_n2 + loss_p_n3 + loss_q_n3



class FirstBlock(nn.Module):
    def __init__(self, args, planes=MAX_WIDTH, candidate_width=BASE_WIDTH, layer_idx=None):
        """
        Args:
          planes: an int, out width max;
          candidate_width: a list of int, the candidate width;

        NOTE: inside the block, we should properly feed the correct cin_p,
              cout_p and candidate_width to ProjConv2d and TProjection.
        """
        super(FirstBlock, self).__init__()
        self.args = args
        self.planes = planes
        self.candidate_width = candidate_width
        self.layer_idx = layer_idx

        self.conv0_kernel = nn.Parameter(torch.Tensor(self.planes, 3, 7, 7))
        self.reset_parameters()
        self.conv0s = nn.ModuleList()
        self.bn0s = nn.ModuleList()

        for i, cand in enumerate(self.candidate_width):
            self.conv0s.append(ProjConv2d(meta_weights=self.conv0_kernel, \
                                          cin_p=3, \
                                          cout_p=cand, \
                                          cin=3, \
                                          cout=self.planes, \
                                          kernel_size=7, \
                                          candidate_width_p=self.candidate_width, \
                                          candidate_width_q=self.candidate_width, \
                                          overlap=self.args.overlap, \
                                          layer=layer_idx, \
                                          stride=2, \
                                          bias=False, \
                                          padding=3))
            self.bn0s.append(nn.BatchNorm2d(cand, affine=AFFINE))
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv0_kernel, a=math.sqrt(5))

    def forward(self, x, arch):
        arch = arch.int().tolist()  # require arch to be list of list
        list_fn = lambda x: [x] if type(x)==int else x
        arch = list_fn(arch)
        outputs = []
        for i, cand in enumerate(arch):
            h = self.maxpool(self.relu(self.bn0s[cand](self.conv0s[cand](x))))
            outputs.append(h)
        # aggregate in the next layer
        return outputs

    def orthogonal_regularization(self):
        proj_p_as, proj_q_as = [], []
        for conv in self.conv0s:
            proj_p_as.extend(torch.split(conv.tprojection.P, 1, dim=1))
            proj_q_as.extend(torch.split(conv.tprojection.Q, 1, dim=1))

        proj_p_as_T = torch.stack(tuple(proj_p_as), dim=0).squeeze()
        proj_q_as_T = torch.stack(tuple(proj_q_as), dim=0).squeeze()
        # print('Shape of the a:', proj_p_as[0].size())   # (168, 1)
        # print('Shape of the proj_p_as_T is:', proj_p_as_T.size())

        if self.args.ortho_type == 'l1':
            orth_p = torch.abs(proj_p_as_T.mm(proj_p_as_T.transpose(1, 0)))
            orth_q = torch.abs(proj_q_as_T.mm(proj_q_as_T.transpose(1, 0)))
        elif self.args.ortho_type == 'l2':
            orth_p = (proj_p_as_T.mm(proj_p_as_T.transpose(1, 0)))**2
            orth_q = (proj_q_as_T.mm(proj_q_as_T.transpose(1, 0)))**2
        else:
            raise ValueError("unknown ortho type")

        orth_reg_p = orth_p.sum() - orth_p.trace()
        orth_reg_q = orth_q.sum() - orth_q.trace()

        norm_reg_p = torch.abs(orth_p.diag()-torch.ones(orth_p.size(0)).to(orth_p.device)).sum()
        norm_reg_q = torch.abs(orth_q.diag()-torch.ones(orth_q.size(0)).to(orth_q.device)).sum()

        return orth_reg_p+orth_reg_q, norm_reg_p+norm_reg_q


class ClassfierBlock(nn.Module):
    def __init__(self, args, max_width=MAX_WIDTH, candidate_width=BASE_WIDTH, num_classes=10, bias=True, layer_idx=None):
        super(ClassfierBlock, self).__init__()
        # should we use a single out put classification layer or ? yes, a single one.
        self.args = args
        self.candidate_width = candidate_width
        self.num_cand = len(candidate_width)
        self.max_width = max_width
        self.meta_weights = nn.Parameter(torch.Tensor(num_classes, self.max_width))
        self.linears = nn.ModuleList()
        self.bias = bias
        self.layer = layer_idx

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        for i, cand in enumerate(self.candidate_width):
            self.linears.append(ProjLinear(self.meta_weights, cand, num_classes, self.max_width, self.candidate_width, self.layer, overlap=self.args.overlap))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.meta_weights, a=math.sqrt(5))

    def forward(self, x, arch):
        arch = arch.int().tolist()
        list_fn = lambda x: [x] if type(x)==int else x
        arch = list_fn(arch)
        logits = []
        for i, __ in enumerate(arch):
            h = self.avgpool(x[i])
            h = h.view(h.size(0), -1)
            # NOTE: fix to use the last one
            out = self.linears[self.num_cand-1](h)
            logits.append(out)
        logits_aggr = sum(logits)
        return logits_aggr

    def orthogonal_regularization(self):
        proj_q_as = []
        for linear in self.linears:
            proj_q_as.extend(torch.split(linear.Q_k, 1, dim=1))

        proj_q_as_T = torch.stack(tuple(proj_q_as), dim=0).squeeze()

        if self.args.ortho_type == 'l1':
            orth_q = torch.abs(proj_q_as_T.mm(proj_q_as_T.transpose(1, 0)))
        elif self.args.ortho_type == 'l2':
            orth_q = (proj_q_as_T.mm(proj_q_as_T.transpose(1, 0)))**2
        else:
            raise ValueError("unknown ortho type")

        orth_reg_q = orth_q.sum() - orth_q.trace() # +
        norm_reg_q = torch.abs(orth_q.diag()-torch.ones(orth_q.size(0)).to(orth_q.device)).sum()
        return orth_reg_q, norm_reg_q


class ResNetChanWidth(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, \
                 base_width=BASE_WIDTH, base_max_width=MAX_WIDTH, \
                 multiplier=MULTIPLIER, args=None, zero_init_residual=False):
        super(ResNetChanWidth, self).__init__()

        self.num_blocks = num_blocks
        self.args = args
        if self.args.rank == 0:
            print("Overlap ratio: %.4f" % self.args.overlap)
        if base_width == BASE_WIDTH:
            self.base_width = base_width
        else:
            # passed in from outside
            self.base_width = [int(v) for v in base_width.split(',')]

        self.enlarge = 1 # the current accumulated mulptilied
        self.is_basic_block = True if block == BasicBlock else False
        self.expansion = 1 if self.is_basic_block else 4 # different from self.enlarge, this is to upsample in BottleNeck
        self.layer_idx = 0

        self.multiplier = multiplier
        self.base_max_width = base_max_width
        self.num_cand = len(self.base_width)
        self.entropy_coeff = args.entropy_coeff

        self.block_layer_num = 2 if self.is_basic_block else 3
        self.total_search_layers = self.block_layer_num * sum(self.num_blocks) + 1
        if self.args.blockwise:
            print("Using blockwise channel search, inc == outc")
            self.num_search_layers = 1 + sum(self.num_blocks) + 4 # 1: first conv, 4: each out channel in blocks
        else:
            self.num_search_layers = self.block_layer_num * sum(num_blocks) + 1

        self.arch_master = ArchMaster(num_search_layers=self.num_search_layers,
                                      block_layer_num=self.block_layer_num,
                                      num_blocks=self.num_blocks,
                                      n_ops=len(self.base_width),
                                      controller_type=args.controller_type,
                                      controller_hid=args.controller_hid,
                                      controller_temperature=args.controller_temperature,
                                      controller_tanh_constant=args.controller_tanh_constant,
                                      controller_op_tanh_reduce=args.controller_op_tanh_reduce,
                                      max_flops=self.args.max_flops,
                                      blockwise=self.args.blockwise)

        self.layer0 = FirstBlock(self.args, \
                                 planes=self.base_max_width, \
                                 candidate_width=self.base_width, \
                                 layer_idx = self.layer_idx)
        self.layer_idx += 1
        self.layers1 = self._make_layer(block, self.num_blocks[0], stride=1)
        self.layers2 = self._make_layer(block, self.num_blocks[1], stride=2)
        self.layers3 = self._make_layer(block, self.num_blocks[2], stride=2)
        self.layers4 = self._make_layer(block, self.num_blocks[3], stride=2)
        self.enlarge = self.enlarge / self.multiplier

        candidate_width = [int(v * self.enlarge)*self.expansion for v in self.base_width]
        max_width = int(self.base_max_width*self.enlarge) if self.is_basic_block \
            else int(self.base_max_width*self.enlarge)*self.expansion
        self.classifier = ClassfierBlock(self.args, \
                                         max_width=max_width, \
                                         candidate_width=candidate_width, \
                                         num_classes=num_classes, \
                                         layer_idx=self.layer_idx)

        self._init_model()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        # compute flops_table
        self.flops_table_dict = self.compute_flops_table()
        with open('./resnet18_flops.pkl', 'wb') as f:
            pickle.dump(self.flops_table_dict, f)
            print("Flops saved")

        # NOTE: remember to check the net and flops_dict
        with open('./resnet18_flops.pkl', 'rb') as f:
            print("Restoring flops_table.pkl")
            self.flops_table_dict = pickle.load(f)
        list_flop_tensor = [torch.tensor(v).to(self.device) for v in list(self.flops_table_dict.values())]
        assert len(list_flop_tensor) == self.total_search_layers
        self.arch_master.table_flops = list_flop_tensor
        print("Flops counting done.")

    def _make_layer(self, block, num_blocks, stride=1):
        # NOTE: in Bottleneck_Block, planes is mid_planes;
        # NOTE:
        # 1. here we only make in_planes and planes properly defined:
        #   consider both multiplier and expansion.
        # 2. For cin_p,cout_p, do it in the block.
        # 3. candidate_width is defined for the non-expanded conv (in
        #   bottleneck block. modify it for the first/last conv of the bottleneck block.

        strides = [stride] + [1]*(num_blocks-1)
        layers = nn.ModuleList()

        for stride in strides:
            if stride == 2:
                in_planes = int(self.base_max_width * self.enlarge / self.multiplier) * self.expansion
                planes = int(self.base_max_width * self.enlarge)
                candidate_width = [int(v*self.enlarge) for v in self.base_width]

            elif self.layer_idx == 1:
                in_planes = self.base_max_width
                planes = self.base_max_width
                candidate_width = self.base_width

            else:
                in_planes = int(self.base_max_width * self.enlarge) * self.expansion
                planes = int(self.base_max_width * self.enlarge)
                candidate_width = [int(v*self.enlarge) for v in self.base_width]

            layers.append(block(self.args, \
                                in_planes=in_planes, \
                                planes=planes, \
                                candidate_width=candidate_width, \
                                multiplier=self.multiplier, \
                                stride=stride, \
                                layer_idx=self.layer_idx))
            self.layer_idx += self.block_layer_num
        self.enlarge *= self.multiplier
        return nn.Sequential(*layers)

    def _init_model(self):
        for m in self.modules():
            if isinstance(m, ProjConv2d):
                # nn.init.kaiming_normal_(m.meta_weights, mode='fan_out', nonlinearity='relu')
                # NOTE: different branches may have different channel shape
                pass
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and AFFINE:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        archs_tmp, archs_logP, archs_entropy = self.arch_master.forward()
        if self.args.distributed:
            # make sure different GPU share the same arch
            dist.broadcast(archs_tmp, 0)
            # NOTE: do not broadcast archs_logP and archs_entropy, since
            # computational graph may be changed. Luckily we only update on
            # rank=0

        if self.args.blockwise:
            archs = self.extend_blockwise_archs(archs_tmp)
        else:
            archs = archs_tmp

        counts = 0
        h = self.layer0(x, archs[0]) # already max pooled
        for i, layer in enumerate(self.layers1):
            h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
        counts += self.block_layer_num * len(self.layers1)
        for i, layer in enumerate(self.layers2):
            h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
        counts += self.block_layer_num * len(self.layers2)
        for i, layer in enumerate(self.layers3):
            h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
        counts += self.block_layer_num * len(self.layers3)
        for i, layer in enumerate(self.layers4):
            if i < len(self.layers4) - 1:
                h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
            else:
                # NOTE: the last layer chooses the largest channel
                last_arch = torch.tensor([self.num_cand-1]).to(self.device)
                tmp_arch = torch.cat(tuple([archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1)], last_arch]), dim=0)
                h = layer(h, tmp_arch)

        counts += self.block_layer_num * len(self.layers4)
        logits = self.classifier(h, archs[counts])
        counts += 1
        assert counts == self.total_search_layers, "counts index mismatch!"
        return logits, archs_logP, archs_entropy, archs_tmp

    def compute_flops_table(self, input_res=224, multiply_adds=MULTIPLY_ADDS):
        print("Computing flops table...")

        # obtain the proj_conv and feature maps
        layer_flop_dict={} # a dict containing the list of flops in that layer
        def proj_conv_hook(self, input, output):
            if self.layer not in layer_flop_dict.keys():
                layer_flop_dict[self.layer] = []

            _, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output[0].size()

            kernel_ops = self.kernel_size[0] * self.kernel_size[1] * (self.in_channels / self.groups)
            bias_ops = 1 if self.bias is not None else 0

            params = output_channels * (kernel_ops + bias_ops)
            flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * 1
            layer_flop_dict[self.layer].append(flops)

        # whether blockwise or not, use full archs
        flop_archs = torch.tensor([list(range(self.num_cand))]*(self.total_search_layers)).to(self.device)
        flop_handles = []
        for m in self.modules():
            if isinstance(m, ProjConv2d):
                handle = m.register_forward_hook(proj_conv_hook)
                flop_handles.append(handle)

        input_ = torch.rand(1, 3, input_res, input_res).to(self.device)
        input_.requires_grad = True
        out = self.test_forward(input_, flop_archs)
        for handle in flop_handles:
            handle.remove()
        return layer_flop_dict

    def orthogonal_reg_loss(self):
        loss_orthogonal, loss_norm = 0, 0
        loss_orthogonal += self.layer0.orthogonal_regularization()[0]
        loss_norm += self.layer0.orthogonal_regularization()[1]
        for i, layer in enumerate(self.layers1):
            loss_orthogonal += layer.orthogonal_regularization()[0]
            loss_norm += layer.orthogonal_regularization()[1]
        for i, layer in enumerate(self.layers2):
            loss_orthogonal += layer.orthogonal_regularization()[0]
            loss_norm += layer.orthogonal_regularization()[1]
        for i, layer in enumerate(self.layers3):
            loss_orthogonal += layer.orthogonal_regularization()[0]
            loss_norm += layer.orthogonal_regularization()[1]
        for i, layer in enumerate(self.layers4):
            loss_orthogonal += layer.orthogonal_regularization()[0]
            loss_norm += layer.orthogonal_regularization()[1]
        loss_orthogonal += self.classifier.orthogonal_regularization()[0]
        loss_norm += self.classifier.orthogonal_regularization()[1]
        return loss_orthogonal, loss_norm

    def updt_orthogonal_reg_loss(self, proj_opt):

        proj_opt.zero_grad()
        loss_proj = 0.
        ortho = self.layer0.orthogonal_regularization()[0] * self.args.orthg_weight
        ortho.backward()
        loss_proj += ortho.item()

        for i, layer in enumerate(self.layers1):
            ortho = layer.orthogonal_regularization()[0] * self.args.orthg_weight
            ortho.backward()
            loss_proj += ortho.item()

        for i, layer in enumerate(self.layers2):
            ortho = layer.orthogonal_regularization()[0] * self.args.orthg_weight
            ortho.backward()
            loss_proj += ortho.item()

        for i, layer in enumerate(self.layers3):
            ortho = layer.orthogonal_regularization()[0] * self.args.orthg_weight
            ortho.backward()
            loss_proj += ortho.item()

        for i, layer in enumerate(self.layers4):
            ortho = layer.orthogonal_regularization()[0] * self.args.orthg_weight
            ortho.backward()
            loss_proj += ortho.item()

        ortho = self.classifier.orthogonal_regularization()[0] * self.args.orthg_weight
        ortho.backward()
        loss_proj += ortho.item()
        # average gradients once and update, reduce communication
        if self.args.distributed:
            average_group_gradients(self.projection_parameters())
        proj_opt.step()
        return loss_proj

    def test_forward(self, x, archs):
        if self.args.blockwise and len(archs) < self.total_search_layers:
            # len(archs) < self.total_search_layers filters out flops_archs
            archs = self.extend_blockwise_archs(archs)
        elif len(archs) == self.total_search_layers:
            archs = archs
        else:
            raise ValueError("Wrong archs length in test_forward")

        counts = 0
        h = self.layer0(x, archs[0])
        for i, layer in enumerate(self.layers1):
            h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
        counts += self.block_layer_num * len(self.layers1)

        for i, layer in enumerate(self.layers2):
            h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
        counts += self.block_layer_num * len(self.layers2)

        for i, layer in enumerate(self.layers3):
            h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
        counts += self.block_layer_num * len(self.layers3)

        for i, layer in enumerate(self.layers4):
            if i < len(self.layers4) - 1:
                h = layer(h, archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1])
            else:
                last_arch = (self.num_cand-1) * (torch.ones_like(archs[-1:]).to(self.device))
                tmp_arch = torch.cat(tuple([archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1)], last_arch]))
                h = layer(h, tmp_arch)
            # h = layer(h, torch.cat(tuple([archs[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1)],
            #                               torch.tensor([self.num_cand])])))
            # h = layer(h, torch.cat(archs_tmp[counts+self.block_layer_num*i: counts+self.block_layer_num*(i + 1) + 1]))  # the last layer is not fixed.
        counts += self.block_layer_num * len(self.layers4)

        last_arch = last_arch.squeeze() # squeeze out the useless dim
        logits  = self.classifier(h, last_arch)
        counts += 1
        assert counts == self.total_search_layers, "counts index mismatch! counts: %d" % counts
        return logits

    def test_cand_arch(self, test_loader):
        top1 = AverageMeter('candidates_top1', ':3.3f')
        arch_cand, arch_logP, arch_entropy = self.arch_master.forward()

        self.logits = self.arch_master.logits.data
        if self.num_cand > 1:
            arch_info, discrepancy = self.get_arch_info(self.logits)
        else:
            print("Only one candidate for searching!")
            arch_info = ''
            discrepancy = -1.

        # NOTE: to speed up testing, we can only sample archs without evaluation
        for step, (test_input, test_target) in enumerate(test_loader):
            test_input = test_input.to(self.device)
            test_target = test_target.to(self.device)
            n = test_input.size(0)
            logits = self.test_forward(test_input, arch_cand)
            acc = accuracy(logits, test_target)[0]
            top1.update(acc.item(), n)

        return top1.avg, arch_cand, arch_logP, arch_entropy, arch_info, discrepancy

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

    def _loss_arch(self, archs_logP, reduced_acc1, archs_entropy, arch_tmp, baseline=None):
        # _loss_arch use the same arch for training LSTM controller.
        if self.args.rank == 0:
            # calculate reward and policy loss on rank 0 for single GPU update
            if self.args.flops:
                flops = self.arch_master._compute_flops(arch_tmp)
                if flops <= self.args.max_flops:
                    reward_raw = reduced_acc1 * ((flops / self.args.max_flops) ** self.args.flops_coeff[0])
                else:
                    reward_raw = reduced_acc1 * ((flops / self.args.max_flops) ** self.args.flops_coeff[1])
            else:
                reward_raw = reduced_acc1
            reward = reward_raw - baseline if baseline else reward_raw
            policy_loss = -archs_logP * reward - (self.entropy_coeff * archs_entropy)
        else:
            # initialize for parallel
            policy_loss = torch.tensor(0.).cuda()
            reward_raw = torch.tensor(0.).cuda()

        if self.args.distributed:
            dist.barrier()
            dist.broadcast(policy_loss, 0)  # broadcast just for consistent return
            dist.broadcast(reward_raw, 0)

        return policy_loss, reward_raw

    def extend_blockwise_archs(self, archs):
        extended_archs = [archs[0]] # add the first conv cand inside

        idx = 1
        block_out_idx = 0
        for num in self.num_blocks:
            # num + 1 because of inc and outc
            for block_id in range(num+1):
                if block_id == 0:
                    extended_archs += [archs[idx]]*(self.block_layer_num-1)
                elif block_id == 1:
                    block_out_idx = idx
                    extended_archs += [archs[idx]]
                else:
                    extended_archs += [archs[idx]]*(self.block_layer_num-1) + [archs[block_out_idx]]
                idx += 1
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

    def check_submodel(self, archs):
        def proj_conv_hook(self, input, output):
            print(self)

        def proj_linear_hook(self, input, output):
            print(self)

        submodel_handles = []
        for m in self.modules():
            if isinstance(m, ProjConv2d):
                handle = m.register_forward_hook(proj_conv_hook)
                submodel_handles.append(handle)
            if isinstance(m, ProjLinear):
                handle = m.register_forward_hook(proj_linear_hook)
                submodel_handles.append(handle)

        print("Checking submodel indexed by: ", archs)
        input_res = 224
        input_ = torch.rand(1, 3, input_res, input_res).to(self.device)
        input_.requires_grad = True
        out = self.test_forward(input_, archs)
        for handle in submodel_handles:
            handle.remove()
        pass

    @property
    def device(self):
        return self.layer0.conv0_kernel.device


def resnet18_width(num_classes, \
                    base_width=BASE_WIDTH, \
                    base_max_width=MAX_WIDTH, \
                    pretrained=False, \
                    args=None, \
                    **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetChanWidth(BasicBlock, \
                             [2,2,2,2], \
                             num_classes=num_classes, \
                             base_width=base_width, \
                             base_max_width=base_max_width, \
                             multiplier=args.multiplier, \
                             args=args, \
                             **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def resnet50_width(num_classes, \
                    base_width=BASE_WIDTH, \
                    base_max_width=MAX_WIDTH, \
                    pretrained=False, \
                    args=None, \
                    **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNetChanWidth(Bottleneck, \
                             [3,4,6,3], \
                             num_classes=num_classes, \
                             base_width=base_width, \
                             base_max_width=base_max_width, \
                             multiplier=args.multiplier, \
                             args=args, \
                             **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

