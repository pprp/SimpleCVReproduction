import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.distributed as dist
from utils.dist_utils import *
from utils.beam_search import beam_decode
import numpy as np
from utils.projection import project
from utils.utils import AverageMeter, accuracy
import math
import pickle
from pdb import set_trace as br
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url


__all__ = ['mobilenet_v2_width']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}

MAX_WIDTH = 32
BASE_WIDTH = [8,12,16,20]
MULTIPLIER = 1
OVERLAP = 1.0 # when OVERLAP=1, it reduces to partial
DETACH_PQ = True
AFFINE = False
MULTIPLY_ADDS = False


class ArchMaster(nn.Module):
    def __init__(self, num_search_layers, num_blocks, n_ops, controller_type='ENAS', controller_hid=None,
                 controller_temperature=None, controller_tanh_constant=None,
                 controller_op_tanh_reduce=None, max_flops=None, lstm_num_layers=2, blockwise=False):
        super(ArchMaster, self).__init__()
        self.num_search_layers = num_search_layers
        self.num_blocks = num_blocks
        self.extended_search_layers = (num_search_layers-7) * 2 + 1 if blockwise else self.num_search_layers
        assert self.extended_search_layers == 35, 'Wrong extended search layers'
        self.n_ops = n_ops
        self.max_flops = max_flops
        self.controller_type = controller_type
        self.force_uniform = None
        self.blockwise = blockwise

        if controller_type == 'ENAS':
            self.controller_hid = controller_hid  # 100 by default
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
        prev_c = [torch.zeros(1, self.controller_hid).to(self.w_soft.weight.device) for _ in
                  range(self.lstm_num_layers)]
        prev_h = [torch.zeros(1, self.controller_hid).to(self.w_soft.weight.device) for _ in
                  range(self.lstm_num_layers)]
        # initialize the first two nodes
        # inputs = self.w_emb(torch.LongTensor([self.n_ops]).to(self.w_soft.weight.device))   # use the last embed vector
        inputs = torch.cat(tuple([self.w_emb(torch.LongTensor([self.n_ops]).to(self.w_soft.weight.device)),
                                  torch.zeros((1, self.num_search_layers)).to(self.w_soft.weight.device),
                                  torch.ones((1, 1)).to(self.w_soft.weight.device)]), dim=-1)
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
            assert curr_layers + 1 == len(self.table_flops), 'total layers does not match total network layers'
        flops_decode = 0
        for layer in range(curr_layers):
            # for flop in self.table_flops[layer]:
            if len(self.table_flops[layer]) == self.n_ops:
                # layer = 0 or dw layers, or first pw layer when blockwise==True
                flops_decode += self.table_flops[layer][full_archs[layer]]
            else:
                flops_decode += self.table_flops[layer][full_archs[layer]*self.n_ops + full_archs[layer-1]]

        return flops_decode

    def obtain_full_archs(self, archs):
        # first convert archs with search_layers to full layers
        if self.blockwise:
            extend_archs = []
            dc_used = 0
            block_id = 0
            for layer_id, arch in enumerate(archs):
                if layer_id == 0:
                    extend_archs += [arch]*2
                    dc_used += 1
                elif layer_id == self.num_search_layers - 1:
                    # the last c, directly put in
                    extend_archs += [arch]
                elif dc_used == self.num_blocks[block_id]:
                    # arrive at ci==co
                    block_id += 1
                    dc_used = 0
                    cio = arch
                else:
                    # add dc
                    extend_archs += [arch, cio]
                    dc_used += 1
            extend_archs = torch.stack(extend_archs)
        else:
            extend_archs = archs

        # obtain the full arch
        curr_search_layers = len(extend_archs)
        full_archs = []
        for idx, arch in enumerate(extend_archs):
            if idx % 2 == 0 and idx < self.extended_search_layers - 2:
                full_archs.append(arch)
                full_archs.append(arch)
            else:
                full_archs.append(arch)
        full_archs = torch.stack(full_archs)
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
                layer_idx_onehot = torch.tensor(
                    np.eye(self.num_search_layers)[layer_idx].reshape(1, -1).astype(np.float32)).to(
                    self.w_soft.weight.device)
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
        # print("initialized P Q with %.4f" % self.overlap)
        meta_c, curr_c = W.shape
        if meta_c == curr_c == 3 or meta_c == curr_c == 1:
            # seperately deal with input/depthwise conv
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
        W.data += torch.randn(W.shape) * 1e-2    # 1e-2 makes it converge, 1e-3 cannot
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
    def __init__(self, cin, cout, stride, **kwargs):
        super(ShortCutLayer, self).__init__()
        # NOTE: use a 1x1 conv to match the out dim, even cin=cout
        # make sure this stride keeps consistent with the block stride
        self.cin, self.cout = cin, cout
        self.conv = nn.Conv2d(cin, cout, kernel_size=1, stride=stride, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(cout, affine=AFFINE)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out

    def __repr__(self):
        return "nn.Module: ShortCutLayer. Cin: %d; Cout: %d" % (self.cin, self.cout)


class ProjConv2d(nn.Conv2d):
    def __init__(self, meta_weights, cin_p, cout_p, cin, cout, kernel_size, \
                 candidate_width_p, candidate_width_q, overlap, layer=None, dw_groups=1, **kwargs):
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
        self.groups = dw_groups

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


class FirstBlock(nn.Module):
    def __init__(self, args, planes=MAX_WIDTH, candidate_width=BASE_WIDTH, layer_idx=None):
        """
        Args:
          args: block configuration
          planes: an int, out width max;
          candidate_width: a list of int, the candidate width of OUTPUT channels;

        NOTE: inside the block, we should properly feed the correct cin_p,
              cout_p and candidate_width to ProjConv2d and TProjection.
        """
        super(FirstBlock, self).__init__()
        self.args = args
        self.planes = planes
        self.candidate_width = candidate_width
        self.layer_idx = layer_idx

        self.conv0_kernel = nn.Parameter(torch.Tensor(self.planes, 3, 3, 3))
        self.reset_parameters()
        self.conv0s = nn.ModuleList()
        self.bn0s = nn.ModuleList()

        for i, cand in enumerate(self.candidate_width):
            self.conv0s.append(ProjConv2d(meta_weights=self.conv0_kernel, \
                                          cin_p=3, \
                                          cout_p=cand, \
                                          cin=3, \
                                          cout=self.planes, \
                                          kernel_size=3, \
                                          candidate_width_p=self.candidate_width, \
                                          candidate_width_q=self.candidate_width, \
                                          overlap=self.args.overlap, \
                                          layer=layer_idx, \
                                          stride=2, \
                                          bias=False, \
                                          padding=1))
            self.bn0s.append(nn.BatchNorm2d(cand, affine=AFFINE))
        self.relu = nn.ReLU6(inplace=True)
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv0_kernel, a=math.sqrt(5))

    def forward(self, x, arch):
        arch = arch.int().tolist()  # require arch to be list of list
        list_fn = lambda x: [x] if type(x)==int else x
        arch = list_fn(arch)
        outputs = []
        for i, cand in enumerate(arch):
            h = self.relu(self.bn0s[cand](self.conv0s[cand](x)))
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


class LastBlock(nn.Module):
    def __init__(self, args, inp, oup, candidates_in, candidates_out, layer_idx=None):
        """
        The last conv2d 1x1
        """
        super(LastBlock, self).__init__()
        self.args = args
        self.inp = inp
        self.oup = oup
        self.candidates_in = candidates_in
        self.candidates_out = candidates_out
        self.num_cand = len(candidates_in)
        self.layer_idx = layer_idx

        self.conv0_kernel = nn.Parameter(torch.Tensor(self.oup, self.inp, 1, 1))
        self.reset_parameters()
        self.conv0s = nn.ModuleList()
        self.bn0s = nn.ModuleList()

        for i, cand_out in enumerate(self.candidates_out):
            for j, cand_in in enumerate(self.candidates_in):
                self.conv0s.append(ProjConv2d(meta_weights=self.conv0_kernel, \
                                              cin_p=cand_in, \
                                              cout_p=cand_out, \
                                              cin=self.inp, \
                                              cout=self.oup, \
                                              kernel_size=1, \
                                              candidate_width_p=self.candidates_out, \
                                              candidate_width_q=self.candidates_in, \
                                              overlap=self.args.overlap, \
                                              layer=layer_idx, \
                                              stride=1, \
                                              bias=False))
                self.bn0s.append(nn.BatchNorm2d(cand_out, affine=AFFINE))
        self.relu = nn.ReLU6(inplace=True)

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv0_kernel, a=math.sqrt(5))

    def forward(self, x, arch):
        arch = arch.int().tolist()  # require arch to be list of list
        assert len(arch) == 2, 'error length for arch, should be 2'
        list_fn = lambda x: [x] if type(x)==int else x
        arch1 = list_fn(arch[0])
        arch2 = list_fn(arch[1])
        shots = len(arch1)

        outputs = []
        for i, cand_out in enumerate(arch2):
            h_wis = []
            for j, cand_in in enumerate(arch1):
                ind = cand_out * self.num_cand + cand_in
                h = self.relu(self.bn0s[ind](self.conv0s[ind](x[j])))
                h_wis.append(h)
            h_wi_aggr = sum(h_wis) / shots
            outputs.append(h_wi_aggr)
        return outputs

    def orthogonal_regularization(self):
        proj_p_as, proj_q_as = [], []
        for conv in self.conv0s:
            proj_p_as.extend(torch.split(conv.tprojection.P, 1, dim=1))
            proj_q_as.extend(torch.split(conv.tprojection.Q, 1, dim=1))

        proj_p_as_T = torch.stack(tuple(proj_p_as), dim=0).squeeze()
        proj_q_as_T = torch.stack(tuple(proj_q_as), dim=0).squeeze()

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
    def __init__(self, args, max_width=MAX_WIDTH, candidate_width=BASE_WIDTH, num_classes=1000, dropout=0.2, layer_idx=None):
        super(ClassfierBlock, self).__init__()
        # should we use a single out put classification layer or ? yes, a single one.
        self.args = args
        self.candidate_width = candidate_width
        self.num_cand = len(candidate_width)
        self.max_width = max_width
        self.meta_weights = nn.Parameter(torch.Tensor(num_classes, self.max_width))
        self.linears = nn.ModuleList()
        self.layer = layer_idx

        self.dropout = nn.Dropout(dropout)
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
        for i, cand in enumerate(arch):
            h = nn.functional.adaptive_avg_pool2d(x[i], 1).reshape(x[i].shape[0], -1)
            h = self.dropout(h)
            out = self.linears[cand](h)
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


class InvertedResidual(nn.Module):

    def __init__(self, args, inp, oup, stride, expand_ratio, candidates_in, candidates_out, layer_idx=None, blockwise=True):
        """ NOTE: in_planes: in width max; planes: mid width max;
            in_planes = self.expansion * planes / 2 if stride == 2.
        Args:
          args: block configuration
          inp: an int, block input MAX channel width;
          oup: an int, block mid conv layer MAX channel width;
          stride: an int, the stride for the first conv layer;
          expand_ratio: an float, the expansion multipliers;
          candidate_in: a list of int, candidate width for the INPUT CHANNEL of this block,
          candidate_out: a list of int, candidate width for the OUTPUT CHANNEL of this block,
          layer_idx: an int, the layer index;

        NOTE: inside the block, we should properly feed the correct cin_p,
              cout_p, candidates_in and candidates_out to ProjConv2d and TProjection.
        """

        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.args = args
        self.candidates_in = candidates_in
        self.candidates_out = candidates_out
        self.num_cand = len(candidates_in)
        self.stride = stride
        self.layer_idx = layer_idx if layer_idx else -3
        self.blockwise = blockwise
        self.expand_ratio = expand_ratio

        self.hidden_dims = [int(round(cand * expand_ratio)) for cand in candidates_in]

        self.conv1s, self.conv2s , self.conv3s = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.bn1s, self.bn2s, self.bn3s = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        self.shortcuts = nn.ModuleList()

        self.conv1_kernel = nn.Parameter(torch.Tensor(inp * expand_ratio, inp, 1, 1))
        self.conv2_kernel = nn.Parameter(torch.Tensor(inp * expand_ratio, 1, 3, 3)) # depthwise conv
        self.conv3_kernel = nn.Parameter(torch.Tensor(oup, inp * expand_ratio, 1, 1))
        self.reset_parameters()

        if expand_ratio != 1:
            # pw, NOTE that hidden dim is multiplied by expand_ratio according to cand_in
            for i, hidden in enumerate(self.hidden_dims):
                for j, cand_in in enumerate(self.candidates_in):
                    self.conv1s.append(ProjConv2d(self.conv1_kernel,\
                                                 cin_p=cand_in, \
                                                 cout_p=hidden, \
                                                 cin=inp, \
                                                 cout=inp * expand_ratio, \
                                                 kernel_size=1, \
                                                 candidate_width_p=self.hidden_dims, \
                                                 candidate_width_q=self.candidates_in, \
                                                 overlap=self.args.overlap, \
                                                 layer=self.layer_idx, \
                                                 stride=1, \
                                                 bias=False))
                    self.bn1s.append(nn.BatchNorm2d(hidden, affine=AFFINE))
        else:
            assert self.hidden_dims == self.candidates_in, 'Hidden dims do not match candidates_in'

        # dw, NOTE that in_channels = out_channels, pay attention with [cout, 1, k, k] kernels
        for j, hidden in enumerate(self.hidden_dims):
            self.conv2s.append(ProjConv2d(self.conv2_kernel,\
                                          cin_p=1, \
                                          cout_p=hidden, \
                                          cin=1, \
                                          cout=inp * expand_ratio, \
                                          kernel_size=3, \
                                          candidate_width_p=self.hidden_dims, \
                                          candidate_width_q=[1]*self.num_cand, \
                                          overlap=self.args.overlap, \
                                          layer=self.layer_idx+1, \
                                          stride=self.stride, \
                                          padding=1, \
                                          bias=False, \
                                          dw_groups=hidden))
            self.bn2s.append(nn.BatchNorm2d(hidden, affine=AFFINE))

        # pw-linear
        for i, cand_out in enumerate(self.candidates_out):
            for j, hidden in enumerate(self.hidden_dims):
                self.conv3s.append(ProjConv2d(self.conv3_kernel, \
                                              cin_p=hidden, \
                                              cout_p=cand_out, \
                                              cin=inp * expand_ratio, \
                                              cout=oup, \
                                              kernel_size=1, \
                                              candidate_width_p=self.candidates_out, \
                                              candidate_width_q=self.hidden_dims, \
                                              overlap=self.args.overlap, \
                                              layer=self.layer_idx+2, \
                                              stride=1, \
                                              bias=False))
                self.bn3s.append(nn.BatchNorm2d(cand_out, affine=AFFINE))

        # shortcut
        self.use_res_connect = self.stride == 1 and inp == oup
        if self.use_res_connect:
            for cand_out in self.candidates_out:
                for cand_in in self.candidates_in:
                      shortcut = ShortCutLayer(cand_in, cand_out, self.stride)
                      self.shortcuts.append(shortcut)

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv1_kernel, a=math.sqrt(5))
        init.kaiming_uniform_(self.conv2_kernel, a=math.sqrt(5))
        init.kaiming_uniform_(self.conv3_kernel, a=math.sqrt(5))

    def forward(self, x, archs):
        archs = archs.int().tolist()  # require arch to be list of list
        list_fn = lambda x: [x] if type(x) == int else x

        if self.expand_ratio == 1:
            assert len(archs) == 2, 'error length archs. for first inverted residual, len(arch) should be 2'
            arch2 = list_fn(archs[0])  # input==hidden chan
            arch3 = list_fn(archs[1])  # output chan
        else:
            assert len(archs) == 3, 'error length of archs. hidden layer does not need arch.'
            arch1 = list_fn(archs[0])  # inputs chan
            arch2 = list_fn(archs[1])  # hidden chan
            arch3 = list_fn(archs[2])  # output chan
        shots = len(arch2)

        # layer1 forward
        if self.expand_ratio != 1:
            h_1s = []
            for i, cand_out in enumerate(arch2):
                h_wis = []
                for j, cand_in in enumerate(arch1):
                    ind = cand_out * self.num_cand + cand_in
                    h = self.conv1s[ind](x[j])
                    h = self.bn1s[ind](h)
                    h_wis.append(h)
                h_wi_aggr = sum(h_wis) / shots
                h_wi_aggr = F.relu6(h_wi_aggr, inplace=True)  # aggregation before activation
                h_1s.append(h_wi_aggr)
        else:
            h_1s = x

        # layer2 forward, no need to aggregate, one-to-one connection
        h_2s = []
        for i, cand_out in enumerate(arch2):
            h = self.conv2s[cand_out](h_1s[i])
            h = self.bn2s[cand_out](h)
            h = F.relu6(h, inplace=True)
            h_2s.append(h)

        # layer3 forward
        outputs = []
        for i, cand_out in enumerate(arch3):
            h_wis = []
            for j, cand_in in enumerate(arch2):
                ind = cand_out * self.num_cand + cand_in
                h = self.conv3s[ind](h_2s[j])
                h = self.bn3s[ind](h)
                h_wis.append(h)  # aggregation before activation
                if self.use_res_connect:
                    cand_in_prev = arch1[j]
                    ind_prev = cand_out * self.num_cand + cand_in_prev
                    h += self.shortcuts[ind_prev](x[j])
            h_wi_aggr = sum(h_wis) / shots
            # NO ACTIVATION!
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

        # For the first layer orthogonal loss, only for expand_ratio != 1
        loss_p1, loss_q1 = 0, 0
        loss_p_n1, loss_q_n1 = 0, 0
        if self.expand_ratio != 1:
            proj_p_list, proj_q_list = [], []
            for conv in self.conv1s:
                proj_p_list.append(conv.tprojection.P)
                proj_q_list.append(conv.tprojection.Q)

            proj_p_groups, proj_q_groups = [], []
            for i in range(self.num_cand):
                tmp_p_group = torch.cat(proj_p_list[i:self.num_cand**2:self.num_cand], dim=1) # K x N, N > K  (168, 200)
                proj_p_groups.append(tmp_p_group)
                # e.g. [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
                tmp_q_group = torch.cat(proj_q_list[i*self.num_cand:(i+1)*self.num_cand], dim=1)
                proj_q_groups.append(tmp_q_group)
                # e.g. [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

            for p_group, q_group in zip(proj_p_groups, proj_q_groups):
                loss_p_orthog, loss_p_norm = calc_loss(p_group)
                loss_q_orthog, loss_q_norm = calc_loss(q_group)
                loss_p1 += loss_p_orthog
                loss_q1 += loss_q_orthog
                loss_p_n1 += loss_p_norm
                loss_q_n1 += loss_q_norm

        # The second layer, NOTE: no cross connection
        proj_p_list, proj_q_list = [], []
        for conv in self.conv2s:
            proj_p_list.append(conv.tprojection.P)
            proj_q_list.append(conv.tprojection.Q)
        proj_p_groups = torch.cat(proj_p_list, dim=1)
        proj_q_groups = torch.cat(proj_q_list, dim=1)

        loss_p_orthog, loss_p_norm = calc_loss(proj_p_groups)
        loss_q_orthog, loss_q_norm = calc_loss(proj_q_groups)
        loss_p2 = loss_p_orthog
        loss_q2 = loss_q_orthog
        loss_p_n2 = loss_p_norm
        loss_q_n2 = loss_q_norm

        # The last layer
        if self.blockwise:
            # NOTE: if blockwise == True, only calculate ortho loss for diagonal
            # connection (1-1), (2-2) e.t.c to save memory
            proj_p_list, proj_q_list = [], []
            for idx, conv in enumerate(self.conv3s):
                # no cross connection, only diagonal at 0,5,10,15
                if idx in [i*self.num_cand + i for i in range(self.num_cand)]:
                    # print(conv.tprojection.P.shape[1], conv.tprojection.Q.shape[1])
                    proj_p_list.append(conv.tprojection.P)
                    proj_q_list.append(conv.tprojection.Q)
            proj_p_groups = torch.cat(proj_p_list, dim=1)
            proj_q_groups = torch.cat(proj_q_list, dim=1)

            loss_p_orthog, loss_p_norm = calc_loss(proj_p_groups)
            loss_q_orthog, loss_q_norm = calc_loss(proj_q_groups)
            loss_p3 = loss_p_orthog
            loss_q3 = loss_q_orthog
            loss_p_n3 = loss_p_norm
            loss_q_n3 = loss_q_norm
        else:
            # Otherwise momory consuming! divide list into n_ops groups
            proj_p_list, proj_q_list = [], []
            for conv in self.conv3s:
                proj_p_list.append(conv.tprojection.P)
                proj_q_list.append(conv.tprojection.Q)

            proj_p_groups, proj_q_groups = [], []
            for i in range(self.num_cand):
                tmp_p_group = torch.cat(proj_p_list[i:self.num_cand**2:self.num_cand], dim=1) # K x N, N > K  (168, 200)
                proj_p_groups.append(tmp_p_group)
                tmp_q_group = torch.cat(proj_q_list[i*self.num_cand:(i+1)*self.num_cand], dim=1)
                proj_q_groups.append(tmp_q_group)

            loss_p3, loss_q3 = 0, 0
            loss_p_n3, loss_q_n3 = 0, 0
            for p_group, q_group in zip(proj_p_groups, proj_q_groups):
                loss_p_orthog, loss_p_norm = calc_loss_l2(p_group)
                loss_q_orthog, loss_q_norm = calc_loss_l2(q_group)
                loss_p3 += loss_p_orthog
                loss_q3 += loss_q_orthog
                loss_p_n3 += loss_p_norm
                loss_q_n3 += loss_q_norm

        return loss_p1 + loss_q1 + loss_p2 + loss_q2 + loss_p3 + loss_q3, \
               loss_p_n1 + loss_q_n1 + loss_p_n2 + loss_q_n2 + loss_p_n3 + loss_q_n3


def _make_group_divisible(vlist, multi, divisor, min_value=None):
    v_divied = []
    for v in vlist:
        v_divied.append(_make_divisible(v*multi, divisor, min_value))
    return v_divied


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class MobileNetV2ChannWidth(nn.Module):
    def __init__(self, num_classes=1000, \
                 base_width=BASE_WIDTH, \
                 max_width=MAX_WIDTH, \
                 width_multi = MULTIPLIER, \
                 round_nearest=2, \
                 args=None):
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding

        """
        super(MobileNetV2ChannWidth, self).__init__()

        # NOTE: default setting
        # # t, c, n, s
        # [1, 16, 1, 1],
        # [6, 24, 2, 2],
        # [6, 32, 3, 2],
        # [6, 64, 4, 2],
        # [6, 96, 3, 1],
        # [6, 160, 3, 2],
        # [6, 320, 1, 1],

        if args.rank == 0:
            if args.blockwise:
                print("Blockwise Enabled. Only search the outc of first pw conv in each block.")
            else:
                print("Blockwise Disabled. Search both the inc and outc of first pw conv in each block.")

        if base_width == BASE_WIDTH:
            self.base_width = base_width
        else:
            # passed in from outside
            self.base_width = [int(v) for v in base_width.split(',')]

        self.expansions = [1,3,3,3,6,6,6]

        # blockwise incremultal multiplier determined by the original net
        multi_list = [2., 1., 1.5, 1.3333, 2., 1.5, 1.6667, 2.] # default

        self.block_multi = []
        base = 1.
        for multi in multi_list[1:]:
            self.block_multi.append(base*multi)
            base *= multi
        self.width_multi = width_multi  # passed into _make_group_devisible()
        assert self.width_multi == 1.0, "For now lets set width_multi=1.0 in mobilenet_v2_width"
        self.block_base_width = [[int(b * m) for b in self.base_width] for m in self.block_multi] # the output cand width of each block

        self.num_blocks = [1,2,3,4,3,3,1]
        self.block_strides = [1,2,2,2,1,2,1]
        self.block_max_width = [int(max_width * b) for b in self.block_multi]

        # we have add another var: block_max_width into the setting
        self.inverted_residual_setting = [[t, cs, n, s, mw] for t, cs, n, s, mw in \
            zip(self.expansions, self.block_base_width, self.num_blocks, self.block_strides, self.block_max_width)]

        self.args = args
        self.num_cand = len(self.base_width)
        self.entropy_coeff = args.entropy_coeff

        self.block_layer_num = 3
        self.total_layers = sum(self.num_blocks) * self.block_layer_num + 2

        self.block_search_layer = 2
        self.num_search_layers = self.block_search_layer * sum(self.num_blocks) + 1

        if self.args.blockwise:
            arch_search_layers = sum(self.num_blocks) + len(self.num_blocks) # see OneNote, firstblock + first invres takes 2
        else:
            arch_search_layers = self.num_search_layers

        self.arch_master = ArchMaster(num_search_layers=arch_search_layers,
                                      num_blocks = self.num_blocks,
                                      n_ops=self.num_cand,
                                      controller_type=args.controller_type,
                                      controller_hid=args.controller_hid,
                                      controller_temperature=args.controller_temperature,
                                      controller_tanh_constant=args.controller_tanh_constant,
                                      controller_op_tanh_reduce=args.controller_op_tanh_reduce,
                                      blockwise=self.args.blockwise,
                                      max_flops=self.args.max_flops)

        # building first layer
        self.layer_idx = 0
        first_multi = multi_list[0]
        candidates_in = [b * first_multi for b in self.block_base_width[0]] # 2 is determined empirically
        candidates_in = _make_group_divisible(candidates_in, self.width_multi, round_nearest)
        in_mw = int(self.block_max_width[0]*first_multi*self.width_multi)
        layer0 = FirstBlock(self.args, \
                            planes=in_mw, \
                            candidate_width=candidates_in, \
                            layer_idx = self.layer_idx)
        features = [layer0]
        self.layer_idx += 1

        # building inverted residual blocks
        for t, cs, n, s, mw in self.inverted_residual_setting:
            candidates_out = _make_group_divisible(cs, self.width_multi, round_nearest)
            mw = _make_divisible(mw*self.width_multi, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                if self.args.rank == 0:
                    print("Configuration:", t, candidates_out, n, s, mw)
                features.append(InvertedResidual(self.args, in_mw, mw, stride, t, candidates_in, candidates_out, self.layer_idx, self.args.blockwise))
                self.layer_idx += self.block_layer_num
                in_mw = mw
                candidates_in = candidates_out

        # building last conv layers & classification layer
        candidates_out = [v * 4 for v in candidates_out]
        candidates_out = _make_group_divisible(candidates_out, self.width_multi, round_nearest)
        in_mw, mw = mw, 4 * mw
        features.append(LastBlock(self.args, in_mw, mw, candidates_in, candidates_out, self.layer_idx))
        self.layer_idx += 1

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = ClassfierBlock(self.args, mw, candidates_out, num_classes=num_classes, \
                                         dropout=self.args.dropout_rate, layer_idx=self.layer_idx)
        self._init_model()

        # compute flops_table
        self.flops_table_dict = self.compute_flops_table()
        print("Flops counting done.")
        with open('./mobilenet_v2_flops.pkl', 'wb') as f:
            pickle.dump(self.flops_table_dict, f)
        with open('./mobilenet_v2_flops.pkl', 'rb') as f:
            print("Restoring flops_table.pkl")
            self.flops_table_dict = pickle.load(f)
        list_flop_tensor = [torch.tensor(v).float().to(self.device) for v in list(self.flops_table_dict.values())]
        assert len(list_flop_tensor) == self.total_layers
        self.arch_master.table_flops = list_flop_tensor

    def _init_model(self):
        for m in self.modules():
            if isinstance(m, ProjConv2d):
                # alreadi initialized
                pass
            elif isinstance(m, nn.Conv2d):
                # initialize short cut conv
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) and AFFINE:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # already initialized in projlinear
                pass

    def forward(self, x):
        archs, archs_logP, archs_entropy = self.arch_master.forward()
        if self.args.distributed:
            # make sure different GPU share the same arch
            dist.broadcast(archs, 0)
            dist.broadcast(archs_logP, 0)
            dist.broadcast(archs_entropy, 0)

        if self.args.blockwise:
            # raw archs is non-expanded when blockwise=True
            extended_archs = self.extend_blockwise_archs(archs)
        else:
            extended_archs = archs
        assert len(extended_archs) == self.num_search_layers

        counts = 0
        h = x
        for i, layer in enumerate(self.features):
            if isinstance(layer, FirstBlock):
                h = layer(h, extended_archs[0])
            elif isinstance(layer, InvertedResidual) and counts == 0:
                h = layer(h, extended_archs[counts : counts+1+1])
                counts += 1
            elif isinstance(layer, InvertedResidual):
                h = layer(h, extended_archs[counts : counts+self.block_search_layer+1])
                counts += self.block_search_layer
            elif isinstance(layer, LastBlock):
                # counts == 34 if not blockwise
                h = layer(h, extended_archs[counts:])
                counts += 1
            else:
                raise ValueError("Unknown layer type.", layer)

        logits = self.classifier(h, extended_archs[counts])
        counts += 1
        assert counts == self.num_search_layers, "counts: %d, num_search_layers: %d, mismatch!" % (counts, self.num_search_layers)
        return logits, archs_logP, archs_entropy, archs

    def compute_flops_table(self, input_res=224, multiply_adds=MULTIPLY_ADDS):
        print("Computing flops table...")

        # obtain the proj_conv and feature maps
        layer_flop_dict={} # a dict containing the list of flops in that layer
        def proj_conv_hook(self, input, output):
            if self.layer not in layer_flop_dict.keys():
                layer_flop_dict[self.layer] = []

            _, input_channels, input_height, input_width = input[0].size()
            output_channels, output_height, output_width = output[0].size()

            if self.groups != 1:
                # for depth-wise channel conv
                kernel_ops = self.kernel_size[0] * self.kernel_size[1] * 1
            else:
                kernel_ops = self.kernel_size[0] * self.kernel_size[1] * self.in_channels
            bias_ops = 1 if self.bias is not None else 0

            params = output_channels * (kernel_ops + bias_ops)
            flops = (kernel_ops * (2 if multiply_adds else 1) + bias_ops) * output_channels * output_height * output_width * 1
            layer_flop_dict[self.layer].append(flops)

        def proj_linear_hook(self, input, output):
            if self.layer not in layer_flop_dict.keys():
                layer_flop_dict[self.layer] = []

            _, input_dim = input[0].size()
            output_dim = output[0].size(0) # 1000
            assert output_dim == 1000
            bias_ops = output_dim if self.bias is not None else 0
            params =  input_dim * output_dim + bias_ops
            flops = input_dim * output_dim * (2 if multiply_adds else 1) + bias_ops
            layer_flop_dict[self.layer].append(flops)

        flop_archs = torch.tensor([list(range(self.num_cand))]*self.num_search_layers).to(self.device)
        flop_handles = []
        for m in self.modules():
            if isinstance(m, ProjConv2d):
                handle = m.register_forward_hook(proj_conv_hook)
                flop_handles.append(handle)
            if isinstance(m, ProjLinear):
                handle=m.register_forward_hook(proj_linear_hook)
                flop_handles.append(handle)

        input_ = torch.rand(1, 3, input_res, input_res).to(self.device)
        input_.requires_grad = True
        out = self.test_forward(input_, flop_archs)
        for handle in flop_handles:
            handle.remove()
        return layer_flop_dict

    def orthogonal_reg_loss(self):
        loss_orthogonal, loss_norm = 0, 0
        for i, layer in enumerate(self.features):
            loss_orthogonal += layer.orthogonal_regularization()[0]
            loss_norm += layer.orthogonal_regularization()[1]
        loss_orthogonal += self.classifier.orthogonal_regularization()[0]
        loss_norm += self.classifier.orthogonal_regularization()[1]
        return loss_orthogonal, loss_norm

    def updt_orthogonal_reg_loss(self, proj_opt):
        # NOTE: update orthogonal loss inside to save memory.
        # return a python scalar
        loss_proj = 0.
        proj_opt.zero_grad()
        for i, layer in enumerate(self.features):
            ortho = layer.orthogonal_regularization()[0] * self.args.orthg_weight
            ortho.backward()
            loss_proj += ortho.item()

        ortho = self.classifier.orthogonal_regularization()[0] * self.args.orthg_weight
        ortho.backward()
        # average gradients and update all projection matrices once
        if self.args.distributed:
            average_group_gradients(self.model_parameters()) # not necessary
        proj_opt.step()
        loss_proj += ortho.item()
        return loss_proj

    def test_forward(self, x, archs):
        # forward with given archs
        if self.args.blockwise and len(archs) < self.num_search_layers:
            archs = self.extend_blockwise_archs(archs)
        assert len(archs) == self.num_search_layers

        counts = 0
        h = x
        for i, layer in enumerate(self.features):
            if isinstance(layer, FirstBlock):
                h = layer(h, archs[0])
            elif isinstance(layer, InvertedResidual) and counts == 0:
                h = layer(h, archs[counts : counts+1+1])
                counts += 1
            elif isinstance(layer, InvertedResidual):
                h = layer(h, archs[counts : counts+self.block_search_layer+1])
                counts += self.block_search_layer
            elif isinstance(layer, LastBlock):
                # counts == 34 if not blockwise
                h = layer(h, archs[counts:])
                counts += 1
            else:
                raise ValueError("Unknown layer type.", layer)

        logits = self.classifier(h, archs[counts])
        counts += 1
        assert counts == self.num_search_layers, "counts: %d, num_search_layers: %d, mismatch!" % (counts, self.num_search_layers)
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

        # NOTE: to speed up testing, we only sample archs without evaluation
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
        extend_archs = []
        block_id = 0
        dc_used = 0
        for layer_id, arch in enumerate(archs):
            if layer_id == 0:
                extend_archs += [arch]*2
                dc_used += 1
            elif layer_id == len(archs) - 1:
                # the last c, directly put in
                extend_archs += [arch]
            elif dc_used == self.num_blocks[block_id]:
                # arrive at ci==co
                block_id += 1
                dc_used = 0
                cio = arch
            else:
                # add dc
                extend_archs += [arch, cio]
                dc_used += 1
        extend_archs = torch.tensor(extend_archs).to(archs.device)
        return extend_archs

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
        return self.features[0].conv0_kernel.device


def mobilenet_v2_width(base_width=BASE_WIDTH, \
                       base_max_width=MAX_WIDTH, \
                       pretrained=False, \
                       progress=True,\
                       args=None):
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = MobileNetV2ChannWidth(base_width=base_width, \
                                  max_width=base_max_width, \
                                  width_multi=args.multiplier, \
                                  args=args)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v2'],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

