import os
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from config import config
import sys
sys.setrecursionlimit(10000)
import functools
import copy
print=functools.partial(print,flush=True)
sys.path.append("../..")
from utils import *

# Algorithm 1
def get_arch_vector(model, cand):
    conv_bn = torch.reshape(model.conv_bn[0].weight.data,(-1,))
    conv1 = torch.reshape(model.MBConv_ratio_1.conv[0].weight.data,(-1,))
    conv2 = torch.reshape(model.MBConv_ratio_1.conv[3].weight.data,(-1,))
    conv_1x1_bn = torch.reshape(model.conv_1x1_bn[0].weight.data,(-1,))
    classifier = torch.reshape(model.classifier[0].weight.data,(-1,))
    arch_vector = [conv_bn, conv1, conv2, conv_1x1_bn, classifier]
    # block-like weight vector construction procedure is adopted
    for i, c in enumerate(cand):    
        if c >= 0:
            conv1 = torch.reshape(model.features[i]._ops[c].conv[0].weight.data, (-1,))
            conv2 = torch.reshape(model.features[i]._ops[c].conv[3].weight.data, (-1,))
            conv3 = torch.reshape(model.features[i]._ops[c].conv[6].weight.data, (-1,))
            arch_vector += [torch.cat([conv1, conv2, conv3], dim=0)]
    arch_vector = torch.cat(arch_vector, dim=0)
    return arch_vector

# Compute angle
def get_angle(cand, base_model, model):
    cosine = nn.CosineSimilarity(dim=0)
    vec1 = get_arch_vector(base_model, cand)
    vec2 = get_arch_vector(model, cand)
    angle = torch.acos(cosine(vec1, vec2))
    return angle

# 1) Make sure each child model is sampled only once
# 2) Dump models that don't satisfy flops constraint
def legal(cand, op_flops_dict, vis_dict):
    if len(cand) == 0:
        return False    
    assert isinstance(cand,tuple)
    if cand not in vis_dict:
        vis_dict[cand]={}
    info=vis_dict[cand]
    if 'visited' in info:
        return False
    flops = None
    if config.limit_flops:
        if 'flops' not in info:
            info['flops']= get_arch_flops(op_flops_dict, cand, config.backbone_info, config.blocks_keys)
        flops=info['flops']

        if config.max_flops is not None and flops > config.max_flops:
            return False
        if config.min_flops is not None and flops < config.min_flops:
            return False
    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print('{} cand = {} flops = {}'.format(now, cand, flops))
    info['visited']=True
    vis_dict[cand]=info
    return True

# Randomly sample the number of identity operators
def get_depth_full_rngs(ops, rngs, layer):
    identity_locs = []
    for i, op in enumerate(ops):
        if -1 in op and not i == layer:
            identity_locs.append(i)
    max_identity_num = len(identity_locs)
    identity_num = np.random.randint(max_identity_num+1)
    select_identity = np.random.choice(identity_locs, identity_num, replace=False)
    select_identity = list(select_identity)
    for i in range(len(select_identity)):
        rngs[select_identity[i]] = -1
    return rngs

def get_random_extend(num, op_flops_dict, extend_operator, vis_dict, ops):
    # 随机扩展

    def get_random_cand_(extend_operator, ops):
        # 得到随机的一个子网
        layer, extend_op = extend_operator
        rng = []
        for i, op in enumerate(ops):
            if i == layer and extend_op is not None:
                select_op = extend_op
            else:
                if len(op) == 1:
                    select_op = op[0]
                else:
                    if -1 in op:
                        assert(op[-1]==-1)
                        k = np.random.randint(len(op)-1)
                    else:
                        k = np.random.randint(len(op))
                    select_op = op[k]
            rng.append(select_op)
        rng = get_depth_full_rngs(ops, rng, layer)
        return tuple(rng)

    max_iters = num*100
    candidates = []
    i = 0
    while i<num and max_iters>0: 
        max_iters-=1
        cand = get_random_cand_(extend_operator, ops)
        if not legal(cand, op_flops_dict, vis_dict):
            continue
        candidates.append(cand)
        i+=1
        print('random {}/{}'.format(len(candidates),num))
    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print('{} random_num = {}'.format(now, len(candidates)))
    return candidates

def compute_scores(base_model, model, operations, extend_operators, vis_dict_slice, vis_dict, op_flops_dict):
    candidates = []
    # Randomly sample some architectures which contain current operator
    for idx, extend_operator in enumerate(extend_operators):
        info = vis_dict_slice[extend_operator] # 对应层、对应op所记录的数据结构
        if config.sample_num-len(info['cand_pool']) > 0: # 这里采样1000个子网络进行统计
            num = config.sample_num-len(info['cand_pool'])
            cands = get_random_extend(num, op_flops_dict, extend_operator, vis_dict, operations) # 取1000个候选网络
            for cand in cands:
                # 每个候选网络的处理
                for i, c in enumerate(cand): 
                    extend_operator_ = (i, c)
                    if extend_operator_ in vis_dict_slice:
                        info = vis_dict_slice[extend_operator_]
                        if cand not in info['cand_pool']:
                            info['cand_pool'].append(cand) # 加入候选网络池
                if cand not in candidates:
                    candidates.append(cand)

    # Compute angles of all candidate architecures
    for i, cand in enumerate(candidates):
        info=vis_dict[cand]
        info['angle'] = get_angle(cand, base_model, model) # 计算对应的angle评分
        print('idx: {}, angle: {}'.format(i, info['angle']))

    # Caculate sum of angles for each operator
    # 计算每个op对应的angle值，用于评价其效果
    for cand in candidates:
        cand_info = vis_dict[cand]
        for i, c in enumerate(cand): 
            extend_operator_ = (i, c)
            if extend_operator_ in vis_dict_slice:
                slice_info = vis_dict_slice[extend_operator_]
                if cand in slice_info['cand_pool'] and slice_info['count'] < config.sample_num:
                    slice_info['angle'] += cand_info['angle']
                    slice_info['count'] += 1 # TODO

    # Compute scores of all candidate operators           
    for extend_operator in extend_operators:
        if vis_dict_slice[extend_operator]['count'] > 0:
            vis_dict_slice[extend_operator]['angle'] = vis_dict_slice[extend_operator]['angle'] * 1. / vis_dict_slice[extend_operator]['count']

def drop_operators(extend_operators, vis_dict_slice, operations, iters):
    # Each operator is ranked according to its score
    extend_operators.sort(key=lambda x:vis_dict_slice[x]['angle'], reverse=False)
    for idx, cand in enumerate(extend_operators):
        info = vis_dict_slice[cand]
        print('Iter={}, shrinking: top {} cand={}, angle={}, count={}'.format(iters+1, idx+1, cand, info['angle'], info['count']))

    # Drop operators whose ranking fall at the tail.
    num, drop_ops = 0, []
    for idx, cand in enumerate(extend_operators):
        layer, op = cand
        drop_legal = False
        # Make sure that at least a operator is reserved for each layer.
        for i in range(idx+1, len(extend_operators)):
            layer_, op_ = extend_operators[i]
            if layer_ == layer:
                drop_legal = True
        if drop_legal:
            print('no.{} drop_op={}'.format(num+1, cand))
            drop_ops.append(cand)
            operations[layer].remove(op)
            num += 1
        if num >= config.per_stage_drop_num:
            break
    return operations, drop_ops

# Algorithm 2
def ABS(base_model, model, operations, iters):
    '''
    base_model 代表初始化的权重
    model： 代表训练的权重
    '''
    vis_dict_slice, vis_dict = {}, {}

    op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))
    print('|=> Iters={}, shrinking: operations={}'.format(iters+1, operations))

    # At least one operator is preserved for each edge
    # Each operator is identified by its layer and type
    extend_operators = []
    for layer, ops in enumerate(operations): # layer: 4
        if len(ops) > 1: # ops: [1,2,3]
            for op in ops: # op: 2
                operator = tuple([layer, op]) # [4, 2]
                vis_dict_slice[operator]={}
                info=vis_dict_slice[operator]
                info['count'] = 0. # 计算该层4,选择了第2个op，该操作对应操作的参数
                info['angle'] = 0.
                info['cand_pool'] = []
                extend_operators.append(operator)

    # 计算score，随机采样大批子网，然后排序，统计op的好坏
    compute_scores(base_model, model, operations, extend_operators, vis_dict_slice, vis_dict, op_flops_dict)
    operations, drop_ops = drop_operators(extend_operators, vis_dict_slice, operations, iters)
    print('Iter={}, shrinking: drop_ops={}, operations={}'.format(iters+1, drop_ops, operations))
    return operations
