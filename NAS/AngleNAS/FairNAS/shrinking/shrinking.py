import os
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F

from super_model import SuperNetwork
from torch.autograd import Variable
from config import config
import sys
sys.setrecursionlimit(10000)
import copy
import functools
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
            info['flops'] = get_arch_flops(op_flops_dict, cand, config.backbone_info, config.blocks_keys)
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

def get_random_extend(num, op_flops_dict, extend_operator, vis_dict, ops):
    def get_random_cand_(extend_operator, ops):
        layer, extend_op = extend_operator
        rng = []
        for i, op in enumerate(ops):
            if i == layer and extend_op is not None:
                select_op = extend_op
            else:
                k = np.random.randint(len(op))
                select_op = op[k]
            rng.append(select_op)
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
    # Randomly sample some architectures which contain one operator
    print('Extend_cands={}'.format(extend_operators))
    for idx, extend_operator in enumerate(extend_operators):
        info = vis_dict_slice[extend_operator]
        num = config.random_num-len(info['cand_pool'])
        if num > 0:
            cands = get_random_extend(num, op_flops_dict, extend_operator, vis_dict, operations)
            for cand in cands:
                for i, c in enumerate(cand): 
                    extend_operator_ = (i, c)
                    if extend_operator_ in vis_dict_slice:
                        info = vis_dict_slice[extend_operator_]
                        if cand not in info['cand_pool']:
                            info['cand_pool'].append(cand)
                if cand not in candidates:
                    candidates.append(cand)

    # Compute angles of all candidate architecures
    for i, cand in enumerate(candidates):
        info=vis_dict[cand]
        info['angle'] = get_angle(cand, base_model, model)
        print('idx: {}, angle: {}'.format(i, info['angle']))

    # Caculate sum of angles for each operator
    for cand in candidates:
        cand_info = vis_dict[cand]
        for i, c in enumerate(cand): 
            extend_operator_ = (i, c)
            if extend_operator_ in vis_dict_slice:
                slice_info = vis_dict_slice[extend_operator_]
                if cand in slice_info['cand_pool'] and slice_info['count'] < config.random_num:
                    slice_info['angle'] += cand_info['angle']
                    slice_info['count'] += 1

    # Compute scores of all candidate operators           
    for extend_operator in extend_operators:
        if vis_dict_slice[extend_operator]['count'] > 0:
            vis_dict_slice[extend_operator]['angle'] = vis_dict_slice[extend_operator]['angle'] * 1. / vis_dict_slice[extend_operator]['count']
    
def drop_operators(extend_operators, vis_dict_slice, operations, iters):
    # Sort via the angle-based metric
    num = 0
    extend_operators.sort(key=lambda x:vis_dict_slice[x]['angle'], reverse=False)
    for idx, cand in enumerate(extend_operators):
        info = vis_dict_slice[cand]
        print('Iter={} shrinking: top {} cand={}, angle={}, count={}'.format(iters+1, idx+1, cand, info['angle'], info['count']))

    # Drop operators whose ranking fall at the tail. 
    # For FairNAS, ABS removes one operator for each layer each time because of its fair constraint 
    num, drop_ops = 0, []
    for i in range(len(operations)):
        for idx, cand in enumerate(extend_operators):
            layer, op = cand
            if layer == i:
                print('no.{} drop_op={}'.format(num+1, cand))
                drop_ops.append(cand)
                operations[layer].remove(op)
                extend_operators.remove(cand)
                num += 1
                break
    return operations, drop_ops

# Algorithm 2
def ABS(base_model, model, operations, iters):
    vis_dict_slice, vis_dict= {}, {}
    op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))
    print('|=> Iter={}, shrinking: operations={}'.format(iters, operations))
    # At least one operator is preserved in each edge
    # Each operator is identified by its layer and type
    extend_operators = []
    for layer, ops in enumerate(operations):
        if len(ops) > 1:
            for op in ops:
                cand = tuple([layer, op])
                vis_dict_slice[cand]={}
                info=vis_dict_slice[cand]
                info['count'] = 0.
                info['angle'] = 0.
                info['cand_pool'] = []
                extend_operators.append(cand)

    compute_scores(base_model, model, operations, extend_operators, vis_dict_slice, vis_dict, op_flops_dict)
    operations, drop_ops = drop_operators(extend_operators, vis_dict_slice, operations, iters)
    print('Iter={}, shrinking: drop_ops={}, operations={}'.format(iters, drop_ops, operations))
    return operations
