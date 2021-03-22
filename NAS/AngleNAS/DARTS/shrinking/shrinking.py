import os
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
from super_model import Network_ImageNet
from torch.autograd import Variable
from config import config
import sys
sys.setrecursionlimit(10000)
import functools
import copy
print=functools.partial(print,flush=True)
from angle import get_angle

sys.path.append("../..")
from utils import *

# Make sure each child model is sampled only once
def legal(cand, vis_dict):
    if len(cand) == 0:
        return False    
    assert isinstance(cand,tuple)
    if cand not in vis_dict:
        vis_dict[cand]={}
    info=vis_dict[cand]
    if 'visited' in info:
        return False
    info['visited']=True
    vis_dict[cand]=info
    return True

# Randomly sample finite number of child models containing current operator
def get_random_extend(num, extend_operator, vis_dict, operations):
    def get_random_cand_(extend_operator, operations):
        edge, extend_op = extend_operator
        rng, cell_rng = [], []

        for op in operations:
            k = np.random.randint(len(op))
            select_op = op[k]
            cell_rng.append(select_op)

        for _ in range(config.layers):
            rng.append(copy.deepcopy(cell_rng))
        rng = check_cand(rng, operations, config.edges)
        
        if extend_op is not None:
            for i in range(config.layers):
                rng[i][edge] = extend_op
        rng = np.reshape(rng, -1)
        return tuple(rng)

    max_iters = num*100
    candidates = []
    i = 0
    while i<num and max_iters>0: 
        max_iters-=1
        cand = get_random_cand_(extend_operator, operations)
        if not legal(cand, vis_dict):
            continue
        candidates.append(cand)
        i+=1
        print('random {}/{}'.format(len(candidates),num))
    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
    print('{} random_num = {}'.format(now, len(candidates)))
    return candidates

def compute_scores(base_model, model, operations, extend_operators, vis_dict_slice, vis_dict):
    candidates = []
    # 1000 child models are collected for each operator
    for idx, extend_operator in enumerate(extend_operators):
        info = vis_dict_slice[extend_operator]
        if config.sample_num-len(info['cand_pool']) > 0:
            random_n = config.sample_num - len(info['cand_pool'])
            cands = get_random_extend(random_n, extend_operator, vis_dict, operations)
            for cand in cands:
                cand_ = np.reshape(cand, [config.layers, -1])
                for j, c in enumerate(cand_[0]): 
                    extend_operator_ = (j, c)
                    if extend_operator_ in vis_dict_slice:
                        info = vis_dict_slice[extend_operator_]
                        if cand not in info['cand_pool']:
                            info['cand_pool'].append(cand)
                if cand not in candidates:
                    candidates.append(cand)

    # Compute angles of all candidate architecures
    for i, cand in enumerate(candidates):
        info=vis_dict[cand]
        info['angle'] = get_angle(base_model, model, cand)
        print('idx: {}, angle: {}'.format(i, info['angle']))

    # Caculate sum of angles for each operator
    for cand in candidates:
        cand_info = vis_dict[cand]
        cand_ = np.reshape(cand, [config.layers, -1])
        for j, c in enumerate(cand_[0]):   
            extend_operator_ = (j, c)
            if extend_operator_ in vis_dict_slice:
                slice_info = vis_dict_slice[extend_operator_]
                if cand in slice_info['cand_pool'] and slice_info['count'] < config.sample_num:
                    slice_info['angle'] += cand_info['angle']
                    slice_info['count'] += 1

    # The score of each operator is acquired by averaging the angle of child models containing it
    for extend_operator in extend_operators:
        if vis_dict_slice[extend_operator]['count'] > 0:
            vis_dict_slice[extend_operator]['angle'] = vis_dict_slice[extend_operator]['angle'] * 1. / vis_dict_slice[extend_operator]['count'] 

def drop_operators(extend_operators, vis_dict_slice, operations, drop_iter):
    # Each operator is ranked according to its score
    extend_operators.sort(key=lambda x:vis_dict_slice[x]['angle'], reverse=False)
    for idx, cand in enumerate(extend_operators):
        info = vis_dict_slice[cand]
        print('Iter={}, shrinking: top {} cand={}, angle={}, count={}'.format(drop_iter+1, idx+1, cand, info['angle'], info['count']))
    
    # ABS removes one operator for each edge each time
    num, drop_ops = 0, []
    for j in range(len(operations)):
        for idx, cand in enumerate(extend_operators):
            edge, op = cand
            if edge == j:
                print('no.{} drop_op={}'.format(num+1, cand))
                drop_ops.append(cand)
                operations[edge].remove(op)
                extend_operators.remove(cand)
                num += 1
                break
    return operations, drop_ops

# Algorithm 2
def ABS(base_model, model, operations, iters):
    vis_dict_slice, vis_dict  = {}, {}
    print('|=> Iters={}, shrinking: operations={}'.format(iters, operations))
    # At least one operator is preserved in each edge
    # Each operator is identified by its edge and type
    extend_operators = []
    for edge, op in enumerate(operations):
        if len(op) > 1:
            for op_ in op:
                cand = tuple([edge, op_])
                vis_dict_slice[cand]={}
                info=vis_dict_slice[cand]
                info['angle'] = 0.
                info['count'] = 0.
                info['cand_pool'] = []
                extend_operators.append(cand)

    compute_scores(base_model, model, operations, extend_operators, vis_dict_slice, vis_dict)
    operations, drop_ops = drop_operators(extend_operators, vis_dict_slice, operations, iters)
    print('Iter={}, shrinking: drop_ops={}, operations={}'.format(iters, drop_ops, operations))
    return operations
