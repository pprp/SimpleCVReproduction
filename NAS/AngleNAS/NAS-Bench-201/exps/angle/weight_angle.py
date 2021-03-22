import os, sys, time, random
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from models import OPS_CODING

def get_weight(model):
  weight = {}
  for name, param in model.named_parameters():
    weight[name] = param.data
  return weight

def get_arch_real_acc(api, genotype):
  info = api.query_by_arch(genotype).split("\n")[7]
  acc = info.split(":")[-1].split("=")[-1].split("]")[0]
  return float(acc[:-1])

def get_head_vector(weight):
  param_vectors = []
  param_vectors.append(weight["stem.0.weight"].reshape(-1))
  param_vectors.append(weight["stem.1.weight"].reshape(-1))
  param_vectors.append(weight["stem.1.bias"].reshape(-1))

  return torch.cat(param_vectors, dim=0)


def get_fixed_cell_vector(index, weight):
  param_vectors = []
  param_vectors.append(weight["cells.{}.conv_a.op.1.weight".format(index)].reshape(-1))
  param_vectors.append(weight["cells.{}.conv_a.op.2.weight".format(index)].reshape(-1))
  param_vectors.append(weight["cells.{}.conv_a.op.2.bias".format(index)].reshape(-1))
  param_vectors.append(weight["cells.{}.conv_b.op.1.weight".format(index)].reshape(-1))
  param_vectors.append(weight["cells.{}.conv_b.op.2.weight".format(index)].reshape(-1))
  param_vectors.append(weight["cells.{}.conv_b.op.2.bias".format(index)].reshape(-1))
  param_vectors.append(weight["cells.{}.downsample.1.weight".format(index)].reshape(-1))

  return torch.cat(param_vectors, dim=0)


def get_tail_vector(weight):
  param_vectors = []
  param_vectors.append(weight["lastact.0.weight"].reshape(-1))
  param_vectors.append(weight["lastact.0.bias"].reshape(-1))
  param_vectors.append(weight["classifier.weight"].reshape(-1))
  param_vectors.append(weight["classifier.bias"].reshape(-1))

  return torch.cat(param_vectors, dim=0)

def get_arch_angle(model1, model2, arch, search_space):
  cosine = nn.CosineSimilarity(dim=0).cuda()
  model1, model2 = model1.cpu(), model2.cpu() 
  init_weight = get_weight(model1)
  weight = get_weight(model2)

  cell_index_list = [0, 1, 2, 3, 4, 5,  6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
  fixed_cell_index_list = [5, 11]
  name_template = "cells.{}.edges.{}<-{}.{}.op.1.weight"
  nodes = arch.nodes

  angle_vector_history = []
  init_angle_vector_history = []

  # Get weights of tail vectors
  init_tail_vector = get_tail_vector(init_weight)
  tail_vector = get_tail_vector(weight)
  angle_vector_history.append(deepcopy(tail_vector))
  init_angle_vector_history.append(deepcopy(init_tail_vector))

  # Get weights of head vectors
  init_head_vector = get_head_vector(init_weight)
  head_vector = get_head_vector(weight)
  angle_vector_history.append(deepcopy(head_vector))
  init_angle_vector_history.append(deepcopy(init_head_vector))

  # Get fixed weights of cells
  for cell_index in fixed_cell_index_list:
      init_weight_vector = get_fixed_cell_vector(cell_index, init_weight)
      weight_vector = get_fixed_cell_vector(cell_index, weight)
      init_angle_vector_history.append(init_weight_vector)
      angle_vector_history.append(weight_vector)

  # Enumerate all paths of a single cell. 
  # Path encoding is defined by node No. (e.g., [0,3] represents a path from node 0 to node 3) 
  paths = [[0, 3], [0, 2, 3], [0, 1, 2, 3], [0, 1, 3]]

  # Algorithm 1 (ABS): the block-like weight vector construction procedure is adopted
  for path in paths:
    for cell_index in cell_index_list:
      if cell_index in fixed_cell_index_list:
        continue
      init_param_list, param_list = [], []
      for index in range(1, len(path)):
        i, j = path[index], path[index-1]
        op_index = search_space.index(nodes[i-1][j][0])
        # "none " totally changes the connectivity of the child model
        if op_index == OPS_CODING['none']: # none
          param_list = []
          break
        #  For “pooling” and “identity” operators, we assign a fixed weight to them
        # “identity” has empty weights
        elif op_index == OPS_CODING['skip_connect']: # skip connect
          pass
        # “pooling” has k × k kernel, where elements are all 1/k^2, k is the pooling size
        elif op_index == OPS_CODING['avg_pool_3x3']: # 3x3 avg pooling
          weight_name = name_template.format(cell_index, i, j, 3)
          shape = list(init_weight[weight_name].shape)
          shape = [shape[0], shape[2], shape[3]]
          pooling_param = torch.ones(shape) * (1. / 9.)
          init_param_list += [pooling_param.reshape(-1)]
          param_list += [deepcopy(pooling_param).reshape(-1)]
        # For "convolution" operators, we reshape their weight into a vector
        elif op_index == OPS_CODING['nor_conv_1x1'] or op_index == OPS_CODING['nor_conv_3x3']: # 1x1 conv or 3x3 conv
          weight_name = name_template.format(cell_index, i, j, op_index)
          init_param_list += [init_weight[weight_name].reshape(-1)]
          param_list += [weight[weight_name].reshape(-1)]
        else:
          raise Exception('Invalid Operators!')

      if len(param_list) != 0:
        angle_vector_history.append(torch.cat(param_list, dim=0))
        init_angle_vector_history.append(torch.cat(init_param_list, dim=0))

  # Angle is aquired by concating weights of all paths
  return torch.acos(cosine(torch.cat(init_angle_vector_history, dim=0).cuda(), torch.cat(angle_vector_history, dim=0).cuda())).cpu().item()