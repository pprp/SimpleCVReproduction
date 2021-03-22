import torch
from config import config
from operations import *
import numpy as np
import copy
from torch.autograd import Variable

# Get weight vector of a child model (Algorithm 1)
# Block-like weight vector construction procedure is adopted
def get_arch_vector(model, cand):
    cand = np.reshape(cand, [config.layers, -1])
    arch_vector, extra_params = [], []
    # Collect extra parameters
    stem0 = torch.cat([model.stem0[0].weight.data.reshape(-1), model.stem0[3].weight.data.reshape(-1)])
    stem1 = model.stem1[1].weight.data.reshape(-1)
    extra_params += [stem0, stem1]
    
    for i in range(len(model.cells)):
        # Collect extra parameters
        if isinstance(model.cells[i].preprocess0, FactorizedReduce):
            s0 = torch.cat([model.cells[i].preprocess0.conv_1.weight.data.reshape(-1),model.cells[i].preprocess0.conv_2.weight.data.reshape(-1)])
        else:
            s0 = model.cells[i].preprocess0.op[1].weight.data.reshape(-1)
        s1 = model.cells[i].preprocess1.op[1].weight.data.reshape(-1)
        extra_params += [s0, s1]

        # Collect weight vecors of all paths
        param_list = []
        for path in config.paths:
            param_cell = []
            for index in range(1, len(path)):
                j = path[index]
                k = path[index-1]
                assert(j>=2)
                offset = 0
                for tmp in range(2,j):
                    offset += tmp

                if cand[i][k+offset] == config.NONE: # None
                    param_cell = []
                    break

                elif cand[i][k+offset] == config.MAX_POOLING_3x3 or cand[i][k+offset] == config.AVG_POOL_3x3: # pooling
                    # Simulate convolution
                    shape = model.cells[i]._ops[k+offset]._ops[4].op[1].weight.data.shape
                    shape = [shape[0], shape[2], shape[3]]
                    pooling_param = Variable(torch.ones(shape) * (1 / 9.),requires_grad=False).cuda()
                    param_cell += [copy.deepcopy(pooling_param).reshape(-1)]

                elif cand[i][k+offset] == config.SKIP_CONNECT: # identity
                    pass

                elif cand[i][k+offset] == config.SEP_CONV_3x3 or cand[i][k+offset] == config.SEP_CONV_5x5: #sep conv
                    conv1 = torch.reshape(model.cells[i]._ops[k+offset]._ops[cand[i][k+offset]].op[1].weight.data, (-1,))
                    conv2 = torch.reshape(model.cells[i]._ops[k+offset]._ops[cand[i][k+offset]].op[2].weight.data, (-1,))
                    conv3 = torch.reshape(model.cells[i]._ops[k+offset]._ops[cand[i][k+offset]].op[5].weight.data, (-1,))
                    conv4 = torch.reshape(model.cells[i]._ops[k+offset]._ops[cand[i][k+offset]].op[6].weight.data, (-1,))
                    conv_cat = torch.cat([conv1, conv2, conv3, conv4])
                    param_cell += [conv_cat]


                elif cand[i][k+offset] == config.DIL_CONV_3x3 or cand[i][k+offset] == config.DIL_CONV_5x5: 
                    conv1 = torch.reshape(model.cells[i]._ops[k+offset]._ops[cand[i][k+offset]].op[1].weight.data, (-1,))
                    conv2 = torch.reshape(model.cells[i]._ops[k+offset]._ops[cand[i][k+offset]].op[2].weight.data, (-1,))
                    conv_cat = torch.cat([conv1, conv2])
                    param_cell += [conv_cat]

                else:
                    raise Exception("Invalid operators !")

            # Get weight vector of a single path  
            if len(param_cell) != 0:
                param_list.append(torch.cat(param_cell))

        # Get weight vector of a cell
        if len(param_list)!=0: 
            arch_vector.append(torch.cat(param_list, dim=0))
        
    # Collect extra parameters
    extra_params.append(torch.reshape(model.classifier.weight.data,(-1,)))
    arch_vector += extra_params

    # Get weight vector of the whole model
    if len(arch_vector) != 0:
        arch_vector = torch.cat(arch_vector, dim=0)
    return arch_vector

def get_angle(base_model, model, cand):
    cosine = torch.nn.CosineSimilarity(dim=0)
    vec1 = get_arch_vector(base_model, cand)
    vec2 = get_arch_vector(model, cand)
    return torch.acos(cosine(vec1, vec2))