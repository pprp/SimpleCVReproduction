import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from model.dynamic_resnet20 import dynamic_resnet20


arc_representation = "16-8-16-16-8-12-12-20-12-4-12-32-32-24-48-8-52-16-12-36"

'''
conv1_1   : 1-16  : 12-12 : 32-32 :
conv1_2   : 16-8  : 12-20 : 32-24 :
shortcut1 : 3-8   : 12-20 : 32-24 :

conv1_1   : 8-16  : 20-12 : 24-48 :
conv1_2   : 16-16 : 12-4  : 48-8  :
shortcut1 : 8-16  : 20-4  : 24-8  :

conv1_1   : 16-8  : 4-12  : 8-52  :
conv1_2   : 8-12  : 12-32 : 52-16 :
shortcut1 : 16-12 : 4-32  : 8-16  :
'''

"""
first conv
block1
    layer1
        conv1
        conv2
        downsample
    layer2
        conv1
        conv2
        downsample
    layer3
        conv1
        conv2
        downsample
block2
    layer1
        conv1
        conv2
        downsample
    layer2
        conv1
        conv2
        downsample
    layer3
        conv1
        conv2
        downsample
block3
    layer1
        conv1
        conv2
        downsample
    layer2
        conv1
        conv2
        downsample
    layer3
        conv1
        conv2
        downsample
"""


def generate_arch_vector(model: dynamic_resnet20, candidate):
    # process candidate
    cand = [int(item) for item in candidate.split('-')]
    # print(cand, len(cand))

    cnt = 0
    idx = cnt * 2

    # first conv
    first_conv = torch.reshape(
        model.first_conv.conv.conv.weight[:cand[idx], :1, :, :].data, (-1,))
    arch_vector = [first_conv]

    last_channel = 1  # 3 when cifar100 1 when mnist

    for i in range(3):
        block_i = getattr(model, "block%d" % (i+1))
        for j in range(3):
            # print(idx, cand[idx], cand[idx+1], cand[idx+2])

            layer_i = getattr(block_i, "layer%d" % (j+1))

            conv1 = torch.reshape(
                layer_i.conv1.conv.conv.weight[:cand[idx+1], :cand[idx], :, :].data, (-1,))
            conv2 = torch.reshape(
                layer_i.conv2.conv.conv.weight[:cand[idx+2], :cand[idx+1], :, :].data, (-1,))
            downs = torch.reshape(
                layer_i.downsample.conv.conv.weight[:cand[idx+2], :cand[idx], :, :].data, (-1,))

            arch_vector += [torch.cat([conv1, conv2, downs], dim=0)]

            cnt += 1
            idx = cnt * 2

    return torch.cat(arch_vector, dim=0)


def generate_angle(b_model, t_model, candidate):
    vec1 = generate_arch_vector(b_model, candidate)
    vec2 = generate_arch_vector(t_model, candidate)
    cos = nn.CosineSimilarity(dim=0)
    angle = torch.acos(cos(vec1, vec2))
    return angle


if __name__ == "__main__":
    m1 = dynamic_resnet20()
    m2 = dynamic_resnet20()
    print(generate_angle(m1, m2, arc_representation))
