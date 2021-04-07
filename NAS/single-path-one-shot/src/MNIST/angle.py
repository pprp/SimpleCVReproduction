import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from slimmable_resnet20 import mutableResNet20


arc_representation = "16-8-16-16-8-12-12-20-12-4-12-32-32-24-48-8-52-16-12-36"


def generate_arch_vector(model: mutableResNet20, candidate):
    # first conv
    first_conv = torch.reshape(model.first_conv.weight.data, (-1,))
    arch_vector = [first_conv]

    for i in range(1):  # 3个layer
        layer_i = getattr(model, 'layer%d' % (i+1))

        conv1_1 = torch.reshape(layer_i[0].body[0].weight.data, (-1,))
        conv1_2 = torch.reshape(layer_i[0].body[3].weight.data, (-1,))
        shortcut1 = torch.reshape(layer_i[0].shortcut2[0].weight.data, (-1,))

        conv2_1 = torch.reshape(layer_i[1].body[0].weight.data, (-1,))
        conv2_2 = torch.reshape(layer_i[1].body[3].weight.data, (-1,))
        shortcut2 = torch.reshape(layer_i[1].shortcut2[0].weight.data, (-1,))

        conv3_1 = torch.reshape(layer_i[2].body[0].weight.data, (-1,))
        conv3_2 = torch.reshape(layer_i[2].body[0].weight.data, (-1,))
        shortcut3 = torch.reshape(layer_i[2].shortcut2[0].weight.data, (-1,))

        arch_vector += [torch.cat([conv1_1, conv1_2, conv2_1, conv2_2,
                                   conv3_1, conv3_2, shortcut1, shortcut2, shortcut3], dim=0)]

    return torch.cat(arch_vector, dim=0)


def generate_angle(b_model, t_model, candidate):
    vec1 = generate_arch_vector(b_model, candidate)
    vec2 = generate_arch_vector(t_model, candidate)
    cos = nn.CosineSimilarity(dim=0)
    angle = torch.acos(cos(vec1, vec2))
    return angle


if __name__ == "__main__":
    m1 = mutableResNet20()
    m2 = mutableResNet20()
    print(generate_angle(m1, m1, arc_representation))
