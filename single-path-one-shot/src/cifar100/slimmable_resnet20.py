import torch 
import torch.nn as nn

# from slimmable_op import SlimmableConv2d, SlimmableLinear

arc_representation = "4-12-4-4-16-8-4-12-32-24-16-8-8-24-60-12-64-64-52-60"

def get_arc_list(arc_rep):
    return [int(item) for item in arc_rep.split('-')]


print(get_arc_list(arc_representation))