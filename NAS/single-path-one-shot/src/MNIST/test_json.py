import json 
import random


def get_arch_list(path):
    with open(path, "r") as f:
        arc_dict = json.load(f)
    
    arc_list = []

    for i, v in arc_dict.items():
        arc_list.append(v["arch"])    
    return arc_list, arc_dict



arch_l, arch_d = get_arch_list("Track1_final_archs.json")

# print(random.sample(arch_d, 4))



