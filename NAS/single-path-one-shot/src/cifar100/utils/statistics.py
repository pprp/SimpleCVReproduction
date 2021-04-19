import json
import matplotlib.pyplot as plt


def get_arch_list(path):
    with open(path, "r") as f:
        arc_dict = json.load(f)

    arc_list = []

    for i, v in arc_dict.items():
        arc_list.append(v["arch"])
    return arc_list, arc_dict


# arc_list, _ = get_arch_list("Track1_final_archs.json")

# plt.figure() 

# index = 17

# count_dict = {}

# for i in range(len(arc_list)):
#     arc_i = arc_list[i].split('-')
#     arc_choices = [int(item) for item in arc_i]

#     choice_index = arc_choices[index]

#     if choice_index not in count_dict.keys():
#         count_dict[choice_index] = 1
#     else:
#         count_dict[choice_index] += 1
    

# x = list(count_dict.keys())
# y = list(count_dict.values())

# plt.bar(x,y)
# plt.show()

