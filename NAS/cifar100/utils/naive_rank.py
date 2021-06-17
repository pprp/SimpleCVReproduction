from statistics import get_arch_list
import json 

'''
直接根据参数量进行排序
'''

arch_l, arch_d = get_arch_list("data/Track1_final_archs.json")


def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort()
    return [backitems[i][1] for i in range(0, len(backitems))]


dict_arch_param = {}

result_dict = {}

for key, value in arch_d.items():
    sum_s = sum([int(i) for i in value['arch'].split("-")])
    acc = sum_s / 752
    print(key, sum_s, '\t', acc)
    dict_arch_param[key] = sum_s

    tmp_dict = {}
    tmp_dict['acc'] = acc 
    tmp_dict['arch'] = value['arch']

    result_dict[key] = tmp_dict

print(result_dict)

# print(sort_by_value(dict_arch_param))

with open("naive_result.json",'w') as f:
    json.dump(result_dict, f)