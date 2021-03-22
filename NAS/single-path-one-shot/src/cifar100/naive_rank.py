from test_json import get_arch_list


'''
直接根据参数量进行排序
'''

arch_l, arch_d = get_arch_list("Track1_final_archs.json")


def sort_by_value(d):
    items = d.items()
    backitems = [[v[1], v[0]] for v in items]
    backitems.sort()
    return [backitems[i][1] for i in range(0, len(backitems))]


dict_arch_param = {}

for key, value in arch_d.items():
    sum_s = sum([int(i) for i in value['arch'].split("-")])
    # print(key, value['arch'], sum_s)
    dict_arch_param[key] = sum_s


print(sort_by_value(dict_arch_param))
