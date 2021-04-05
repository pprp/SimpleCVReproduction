

import json
import math
import random
import numpy as np
import re
import torch
import os
import copy
from mobilenetv2 import MobileNetLike



class GeneCoder:
    def __init__(self, json_path):
        self.ss_path = json_path
        self.search_space = []
        with open(self.ss_path, "r") as fin:
            self.search_space = json.load(fin)
        self.key_ordered = tuple(self.search_space.keys())
        self._init_detail()

    def _init_detail(self):
        '''
        统计每个参数编码成二进制所需要的位数"choice_len"以及每个参数的可选项数"choice_num，
        uniform类占用10位，可选项数目为0，
        choice的占用位数用可选项个数的二进制占用位数来表示，
        例如choice有5个可选项，因为0b101=5, 所以占用3位

        '''
        # 参数个数
        self.n_fields = len(self.key_ordered)
        # 每个参数占用二进制的位数， uniform占10位， choice用选项个数的二进制位数来表示
        self.choice_len = [0] * self.n_fields
        # 每个参数的可选数目，uniform用0表示，choice用选项的个数表示
        self.choice_num = [0] * self.n_fields
        for i in range(self.n_fields):
            ckey = self.key_ordered[i]
            cvalue = self.search_space[ckey]["_value"]
            self.search_space[ckey]["choice_num"] = len(cvalue)
            self.search_space[ckey]["choice_len"] = len(bin(len(cvalue) - 1)) - 2  # 减去最前面的'0b'占用位数
        self.choice_num = [self.search_space[self.key_ordered[i]]["choice_num"] for i in range(self.n_fields)]
        self.choice_len = [self.search_space[self.key_ordered[i]]["choice_len"] for i in range(self.n_fields)]
        # print(self.search_space['layer2_index']["_value"])  # [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]


    def gen_random_param(self):
        '''
            Randomly generate a set of net params from search space
        '''
        res_gene = []
        cur_layer = 0
        layer2_index_range = self.search_space['layer2_index']["_value"]
        for i in range(2, 18):
            # print('cur layer:', cur_layer)
            temp_candidate=list(range(i+1,18))+[0]
            print('temp candidate:', temp_candidate)
            layer2_choice = np.random.choice(temp_candidate)
            # print('choice:',layer2_choice)
            # layer2_index=layer2_index_range.index(layer2_choice)
            # # print('index:',layer2_index)
            # layer2_gene= list(format(layer2_index, '02b').zfill(self.choice_len[0]))
            # print('gene:',layer2_gene)
            res_gene.append(layer2_choice)

            op1_range=self.search_space['op1']["_value"]
            # print('op1 range:', op1_range)
            op1_choice = np.random.choice(op1_range)
            # print('op1 choice:',op1_choice)
            # op1_index=op1_range.index(op1_choice)
            # # print('op1 index:',op1_index)
            # op1_gene=list(format(op1_index, '02b').zfill(self.choice_len[1]))
            # print('op1 gene:', op1_gene)
            # print('\n')
            res_gene.append(op1_choice)

            op2_range = self.search_space['op2']["_value"]
            # print('op2 range:',op2_range)
            op2_choice = np.random.choice(op2_range)
            # print('op2 choice:', op2_choice)
            # op2_index = op2_range.index(op2_choice)
            # # print('op2 index:', op2_index)
            # op2_gene = list(format(op2_index, '02b').zfill(self.choice_len[2]))
            # print('op2 gene:', op2_gene)
            # print('\n')
            res_gene.append(op2_choice)
            cur_layer+=1

        return res_gene



    def mutate_one_value(self,params):   # mutate one value and accuracy stay the same , for data augmentation
        flag=False
        another_param = []
        block_candidate = []
        for i in range(2, 18):
            temp_candidate = list(range(i + 1, 18)) + [0]
            block_candidate.append(temp_candidate)
        op1_candidate = self.search_space['op1']["_value"]
        op2_candidate = self.search_space['op2']["_value"]
        while not flag:
            mutate_index=random.randint(0,len(params)-1)
            operation=params[mutate_index]
            if mutate_index%3==0:
                print('mutate layer2 index:')
                block_index=mutate_index//3
                candidate=copy.deepcopy(block_candidate)
                candidate[block_index].remove(operation)
                if candidate[block_index]:
                    mutate=random.choice(candidate[block_index])
                    another_param=params[:mutate_index]+[mutate]+params[mutate_index+1:]
                    if self.check_valid(another_param) and another_param!=params:
                        flag=True
                    else:
                        if not self.check_valid(another_param):
                            print('mutate layer2 index not valid!')
                        if another_param==params:
                            print('mutate layer2 index equals to params')
                else:
                    print('no candidate for layer2 index')

            elif mutate_index%3==1:
                print('mutate op1:')
                candi=copy.deepcopy(op1_candidate)
                candi.remove(operation)
                if candi:
                    mutate=random.choice(candi)
                    another_param = params[:mutate_index] + [mutate] + params[mutate_index+1:]
                    if self.check_valid(another_param) and another_param!=params:
                        flag=True
                    else:
                        if not self.check_valid(another_param):
                            print('mutate op1 not valid!')
                        if another_param==params:
                            print('mutate op1 equals to params')
                else:
                    print('no candidate for op1')

            elif mutate_index%3==2:
                print('mutate op2:')
                if mutate_index-2!=0:
                    candid=copy.deepcopy(op2_candidate)
                    candid.remove(operation)
                    if candid:
                        mutate=random.choice(candid)
                        another_param=params[:mutate_index]+[mutate]+params[mutate_index+1:]
                        if self.check_valid(another_param) and another_param!=params:
                            flag = True
                        else:
                            if not self.check_valid(another_param):
                                print('mutate op2 not valid!')
                            if another_param == params:
                                print('mutate op2 equals to params')
                    else:
                        print('no candidate for op2')
                else:
                    print('layer2 index is 0, op2 ignored!')

            else:
                print('error!')

        return another_param



    def decode_gene(self, gene):
        '''
        decode binary gene into net params
        '''
        params = []
        for i in range(0,len(gene), 10):
            block_gene=gene[i:i+10]
            layer2_gene=''.join(str(st) for st in block_gene[:4])
            op1_gene=''.join(str(bl) for bl in block_gene[4:7])
            op2_gene=''.join(str(ge) for ge in block_gene[7:10])
            layer2_index=min(int(layer2_gene, 2), self.search_space['layer2_index']["choice_num"] - 1)
            op1_index = min(int(op1_gene, 2), self.search_space['op1']["choice_num"] - 1)
            op2_index = min(int(op2_gene, 2), self.search_space['op2']["choice_num"] - 1)
            params.append(self.search_space['layer2_index']["_value"][layer2_index])
            params.append(self.search_space['op1']["_value"][op1_index])
            params.append(self.search_space['op2']["_value"][op2_index])

        return params




    def encode_gene(self, net_value):
        '''
        encode net params into binary gene
        '''
        bin_gene = []
        layer2_range = self.search_space['layer2_index']["_value"]
        op1_range = self.search_space['op1']["_value"]
        op2_range = self.search_space['op2']["_value"]

        for i in range(0,len(net_value),3):
            layer2=net_value[i]
            op1=net_value[i+1]
            op2=net_value[i+2]

            layer2_index=layer2_range.index(layer2)
            op1_index=op1_range.index(op1)
            op2_index = op2_range.index(op2)

            bin_layer2_index = list(format(layer2_index, '02b').zfill(self.choice_len[0]))
            bin_op1_index = list(format(op1_index, '02b').zfill(self.choice_len[1]))
            bin_op2_index = list(format(op2_index, '02b').zfill(self.choice_len[2]))

            bin_gene.extend(bin_layer2_index)
            bin_gene.extend(bin_op1_index)
            bin_gene.extend(bin_op2_index)
        return [int(i) for i in bin_gene]





    def check_valid(self,params):
        success=False

        try:
            # net_params = self.decode_gene(params)
            net = MobileNetLike(params)
            x = torch.randn(2, 3, 32, 32)
            y = net(x)
            success = True
        except RuntimeError:
            pass
        except IndexError:
            pass
        return success
    #
    #
    #
    def gen_individual(self):
        success=False

        while not success:
            try:
                params = self.gen_random_param()
                net = MobileNetLike(params)
                x = torch.randn(2, 3, 32, 32)
                y = net(x)
                success=params
            except RuntimeError:
                continue

        print('gen individual params',success)
        # gene = self.encode_gene(success)
        return success
    #


    def gen_popl(self,n):
        res=[]
        for i in range(n):
            success = False
            while not success:
                try:
                    params = self.gen_random_param()
                    net = MobileNetLike(params)
                    x = torch.randn(2, 3, 32, 32)
                    y = net(x)
                    success=params
                    print('gen individual params', success)
                    res.append(success)
                except RuntimeError:
                    continue
        return res


if __name__ == "__main__":
    json_path = "search_space.json"
    coder_ = GeneCoder(json_path)

    # print('choice_len:', coder_.choice_len)
    # print('choice_num:', coder_.choice_num)
    #
    params = coder_.gen_random_param()
    print('randomly generate:')
    print(params)
    # print(len(params))
    #
    # gene = coder_.encode_gene(params)
    # print('encode:')
    # print(gene)
    # print(len(gene))
    #
    # net_params = coder_.decode_gene(gene)
    # print('decode:')
    # print(net_params)
    # print(len(net_params))
    #
    #
    # print(net_params==params)

    # res=coder_.gen_popl(200)
    # state = {
    #     'list': res,
    #
    # }
    # torch.save(state, os.path.join('save', 'gen1.pth'))




