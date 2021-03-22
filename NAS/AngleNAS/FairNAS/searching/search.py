import os
import time
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from super_model import SuperNetwork

from torch.autograd import Variable
from config import config
from test_server import TestClient
import sys
sys.setrecursionlimit(10000)

import copy
import functools
print=functools.partial(print,flush=True)

choice=lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))

class EvolutionTrainer(object):
    def __init__(self,random=False):
        self.max_epochs=20
        self.select_num=10
        self.population_num=50
        self.m_prob=0.5
        self.crossover_num=25
        self.mutation_num=25
        self.op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))

        self.tester = TestClient(random=random)
        self.tester.connect()
        self.reset()

    def reset(self):
        self.memory=[]
        self.keep_top_k = {self.select_num:[],50:[]}
        self.epoch=0
        self.candidates=[]
        self.vis_dict = {}

    def get_arch_flops(self, cand):
        assert len(cand) == len(config.backbone_info) - 2
        preprocessing_flops = self.op_flops_dict['PreProcessing'][config.backbone_info[0]]
        postprocessing_flops = self.op_flops_dict['PostProcessing'][config.backbone_info[-1]]
        total_flops = preprocessing_flops + postprocessing_flops
        for i in range(len(cand)):
            inp, oup, img_h, img_w, stride = config.backbone_info[i+1]
            op_id = cand[i]
            if op_id >= 0:
                key = config.blocks_keys[op_id]
                total_flops += self.op_flops_dict[key][(inp, oup, img_h, img_w, stride)]
        return total_flops, -1

    def legal(self,cand):
        if len(cand) == 0:
            return False    
        assert isinstance(cand,tuple)
        if cand not in self.vis_dict:
            self.vis_dict[cand]={}
        info=self.vis_dict[cand]
        if 'visited' in info:
            return False
        flops = None
        if config.limit_flops:
            if 'flops' not in info:
                info['flops'], info['params']= self.get_arch_flops(cand)
            flops = info['flops']
            print('flops:{}'.format(flops))
            if config.max_flops is not None and flops > config.max_flops:
                return False
            if config.min_flops is not None and flops < config.min_flops:
                return False

        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} cand = {} flops = {}'.format(now, cand, flops))
        info['visited']=True
        self.vis_dict[cand]=info
        return True

    def update_top_k(self,candidates,*,k,key,reverse=False):
        assert k in self.keep_top_k
        print('select acc topk......')
        t=self.keep_top_k[k]
        t+=candidates
        t.sort(key=key,reverse=reverse)
        k_ = min(k, 50)
        self.keep_top_k[k]=t[:k_]
    
    def get_topk(self):
        if len(self.keep_top_k[self.select_num]) < 1:
            return None
        topks = []
        for i in range(config.topk):
            topks.append(list(self.keep_top_k[self.select_num][i]))
            print('topk={}'.format(self.keep_top_k[self.select_num]))
        return topks

    def sync_candidates(self, t=300):
        while True:
            ok=True
            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'acc' in info:
                    continue
                ok=False
                if 'test_key' not in info:
                    content = list(cand)
                    now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                    print('{} Sending cand: {}'.format(now, content))
                    info['test_key']=self.tester.send(tuple(content))
            time.sleep(5)
            timeout=t
            for cand in self.candidates:
                info=self.vis_dict[cand]
                if 'acc' in info:
                    continue
                key=info['test_key']
                try:
                    print('try to get {}'.format(key))
                    res=self.tester.get(key,timeout=timeout)
                    if res is not None:
                        print('status : {}'.format(res['status']))
                        info.pop('test_key')
                        if 'acc' in res:
                            info['acc']=res['acc']
                            info['err']=100-info['acc']
                except:
                    import traceback
                    traceback.print_exc()
                    time.sleep(1)
                    info.pop('test_key')
            time.sleep(1)
            if ok:
                break

    def stack_random_cand(self,random_func,*,batchsize=10):
        while True:
            cands=[random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                info=self.vis_dict[cand]
            for cand in cands:
                yield cand
                
    def get_random(self,num):
        print('random select ........')
        def get_random_cand():
            rng = []
            for i, ops in enumerate(self.operations):
                k = np.random.randint(len(ops))
                select_op = ops[k]
                rng.append(select_op)
            return tuple(rng)

        cand_iter=self.stack_random_cand(get_random_cand)
        max_iters = num*10000
        while len(self.candidates)<num and max_iters>0: 
            max_iters-=1
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            self.candidates.append(cand)
            print('random {}/{}'.format(len(self.candidates),num))

        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} random_num = {}'.format(now, len(self.candidates)))

    def get_mutation(self,k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num*10000

        def random_func():
            cand = []
            if len(self.keep_top_k[self.select_num]) > 0:
                cand=list(choice(self.keep_top_k[self.select_num]))
            for i in range(len(cand)):
                if np.random.random_sample()<m_prob:
                    k = np.random.randint(len(self.operations[i]))
                    cand[i]=self.operations[i][k]
            return tuple(list(cand))

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            max_iters-=1
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            print('mutation {}/{} cand={}'.format(len(res),mutation_num,cand))
    
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} mutation_num = {}'.format(now, len(res)))
        return res

    def get_crossover(self,k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        iter = 0
        max_iters = 10000 * crossover_num
        def random_func():
            if len(self.keep_top_k[self.select_num]) > 0:
                cand1=choice(self.keep_top_k[self.select_num])
                cand2=choice(self.keep_top_k[self.select_num])
                cand = [choice([i,j]) for i,j in zip(cand1,cand2)]
                return tuple(cand)
            else:
                return tuple([])

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<crossover_num and max_iters>0:
            max_iters-=1
            cand=next(cand_iter)
            if not self.legal(cand):
                continue
            res.append(cand)
            print('crossover {}/{} cand={}'.format(len(res),crossover_num,cand))

        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} crossover_num = {}'.format(now, len(res)))
        return res

    def search(self, operations):
        self.operations = operations
        self.model = SuperNetwork()
        self.layer = len(self.model.features)
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} layer = {} population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'\
            .format(now, self.layer, self.population_num, self.select_num, self.mutation_num, self.crossover_num, self.population_num - \
                self.mutation_num - self.crossover_num, self.max_epochs))
        self.get_random(self.population_num)

        while self.epoch<self.max_epochs and len(self.candidates) > 0:
            print('epoch = {}'.format(self.epoch))
            self.sync_candidates()
            print('sync finish')

            self.update_top_k(self.candidates,k=self.select_num,key=lambda x:self.vis_dict[x]['acc'],reverse=True)
            self.update_top_k(self.candidates,k=50,key=lambda x:self.vis_dict[x]['acc'],reverse=True)

            now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print('{} epoch = {} : top {} result'.format(now, self.epoch, len(self.keep_top_k[50])))
            for i,cand in enumerate(self.keep_top_k[50]):
                if 'flops' in self.vis_dict[cand]:
                    flops = self.vis_dict[cand]['flops']
                print('No.{} cand={} Top-1 acc = {} flops = {:.2f}M'.format(i+1, cand, self.vis_dict[cand]['acc'], flops/1e6))

            crossover = self.get_crossover(self.select_num, self.crossover_num)
            mutation = self.get_mutation(self.select_num, self.population_num-len(crossover), self.m_prob)
            self.candidates = mutation+crossover
            self.get_random(self.population_num)
            self.epoch+=1
        topks = self.get_topk()
        self.reset()
        print('finish!')
        return topks
