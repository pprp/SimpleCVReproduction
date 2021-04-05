import numpy as np
import random
from nni.tuner import Tuner
from coder import GeneCoder
from deap import base, creator, tools, algorithms
import torch





class MyTuner(Tuner):

    def __init__(self, gen1_path):

        self.gen1_list = torch.load(gen1_path)['list']

    def update_search_space(self, search_space):
        pass

    def generate_parameters(self, parameter_id, **kwargs):
        '''
                返回 Trial 的超参组合的序列化对象
                parameter_id: int
                '''
        '''generate
        '''
        net_param = self.gen1_list.pop(0)
        return net_param

    def receive_trial_result(self, parameter_id, parameters, value, **kwargs):
        '''
                接收 Trial 的最终结果。
                parameter_id: int
                parameters: 'generate_parameters()' 创建出的对象
                value: Trial 的最终指标，包括 default 指标
                '''
        '''receive
        '''
        print("\n***********\nreceive_trial_result")

        print('parameter_id:',parameter_id)
        print('parameters:',parameters)
        print('acc:',value)
        print("***********\n")






