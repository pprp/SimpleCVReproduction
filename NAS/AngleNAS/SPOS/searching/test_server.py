#!/usr/bin/env python3
from mq_server_base import MessageQueueServerBase,MessageQueueClientBase
import argparse
import os, sys
import time
import re
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from config import config
from super_model import SuperNetwork
from train import infer
import numpy as np
import functools
print=functools.partial(print,flush=True)

sys.path.append("../..")
from utils import *

class CrossEntropyLabelSmooth(nn.Module):
  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss

class TorchMonitor(object):
    def __init__(self):
        self.obj_set=set()
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                self.obj_set.add(obj)

    def find_leak_tensor(self):
        for obj in gc.get_objects():
            if torch.is_tensor(obj) and obj not in self.obj_set:
                print(obj.size())


class TestClient(MessageQueueClientBase):
    def __init__(self,*,random):
        if random:
            super().__init__(config.host, config.port, config.username,
                                config.random_test_send_pipe, config.random_test_recv_pipe)
        else:
            super().__init__(config.host, config.port, config.username,
                                config.test_send_pipe, config.test_recv_pipe)

    def send(self,cand):
        assert isinstance(cand,tuple)
        return super().send(cand)

class TestServer(MessageQueueServerBase):
    def __init__(self, batchsize, train_dir, val_dir, *,random):
        if random:
            super().__init__(config.host, config.port, config.username, 
                            config.random_test_send_pipe, config.random_test_recv_pipe)
        else:
            super().__init__(config.host, config.port, config.username, 
                            config.test_send_pipe, config.test_recv_pipe)
        self.model = None
        self.criterion = CrossEntropyLabelSmooth(1000, 0.1)
        self.criterion = self.criterion.cuda()

        # Prepare data
        train_loader = get_train_dataloader(train_dir, batchsize, 0, 100000)
        self.train_dataprovider = DataIterator(train_loader)
        val_loader = get_val_dataloader(val_dir)
        self.val_dataprovider = DataIterator(val_loader)

    def eval(self, cand):
        print('cand={}'.format(cand))
        self.model = SuperNetwork().cuda()
        assert(os.path.exists(config.net_cache))
        load(self.model, config.net_cache)
        res = self._test_candidate(cand)
        return res

    def _test_candidate(self, cand):
        res = dict() 
        try:
            t0 = time.time()
            print('starting inference...')
            Top1_acc = self._inference(np.array(cand).astype(np.int))
            print('time: {}s'.format(time.time() - t0))
            res = {'status': 'success', 'acc': Top1_acc}
            return res
        except:
            import traceback
            traceback.print_exc()
            os._exit(1)
            res['status'] = 'failure'
            return res
    
    def _inference(self, cand):
        t0 = time.time()
        print('testing model {} ..........'.format(cand))
        recalculate_bn(self.model, cand, self.train_dataprovider)
        torch.cuda.empty_cache()
        recal_bn_time = time.time() - t0
        test_top1_acc, _ = infer(self.val_dataprovider, self.model, self.criterion, cand)
        testtime = time.time() - t0
        print('|=> valid: accuracy = {:.3f}%, total_test_time = {:.2f}s, recal_bn_time={:.2f}s, cand = {}'.format(test_top1_acc, testtime, recal_bn_time, cand))
        return test_top1_acc


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=512)
    parser.add_argument('-p', '--process', type=int, default=1)
    parser.add_argument('-r', '--reset', action='store_true')
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--train_dir', type=str, default='/media/niu/niu_g/data/imagenet/train', help='path to training dataset')
    parser.add_argument('--test_dir', type=str, default='/media/niu/niu_g/data/imagenet/val', help='path to test dataset')
    args=parser.parse_args()
    train_server = TestServer(args.batch_size, args.train_dir, args.test_dir,random=args.random)
    train_server.run(args.process, reset_pipe=args.reset)
  
if __name__ == "__main__":
    try:
        main()
    except:
        import traceback
        traceback.print_exc()
        print(flush=True)
        os._exit(1)
