from utils import *
import sys
import os
import torch
from torch import nn
from torch.autograd import Variable
import time
import numpy as np
from config import config
import copy
import functools
print = functools.partial(print, flush=True)

sys.path.append("../..")


def train(train_dataprovider, optimizer, scheduler, model, criterion, operations, iters, train_iters, seed, args):
    objs, top1 = AvgrageMeter(), AvgrageMeter()
    for step in range(train_iters):
        model.train()
        if scheduler.get_lr()[0] > args.min_lr:
            scheduler.step()
        t0 = time.time()
        image, target = train_dataprovider.next()
        datatime = time.time() - t0
        image = Variable(image, requires_grad=False).cuda(args.gpu)
        target = Variable(target, requires_grad=False).cuda(args.gpu)
        n = image.size(0)
        optimizer.zero_grad()

        # Uniform Sampling
        rng = []
        for i, ops in enumerate(operations):
            np.random.seed(seed)
            k = np.random.randint(len(ops))
            select_op = ops[k]
            seed += 1
            rng.append(select_op)
            
        logits = model(image, rng)
        loss = criterion(logits, target)
        loss.backward()

        # Fix a bug of pytorch
        for group in optimizer.param_groups:
            for p in group['params']:
                if not p.grad is None and torch.sum(torch.abs(p.grad.data)) == 0.0:
                    p.grad = None

        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, _ = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0 and args.local_rank == 0:
            now = time.strftime('%Y-%m-%d %H:%M:%S',
                                time.localtime(time.time()))
            print('{} |=> Iters={}, train:  {} / {}, loss={:.2f}, acc={:.2f}, lr={}, datatime={:.2f}, seed={}'
                  .format(now, iters+1, step, train_iters, objs.avg, top1.avg, scheduler.get_lr()[0], float(datatime), seed))
    return seed
