import os
import torch
from torch import nn
from torch.autograd import Variable
import time
import numpy as np
from config import config
import copy
import random
import functools
print=functools.partial(print,flush=True)
from super_model import Network_ImageNet
import apex

import sys
sys.path.append("../..")
from utils import *

def train(train_dataprovider, optimizer, scheduler, model, criterion, operations, iters, train_iters, seed, args):
    objs, top1 = AvgrageMeter(), AvgrageMeter()
    for step in range(train_iters):
        model.train()
        if scheduler.get_lr()[0] > args.min_lr:
            scheduler.step()
        t0 = time.time()
        image, target = train_dataprovider.next()
        datatime = time.time() - t0
        n = image.size(0)
        optimizer.zero_grad()
        image = Variable(image, requires_grad=False).cuda(args.gpu)
        target = Variable(target, requires_grad=False).cuda(args.gpu)

        # Uniform Sampling
        rng, cell = [], []
        for op in operations:
            np.random.seed(seed)
            k = np.random.randint(len(op))
            select_op = op[k]
            cell.append(select_op)
            seed+=1

        for _ in range(len(model.module.cells)):
            rng.append(copy.deepcopy(cell))
        
        # Make sure each node has only two predecessor nodes
        rng = check_cand(rng, operations, config.edges)
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
            now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print('{} |=> Iters={}, train: {} / {}, loss={:.2f}, acc={:.2f}, lr={}, datatime={:.2f}, seed={}'\
                .format(now, iters+1, step, train_iters, objs.avg, top1.avg, scheduler.get_lr()[0], float(datatime), seed)) 
    return seed

def get_warmup_model(train_dataprovider, criterion, operations, per_epoch_iters, seed, args):
    model = Network_ImageNet(args.init_channels, args.classes, 3).cuda(args.gpu)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    per_stage_iters = per_epoch_iters * config.warmup_epochs

    # Warmup supernet for some epochs by progressively inserting layers, due to its low convergence on DARTS search space 
    for i in range(config.insert_layers):
        optimizer, scheduler = get_optimizer_schedule(model, args, per_stage_iters)
        seed = train(train_dataprovider, optimizer, scheduler, model, criterion, \
                operations, -1, per_stage_iters, seed, args)
        model.module.insert_layer()

    if args.local_rank == 0:
        torch.save(model.module.state_dict(), 'warmup_model.pth.tar')
    return model, seed