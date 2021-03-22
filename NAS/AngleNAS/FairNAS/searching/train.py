import os
import torch
from torch import nn
from torch.optim import lr_scheduler
import torch.optim as optim
from torch.autograd import Variable
import time
import numpy as np
from config import config
import copy
import random
import functools
print=functools.partial(print,flush=True)

import sys
sys.path.append("../..")
from utils import *

def train(train_dataprovider, val_dataprovider, optimizer, scheduler, model, criterion, args, val_iters, seed, operations):
    objs, top1 = AvgrageMeter(), AvgrageMeter()
    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    for step in range(args.total_iters):
        model.train()
        scheduler.step()
        t0 = time.time()
        image, target = train_dataprovider.next()
        datatime = time.time() - t0
        n = image.size(0)
        optimizer.zero_grad()
        image = Variable(image, requires_grad=False).cuda(args.gpu)
        target = Variable(target, requires_grad=False).cuda(args.gpu)

        # Fair Sampling
        rngs = []
        for i in range(len(operations)):
            seed += 1
            random.seed(seed)
            rngs.append(random.sample(operations[i], len(operations[i])))
        rngs = np.transpose(rngs)

        for rng in rngs:
            logits = model(image, rng)
            loss = criterion(logits, target)
            loss.backward()

        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        prec1, _ = accuracy(logits, target, topk=(1, 5))
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0 and args.local_rank == 0:
            now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print('{} |=> train: {} / {}, lr={}, loss={:.2f}, acc={:.2f}, datatime={:.2f}, seed={}'\
                .format(now, step, args.total_iters, scheduler.get_lr()[0], objs.avg, top1.avg, float(datatime), seed))

    if args.local_rank == 0:
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} |=> Test rng = {}'.format(now, rng))
        infer(val_dataprovider, model.module, criterion, rng, val_iters)
    
def infer(val_dataprovider, model, criterion, rngs, val_iters=250):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step in range(val_iters):
            t0 = time.time()
            image, target = val_dataprovider.next()
            datatime = time.time() - t0
            image = Variable(image, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()
            logits = model(image, rngs)
            loss = criterion(logits, target)

            prec1, _ = accuracy(logits, target, topk=(1, 5))
            n = image.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        print('{} |=> valid: step={}, loss={:.2f}, acc={:.2f}, datatime={:.2f}'.format(now, step, objs.avg, top1.avg, datatime))
        
    return top1.avg, objs.avg