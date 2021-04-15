import argparse
import copy
import functools
import os
import pickle
import random
import shutil
import sys
import time
import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from datasets.cifar100_dataset import get_train_loader, get_val_loader
from model.slimmable_resnet20 import mutableResNet20
from utils.utils import (ArchLoader, AvgrageMeter, CrossEntropyLabelSmooth,
                         DataIterator, accuracy, bn_calibration_init,
                         reduce_tensor, save_checkpoint, retrain_bn)

print = functools.partial(print, flush=True)

CIFAR100_TRAINING_SET_SIZE = 50000
CIFAR100_TEST_SET_SIZE = 10000

parser = argparse.ArgumentParser("ResNet20-cifar100")
parser.add_argument('--proxy', type=float, default=1,
                    help='smaller dataset ')
parser.add_argument('--local_rank', type=int, default=None,
                    help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=16384, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.653, help='init learning rate')  # sqrt(16384/3/128)*0.1
parser.add_argument('--num_workers', type=int,
                    default=6, help='num of workers')

# hyper parameter
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=4e-5, help='weight decay')

parser.add_argument('--report_freq', type=float,
                    default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=30000,
                    help='num of training epochs')
parser.add_argument('--total_iters', type=int,
                    default=300000, help='total iters')

parser.add_argument('--classes', type=int, default=100,
                    help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--label_smooth', type=float,
                    default=0.1, help='label smoothing')
args = parser.parse_args()

if args.proxy > 0:
    CIFAR100_TRAINING_SET_SIZE = int(CIFAR100_TRAINING_SET_SIZE * args.proxy)
    CIFAR100_TEST_SET_SIZE = int(CIFAR100_TEST_SET_SIZE * args.proxy)


per_epoch_iters = CIFAR100_TRAINING_SET_SIZE // args.batch_size
val_iters = CIFAR100_TEST_SET_SIZE // 200


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    writer = None
    num_gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    if args.local_rank == 0:
        args.exp = datetime.datetime.now().strftime("%YY_%mM_%dD_%HH") + "_" + \
            "{:04d}".format(random.randint(0, 1000))

    print('gpu device = %d' % args.gpu)
    print("args = %s", args)

    torch.distributed.init_process_group(
        backend='nccl', init_method='env://')

    args.world_size = torch.distributed.get_world_size()
    args.batch_size = args.batch_size // args.world_size

    criterion_smooth = CrossEntropyLabelSmooth(args.classes, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    model = mutableResNet20()

    model = model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

    # all_parameters = model.parameters()
    # weight_parameters = []
    # for pname, p in model.named_parameters():
    #     if p.ndimension() == 4 or 'classifier.0.weight' in pname or 'classifier.0.bias' in pname:
    #         weight_parameters.append(p)
    # weight_parameters_id = list(map(id, weight_parameters))
    # other_parameters = list(
    #     filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    # optimizer = torch.optim.SGD(
    #     [{'params': other_parameters},
    #      {'params': weight_parameters, 'weight_decay': args.weight_decay}],
    #     args.learning_rate,
    #     momentum=args.momentum,
    # )

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # // 16  # 16 代表是每个子网的个数
    args.total_iters = args.epochs * per_epoch_iters

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lambda step: (1.0-step/args.total_iters), last_epoch=-1)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5)

    if args.local_rank == 0:
        writer = SummaryWriter("./runs/%s-%05d" %
                               (time.strftime("%m-%d", time.localtime()), random.randint(0, 100)))

    # Prepare data
    train_loader = get_train_loader(
        args.batch_size, args.local_rank, args.num_workers, args.total_iters, args.proxy)
    train_dataprovider = DataIterator(train_loader)
    # 原来跟train batch size一样，现在修改小一点 ， 同时修改val_iters
    val_loader = get_val_loader(
        min(args.batch_size, CIFAR100_TEST_SET_SIZE), args.num_workers, args.proxy)
    val_dataprovider = DataIterator(val_loader)

    archloader = ArchLoader("data/Track1_final_archs.json")

    # niu 实验 将验证集替换为训练集， 加速
    train(train_dataprovider, val_dataprovider, optimizer, scheduler,
          model, archloader, criterion_smooth, args, val_iters, args.seed, writer)
    # train(train_dataprovider, val_dataprovider, optimizer, scheduler,
    #       model, archloader, criterion_smooth, args, val_iters, args.seed, writer)


def train(train_dataprovider, val_dataprovider, optimizer, scheduler, model, archloader, criterion, args, val_iters, seed, writer=None):
    objs, top1 = AvgrageMeter(), AvgrageMeter()

    for p in model.parameters():
        p.grad = torch.zeros_like(p)

    for step in range(args.total_iters):
        model.train()
        t0 = time.time()
        image, target = train_dataprovider.next()
        datatime = time.time() - t0
        n = image.size(0)
        optimizer.zero_grad()
        image = Variable(image, requires_grad=False).cuda(args.gpu)
        target = Variable(target, requires_grad=False).cuda(args.gpu)

        # Fair Sampling
        fair_arc_list = archloader.generate_niu_fair_batch(step)

        for arc in fair_arc_list:
            logits = model(image, archloader.convert_list_arc_str(arc))
            loss = criterion(logits, target)
            loss_reduce = reduce_tensor(loss, 0, args.world_size)
            loss.backward()

        nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        prec1, _ = accuracy(logits, target, topk=(1, 5))
        objs.update(loss_reduce.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0 and args.local_rank == 0:
            now = time.strftime('%Y-%m-%d %H:%M:%S',
                                time.localtime(time.time()))
            print('{} |=> train: {} / {}, lr={}, loss={:.2f}, acc={:.2f}, datatime={:.2f}, seed={}'
                  .format(now, step, args.total_iters, scheduler.get_lr()[0], objs.avg, top1.avg, float(datatime), seed))

        if args.local_rank == 0 and step % 5 == 0 and writer is not None:
            writer.add_scalar("Train/loss", objs.avg, step)
            writer.add_scalar("Train/acc1", top1.avg, step)

        if args.local_rank == 0 and step % args.report_freq == 0:
            # model

            top1_val, objs_val = infer(train_dataprovider, val_dataprovider, model.module, criterion,
                                       fair_arc_list, val_iters, archloader)

            if writer is not None:
                writer.add_scalar("Val/loss", objs_val, step)
                writer.add_scalar("Val/acc1", top1_val, step)

            save_checkpoint(
                {'state_dict': model.state_dict(), }, step, args.exp)


def infer(train_dataprovider, val_dataprovider, model, criterion, fair_arc_list, val_iters, archloader):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    model.eval()
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    print('{} |=> Test rng = {}'.format(now, fair_arc_list[-1]))  # 只测试最后一个模型

    # BN calibration
    retrain_bn(model, 15, train_dataprovider,
               archloader.convert_list_arc_str(fair_arc_list[-1]), device=0)

    with torch.no_grad():
        for step in range(val_iters):
            t0 = time.time()
            image, target = val_dataprovider.next()
            datatime = time.time() - t0
            image = Variable(image, requires_grad=False).cuda()
            target = Variable(target, requires_grad=False).cuda()

            logits = model(
                image, archloader.convert_list_arc_str(fair_arc_list[-1]))
            loss = criterion(logits, target)
            prec1, _ = accuracy(logits, target, topk=(1, 5))
            n = image.size(0)
            objs.update(loss.data.item(), n)
            top1.update(prec1.data.item(), n)

            # for arc in fair_arc_list:
            #     logits = model(image, archloader.convert_list_arc_str(arc))
            #     loss = criterion(logits, target)
            #     prec1, _ = accuracy(logits, target, topk=(1, 5))
            #     n = image.size(0)
            #     objs.update(loss.data.item(), n)
            #     top1.update(prec1.data.item(), n)

        now = time.strftime('%Y-%m-%d %H:%M:%S',
                            time.localtime(time.time()))
        print('{} |=> valid: step={}, loss={:.2f}, acc={:.2f}, datatime={:.2f}'.format(
            now, step, objs.avg, top1.avg, datatime))

    return top1.avg, objs.avg


if __name__ == '__main__':
    main()
