# Universally Slimmable Networks and Improved Training Techniques
import argparse
import copy
import datetime
import functools
import os
import pickle
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils.scheduler import GradualWarmupScheduler

from datasets.cifar100_dataset import get_train_loader, get_val_loader
from model.slimmable_resnet20 import mutableResNet20
from model.dynamic_resnet20 import dynamic_resnet20
from model.resnet20 import resnet20
from model.independent_resnet20 import Independent_resnet20
from utils.utils import (ArchLoader, AvgrageMeter, CrossEntropyLabelSmooth,
                         DataIterator, accuracy, bn_calibration_init,
                         reduce_mean, reduce_tensor, retrain_bn,
                         save_checkpoint, CrossEntropyLossSoft)

print = functools.partial(print, flush=True)

CIFAR100_TRAINING_SET_SIZE = 50000
CIFAR100_TEST_SET_SIZE = 10000

parser = argparse.ArgumentParser("ResNet20-cifar100")

parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=4096,
                    help='batch size')  # 8192
parser.add_argument('--learning_rate', type=float,
                    default=0.5656, help='init learning rate')  # 0.8
parser.add_argument('--num_workers', type=int,
                    default=3, help='num of workers')
parser.add_argument('--model-type', type=str, default="dynamic",
                    help="type of model(dynamic independent slimmable original)")

# hyper parameter
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=5e-4, help='weight decay')

parser.add_argument('--report_freq', type=float,
                    default=5, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=1500,
                    help='num of training epochs')

parser.add_argument('--classes', type=int, default=100,
                    help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--label_smooth', type=float,
                    default=0.1, help='label smoothing')
args = parser.parse_args()

val_iters = CIFAR100_TEST_SET_SIZE // 200


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    writer = None
    num_gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    args.nprocs = num_gpus
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

    if args.model_type == "dynamic":
        model = dynamic_resnet20()
    elif args.model_type == "independent":
        model = Independent_resnet20()
    elif args.model_type == "slimmable":
        model = mutableResNet20()
    elif args.model_type == "original":
        model = resnet20()
    else:
        print("Not Implement")

    # model = resnet20()
    model = model.cuda(args.gpu)

    if num_gpus > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size

    # criterion_smooth = CrossEntropyLabelSmooth(args.classes, args.label_smooth)
    # criterion_smooth = criterion_smooth.cuda()
    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    soft_criterion = CrossEntropyLossSoft()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lambda step: (1.0-step/args.total_iters), last_epoch=-1)
    # a_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #     optimizer, T_0=5)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    # a_scheduler = torch.optim.lr_scheduler.LambdaLR(
    #     optimizer, lambda epoch: 1 - (epoch / args.epochs))
    a_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=[500, 750], last_epoch=-1)  # !!
    scheduler = GradualWarmupScheduler(
        optimizer, 1, total_epoch=5, after_scheduler=a_scheduler)

    if args.local_rank == 0:
        writer = SummaryWriter("./runs/%s-%05d" %
                               (time.strftime("%m-%d", time.localtime()), random.randint(0, 100)))

    # Prepare data
    train_loader = get_train_loader(
        args.batch_size, args.local_rank, args.num_workers)
    # 原来跟train batch size一样，现在修改小一点 ，
    val_loader = get_val_loader(args.batch_size, args.num_workers)

    archloader = ArchLoader("data/Track1_final_archs.json")

    for epoch in range(args.epochs):
        train(train_loader, val_loader,  optimizer, scheduler, model,
              archloader, criterion, soft_criterion, args, args.seed, epoch, writer)

        scheduler.step()
        if (epoch + 1) % args.report_freq == 0:
            top1_val, top5_val,  objs_val = infer(train_loader, val_loader, model, criterion,
                                                  archloader, args, epoch)

            if args.local_rank == 0:
                # model
                if writer is not None:
                    writer.add_scalar("Val/loss", objs_val, epoch)
                    writer.add_scalar("Val/acc1", top1_val, epoch)
                    writer.add_scalar("Val/acc5", top5_val, epoch)

                save_checkpoint(
                    {'state_dict': model.state_dict(), }, epoch, args.exp)


def train(train_dataloader, val_dataloader, optimizer, scheduler, model, archloader, criterion, soft_criterion, args, seed, epoch, writer=None):
    losses_, top1_, top5_ = AvgrageMeter(), AvgrageMeter(), AvgrageMeter()

    model.train()
    widest = [16, 16, 16, 16, 16, 16, 16, 32, 32,
              32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]
    narrowest = [4,  4,  4, 4,  4,  4,  4,  4, 4,
                 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    train_loader = tqdm(train_dataloader)
    train_loader.set_description(
        '[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_last_lr()[0]))
    for step, (image, target) in enumerate(train_loader):
        n = image.size(0)
        image = Variable(image, requires_grad=False).cuda(
            args.gpu, non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(
            args.gpu, non_blocking=True)

        if args.model_type in ["dynamic", "independent", "slimmable"]:
            # sandwich rule
            candidate_list = []
            candidate_list += [narrowest]
            candidate_list += [archloader.generate_spos_like_batch().tolist()
                               for i in range(6)]

            # archloader.generate_niu_fair_batch(step)
            # 全模型来一遍
            soft_target = model(image, widest)
            soft_loss = criterion(soft_target, target)
            soft_loss.backward()
            soft_target = torch.nn.functional.softmax(
                soft_target, dim=1).detach()

            # 采样几个子网来一遍
            for arc in candidate_list:
                logits = model(image, arc)
                # loss = soft_criterion(logits, soft_target.cuda(
                #     args.gpu, non_blocking=True))
                loss = criterion(logits, target)

                # loss_reduce = reduce_tensor(loss, 0, args.world_size)
                loss.backward()
        elif args.model_type == "original":
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))

        if torch.cuda.device_count() > 1:
            torch.distributed.barrier()

            loss = reduce_mean(loss, args.nprocs)
            prec1 = reduce_mean(prec1, args.nprocs)
            prec5 = reduce_mean(prec5, args.nprocs)

        optimizer.step()
        optimizer.zero_grad()

        losses_.update(loss.data.item(), n)
        top1_.update(prec1.data.item(), n)
        top5_.update(prec1.data.item(), n)

        postfix = {'train_loss': '%.6f' % (
            losses_.avg), 'train_acc1': '%.6f' % top1_.avg, 'train_acc5': '%.6f' % top5_.avg}

        train_loader.set_postfix(log=postfix)

        if args.local_rank == 0 and step % 10 == 0 and writer is not None:
            writer.add_scalar("Train/loss", losses_.avg, step +
                              len(train_dataloader) * epoch * args.batch_size)
            writer.add_scalar("Train/acc1", top1_.avg, step +
                              len(train_dataloader) * epoch * args.batch_size)
            writer.add_scalar("Train/acc5", top5_.avg, step +
                              len(train_loader)*args.batch_size*epoch)


def infer(train_loader, val_loader, model, criterion,  archloader, args, epoch):
    objs_, top1_, top5_ = AvgrageMeter(), AvgrageMeter(), AvgrageMeter()

    model.eval()
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    # [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]
    # .generate_width_to_narrow(epoch, args.epochs)
    fair_arc_list = archloader.generate_spos_like_batch().tolist()

    print('{} |=> Test rng = {}'.format(now, fair_arc_list))  # 只测试最后一个模型

    if args.model_type == "dynamic":
        # BN calibration
        retrain_bn(model, train_loader, fair_arc_list, device=0)

    with torch.no_grad():
        for step, (image, target) in enumerate(val_loader):
            t0 = time.time()
            datatime = time.time() - t0
            image = Variable(image, requires_grad=False).cuda(
                args.local_rank, non_blocking=True)
            target = Variable(target, requires_grad=False).cuda(
                args.local_rank, non_blocking=True)

            logits = model(image, fair_arc_list)
            loss = criterion(logits, target)

            top1, top5 = accuracy(logits, target, topk=(1, 5))

            if torch.cuda.device_count() > 1:
                torch.distributed.barrier()
                loss = reduce_mean(loss, args.nprocs)
                top1 = reduce_mean(top1, image.size(0))
                top5 = reduce_mean(top5, image.size(0))

            n = image.size(0)
            objs_.update(loss.data.item(), n)
            top1_.update(top1.data.item(), n)
            top5_.update(top5.data.item(), n)

        now = time.strftime('%Y-%m-%d %H:%M:%S',
                            time.localtime(time.time()))
        print('{} |=> valid: step={}, loss={:.2f}, val_acc1={:.2f}, val_acc5={:2f}, datatime={:.2f}'.format(
            now, step, objs_.avg, top1_.avg, top5_.avg,  datatime))

    return top1_.avg, top5_.avg, objs_.avg


if __name__ == '__main__':
    main()
