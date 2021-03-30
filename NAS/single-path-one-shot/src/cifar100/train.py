import argparse
import json
import logging
import os
import sys
import time

import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from cifar100_dataset import get_dataset
from slimmable_resnet20 import max_arc_rep, mutableResNet20
from utils import (ArchLoader, AvgrageMeter, CrossEntropyLabelSmooth, accuracy,
                   get_lastest_model, get_parameters, save_checkpoint)


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def get_args():
    parser = argparse.ArgumentParser("ResNet20-Cifar100-oneshot")
    parser.add_argument('--warmup', default=200, type=int,
                        help="warmup weight of the whole channels")
    parser.add_argument('--total-iters', default=350, type=int)

    parser.add_argument('--num_workers', default=12, type=int)
    parser.add_argument(
        '--path', default="Track1_final_archs.json", help="path for json arch files")
    parser.add_argument('--batch-size', type=int,
                        default=2048, help='batch size')
    parser.add_argument('--learning-rate', type=float,
                        default=0.2, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float,
                        default=4e-5, help='weight decay')
    parser.add_argument('--label-smooth', type=float,
                        default=0.1, help='label smoothing')

    parser.add_argument('--save', type=str, default='./models',
                        help='path for saving trained models')
    parser.add_argument('--display-interval', type=int,
                        default=100, help='report frequency')
    parser.add_argument('--save-interval', type=int,
                        default=100, help='report frequency')
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str,
                        default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--auto-continue', type=bool,
                        default=False, help='report frequency')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # archLoader
    arch_loader = ArchLoader(args.path)

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m-%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join(
        'log/train-{}-{:02}-{:.3f}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    train_dataset, val_dataset = get_dataset('cifar100')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=200, shuffle=False,
                                             num_workers=args.num_workers, pin_memory=True)

    model = mutableResNet20()

    logging.info('load model successfully')

    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)

    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                  lambda step: (1.0-step/args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=200)

    model = model.to(device)

    all_iters = 0

    if args.auto_continue:  # 自动进行？？
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(
                lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            logging.info('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    # 参数设置
    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_loader = train_loader
    args.val_loader = val_loader

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(
                args.eval_resume, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint, strict=True)
            validate(model, device, args, all_iters=all_iters,
                     arch_loader=arch_loader)
        exit(0)

    # warmup weights
    # if args.warmup is not None:
    #     logging.info("begin warmup weights")
    #     while all_iters < args.warmup:
    #         all_iters = train_supernet(model, device, args, bn_process=False, all_iters=all_iters)

    validate(model, device, args, all_iters=all_iters, arch_loader=arch_loader)

    while all_iters < args.total_iters:
        all_iters = train_subnet(model, device, args, bn_process=False,
                                 all_iters=all_iters, arch_loader=arch_loader)
        logging.info("validate iter {}".format(all_iters))

    validate(model, device, args, all_iters=all_iters,
             arch_loader=arch_loader)

    # all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)
    # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters


def train_supernet(model, device, args, *, bn_process=True, all_iters=None):
    logging.info("start warmup training...")

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_loader = args.train_loader

    t1 = time.time()

    Top1_err, Top5_err = 0.0, 0.0

    model.train()

    if bn_process:
        adjust_bn_momentum(model, all_iters)

    all_iters += 1
    d_st = time.time()

    for ii, (data, target) in enumerate(train_loader):

        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st

        optimizer.zero_grad()

        # 一个批次
        output = model(data, max_arc_rep)
        loss = loss_function(output, target)

        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        if ii % 10 == 0:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            logging.info("warmup batch acc1: {:.6f}".format(
                acc1.item() / 100))

        optimizer.step()
        scheduler.step()

    prec1, prec5 = accuracy(output, target, topk=(1, 5))

    Top1_err += 1 - prec1.item() / 100
    Top5_err += 1 - prec5.item() / 100

    if True:
        printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                    'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                    'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                    'data_time = {:.6f},\ttrain_time = {:.6f}'.format(
                        data_time, (time.time() - t1) / args.display_interval)

        logging.info(printInfo)
        t1 = time.time()
        Top1_err, Top5_err = 0.0, 0.0

    if all_iters % args.save_interval == 0:
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, all_iters)

    return all_iters


def train_subnet(model, device, args, *, bn_process=True, all_iters=None, arch_loader=None):
    logging.info("start architecture training...")
    assert arch_loader is not None

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_loader = args.train_loader

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()

    if bn_process:
        adjust_bn_momentum(model, all_iters)

    all_iters += 1
    d_st = time.time()

    for data, target in train_loader:
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        data_time = time.time() - d_st
        optimizer.zero_grad()

        random_arc_list = arch_loader.get_random_batch(2500)

        for ii, arc in enumerate(random_arc_list):
            # 全部架构
            output = model(data, arc)
            loss = loss_function(output, target)

            loss.backward()

            for p in model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None
            if ii % 100 == 0:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                logging.info(
                    "training architecture - arch: {} batch acc1:{:.6f}".format(ii, acc1.item() / 100))

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

    optimizer.step()
    scheduler.step()

    if all_iters % args.save_interval == 0:
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, all_iters)

    return all_iters


def validate(model, device, args, *, all_iters=None, arch_loader=None):
    assert arch_loader is not None

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_loader = args.val_loader

    model.eval()
    max_val_iters = 250
    t1 = time.time()

    result_dict = {}

    arch_dict = arch_loader.get_part_dict()

    with torch.no_grad():
        for ii, (key, value) in enumerate(arch_dict.items()):
            for data, target in val_loader:
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)

                output = model(data, value["arch"])
                loss = loss_function(output, target)

                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                n = data.size(0)
                objs.update(loss.item(), n)

                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

            if ii % 100:
                logging.info(
                    "validate acc:{:.6f} iter:{}".format(top1.avg/100, ii))

            result_dict[key] = top1.avg / 100

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)

    logging.info("RESULTS")
    for ii, (key, value) in enumerate(result_dict.items()):
        logging.info("{: ^10}  \t  {:.6f}".format(key, value))
        if ii > 10:
            break
    logging.info("E N D")


if __name__ == "__main__":
    main()
