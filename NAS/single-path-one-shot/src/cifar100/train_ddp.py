import argparse
import json
import logging
import os
import random
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from cifar100_dataset import get_train_loader, get_val_loader
from slimmable_resnet20 import (SwitchableBatchNorm2d, max_arc_rep,
                                mutableResNet20)
from utils import (ArchLoader, AvgrageMeter, CrossEntropyLabelSmooth, accuracy,
                   get_lastest_model, get_num_correct, get_parameters,
                   save_checkpoint)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

writer = SummaryWriter("./runs/%s-%05d" %
                       (time.strftime("%m-%d", time.localtime()), random.randint(0, 100)))
# batch size 128 - lr: 0.1


def reduce_tensor(tensor, device=0, world_size=1):
    tensor = tensor.clone()
    torch.distributed.reduce(tensor, device)
    tensor.div_(world_size)
    return tensor


def get_args():
    parser = argparse.ArgumentParser("ResNet20-Cifar100-oneshot")
    parser.add_argument('--warmup', default=0, type=int,
                        help="warmup weight of the whole channels")
    parser.add_argument('--total_iters', default=10000, type=int)

    parser.add_argument('--num_workers', default=15, type=int)
    parser.add_argument(
        '--path', default="Track1_final_archs.json", help="path for json arch files")
    parser.add_argument('--batch_size', type=int,
                        default=16384, help='batch size')
    parser.add_argument('--learning_rate', type=float,
                        default=0.377, help='init learning rate')  # math.sqrt(4096/128)*0.1/3
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=4e-5, help='weight decay')
    parser.add_argument('--label_smooth', type=float,
                        default=0.1, help='label smoothing')

    parser.add_argument('--save', type=str, default='./weights',
                        help='path for saving trained weights')
    parser.add_argument('--save_interval', type=int,
                        default=100, help='report frequency')

    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval_resume', type=str,
                        default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--auto-continue', type=bool,
                        default=False, help='report frequency')

    # parallel
    parser.add_argument('--local_rank', type=int, default=None,
                        help='local rank for distributed training')
    parser.add_argument('--gpu', type=int, default=0,
                        help='gpu device id')  # TODO

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    num_gpus = torch.cuda.device_count()
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.batch_size = args.batch_size // args.world_size

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
        'log/train-{}-{:02}-{:02}-{:.3f}'.format(local_time.tm_year % 2000, local_time.tm_mon, local_time.tm_mday, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    train_loader = get_train_loader(
        args.batch_size, args.local_rank, args.num_workers, args.total_iters)

    val_loader = get_val_loader(args.batch_size, args.num_workers)

    model = mutableResNet20()

    logging.info('load model successfully')

    optimizer = torch.optim.SGD(get_parameters(model),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)

    if use_gpu:
        # model = nn.DataParallel(model)
        model = model.cuda(args.gpu)
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        loss_function = criterion_smooth.cuda()
    else:
        loss_function = criterion_smooth

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5)

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
            validate(model, args, all_iters=all_iters,
                     arch_loader=arch_loader)
        exit(0)

    # warmup weights
    if args.warmup > 0:
        logging.info("begin warmup weights")
        while all_iters < args.warmup:
            all_iters = train_supernet(
                model, args, bn_process=False, all_iters=all_iters)

        validate(model, args, all_iters=all_iters,
                 arch_loader=arch_loader)

    while all_iters < args.total_iters:
        logging.info("="*50)
        all_iters = train_subnet(model, args, bn_process=False,
                                 all_iters=all_iters, arch_loader=arch_loader)

        if all_iters % 200 == 0 and args.local_rank == 0:
            logging.info("validate iter {}".format(all_iters))

            validate(model, args, all_iters=all_iters,
                     arch_loader=arch_loader)


def adjust_bn_momentum(model, iters):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 1 / iters
        elif isinstance(m, SwitchableBatchNorm2d):
            m.momentum = 1 / iters


def train_subnet(model, args, *, bn_process=False, all_iters=None, arch_loader=None):
    logging.info("start subnet training...")
    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_loader = args.train_loader

    t1 = time.time()

    model.train()

    if bn_process:
        adjust_bn_momentum(model, all_iters)

    all_iters += 1
    d_st = time.time()

    total_correct = 0

    for data, target in train_loader:
        target = target.type(torch.LongTensor)
        data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        data_time = time.time() - d_st
        optimizer.zero_grad()

        # fair_arc_list = arch_loader.generate_fair_batch()
        fair_arc_list = arch_loader.generate_niu_fair_batch()
        # fair_arc_list = arch_loader.get_random_batch(25)

        for ii, arc in enumerate(fair_arc_list):
            # 全部架构
            output = model(data, arch_loader.convert_list_arc_str(arc))
            loss = loss_function(output, target)

            loss_reduce = reduce_tensor(loss, 0, args.world_size)

            loss.backward()

            for p in model.parameters():
                if p.grad is not None and p.grad.sum() == 0:
                    p.grad = None

            total_correct += get_num_correct(output, target)

            if ii % 15 == 0 and args.local_rank == 0:
                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                logging.info(
                    "epoch: {:4d} \t acc1:{:.4f} \t acc5:{:.4f} \t loss:{:.4f}".format(all_iters, acc1.item(), acc5.item(), loss.item()))

                writer.add_scalar("Train/Loss", loss.item(),
                                  all_iters * len(train_loader) * args.batch_size+ii)
                writer.add_scalar("Train/acc1", acc1.item(),
                                  all_iters * len(train_loader) * args.batch_size+ii)
                writer.add_scalar("Train/acc5", acc5.item(),
                                  all_iters * len(train_loader) * args.batch_size+ii)

    # 16 when using Fair sampling strategy
    if args.local_rank == 0:
        writer.add_scalar("Accuracy", total_correct /
                          (len(train_loader) * args.batch_size * 16), all_iters)
        writer.add_histogram("first_conv.weight",
                             model.module.first_conv.weight, all_iters)

        writer.add_histogram(
            "layer1[0].weight", model.module.layer1[0].body[0].weight, all_iters)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

    optimizer.step()
    scheduler.step()

    if all_iters % args.save_interval == 0 and args.local_rank == 0:
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, all_iters)

    return all_iters


def validate(model, args, *, all_iters=None, arch_loader=None):
    assert arch_loader is not None

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_loader = args.val_loader

    model.eval()
    t1 = time.time()

    result_dict = {}

    arch_dict = arch_loader.get_part_dict()

    with torch.no_grad():
        for ii, (key, value) in enumerate(arch_dict.items()):
            for data, target in val_loader:
                target = target.type(torch.LongTensor)
                data, target = data.cuda(args.gpu), target.cuda(args.gpu)

                output = model(data, value["arch"])
                loss = loss_function(output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                n = data.size(0)
                objs.update(loss.item(), n)

                top1.update(acc1.item(), n)
                top5.update(acc5.item(), n)

            if ii % 5:
                logging.info(
                    "validate acc:{:.6f} iter:{}".format(top1.avg/100, ii))
                writer.add_scalar("Val/Loss", loss.item(),
                                  all_iters * len(val_loader) * args.batch_size+ii)
                writer.add_scalar("Val/acc1", acc1.item(),
                                  all_iters * len(val_loader) * args.batch_size+ii)
                writer.add_scalar("Val/acc5", acc5.item(),
                                  all_iters * len(val_loader) * args.batch_size+ii)

            result_dict[key] = top1.avg

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 acc = {:.6f},\t'.format(top1.avg) + \
              'Top-5 acc = {:.6f},\t'.format(top5.avg) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)

    logging.info("RESULTS")
    for ii, (key, value) in enumerate(result_dict.items()):
        logging.info("{: ^10}  \t  {:.6f}".format(key, value))
        if ii > 10:
            break
    logging.info("E N D")


def train_supernet(model, args, *, bn_process=False, all_iters=None):
    logging.info("start warmup training...")

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_loader = args.train_loader

    t1 = time.time()

    model.train()

    if bn_process:
        adjust_bn_momentum(model, all_iters)

    all_iters += 1
    d_st = time.time()

    # print(model)

    total_correct = 0

    for ii, (data, target) in enumerate(train_loader):
        target = target.type(torch.LongTensor)
        data, target = data.cuda(args.gpu), target.cuda(args.gpu)
        data_time = time.time() - d_st

        optimizer.zero_grad()

        # 一个批次
        output = model(data, max_arc_rep)
        loss = loss_function(output, target)

        loss.backward()

        for p in model.parameters():
            if p.grad is not None and p.grad.sum() == 0:
                p.grad = None

        total_correct += get_num_correct(output, target)

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

        if ii % 6 == 0:  # 50,000 / bs(4096)=12
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            logging.info("warmup batch acc1: {:.6f} lr: {:.6f}".format(
                acc1.item(), scheduler.get_last_lr()[0]))

            writer.add_scalar("WTrain/Loss", loss.item(),
                              all_iters * len(train_loader) * args.batch_size+ii)
            writer.add_scalar("WTrain/acc1", acc1.item(),
                              all_iters * len(train_loader) * args.batch_size+ii)
            writer.add_scalar("WTrain/acc5", acc5.item(),
                              all_iters * len(train_loader) * args.batch_size+ii)

        optimizer.step()

    writer.add_scalar("Accuracy", total_correct /
                      (len(train_loader)*args.batch_size), all_iters)

    writer.add_histogram("first_conv.weight",
                         model.module.first_conv.weight, all_iters)

    writer.add_histogram(
        "layer1[0].weight", model.module.layer1[0].body[0].weight, all_iters)

    scheduler.step()

    top1, top5 = accuracy(output, target, topk=(1, 5))

    if True:
        printInfo = 'TRAIN EPOCH {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_last_lr()[0], loss.item()) + \
                    'Top-1 acc = {:.5f}%,\t'.format(top1.item()) + \
                    'Top-5 acc = {:.5f}%,\t'.format(top5.item()) + \
                    'data_time = {:.5f},\ttrain_time = {:.5f}'.format(
                        data_time, (time.time() - t1))

        logging.info(printInfo)
        t1 = time.time()

    if all_iters % args.save_interval == 0:
        save_checkpoint({
            'state_dict': model.state_dict(),
        }, all_iters)

    return all_iters


if __name__ == "__main__":
    main()
