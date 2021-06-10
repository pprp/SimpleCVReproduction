# Universally Slimmable Networks and Improved Training Techniques
import argparse
import datetime
import functools
import logging
import os
import random
import sys
import time
import glob

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch import nn
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from datasets.cifar100_dataset import get_train_loader, get_val_loader
import model as mm
from utils.utils import (ArchLoader, AvgrageMeter, CrossEntropyLossSoft,
                         accuracy, create_exp_dir, reduce_mean,
                         save_checkpoint)

print = functools.partial(print, flush=True)

parser = argparse.ArgumentParser("ResNet20-cifar100")

parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=2048,
                    help='batch size')  # 8192
parser.add_argument('--learning_rate', type=float,
                    default=0.1, help='init learning rate')  # 0.8
parser.add_argument('--num_workers', type=int,
                    default=3, help='num of workers')
parser.add_argument('--model-type', type=str, default="sample",
                    help="type of model(sample masked dynamic independent slimmable original)")

parser.add_argument('--finetune', action='store_true',
                    help='finetune model with distill')
parser.add_argument('--distill', action="store_true",
                    help="finetune model with track_200.json")

# hyper parameter
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=5e-4, help='weight decay')

parser.add_argument('--report_freq', type=float,
                    default=2, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=200,
                    help='num of training epochs')

parser.add_argument('--classes', type=int, default=100,
                    help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5, help='gradient clipping')
parser.add_argument('--label_smooth', type=float,
                    default=0.1, help='label smoothing')
parser.add_argument('--config', help="configuration file",
                    type=str, default="configs/meta.yml")
parser.add_argument('--save_dir', type=str,
                    help="save exp floder name", default="exp1_sandwich")
args = parser.parse_args()

# process argparse & yaml
if not args.config:
    opt = vars(args)
    args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(args)
    args = opt
else:  # yaml priority is higher than args
    opt = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(vars(args))
    args = argparse.Namespace(**opt)

args.exp_name = args.save_dir + "_" + datetime.datetime.now().strftime("%mM_%dD_%HH") + "_" + \
    "{:04d}".format(random.randint(0, 1000))

# 文件处理
if not os.path.exists(os.path.join("exp", args.exp_name)):
    os.makedirs(os.path.join("exp", args.exp_name))


# 日志文件
log_format = "%(asctime)s %(message)s"
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt="%m/%d %I:%M:%S %p")

fh = logging.FileHandler(os.path.join("exp", args.exp_name, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)

# 配置文件
with open(os.path.join("exp", args.exp_name, "config.yml"), "w") as f:
    yaml.dump(args, f)

# Tensorboard文件
writer = SummaryWriter("exp/%s/runs/%s-%05d" %
                       (args.exp_name, time.strftime("%m-%d", time.localtime()), random.randint(0, 100)))

create_exp_dir(os.path.join("exp", args.exp_name),
               scripts_to_save=glob.glob('*.py'))


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

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

    logging.info('gpu device = %d' % args.gpu)

    model = mm.build_model(args.model_type)

    if args.finetune:
        checkpoint = torch.load("./weights/2021Y_05M_31D_07H_0329/checkpoint-latest.pth.tar",
                                map_location=None)
        model.load_state_dict(checkpoint['state_dict'])

    model = model.cuda(args.gpu)

    if num_gpus > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size

    criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
    soft_criterion = CrossEntropyLossSoft()

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 200, eta_min=0.0005)

    # Prepare data
    train_loader = get_train_loader(
        args.batch_size, args.local_rank, args.num_workers)
    # 原来跟train batch size一样，现在修改小一点 ，
    val_loader = get_val_loader(args.batch_size, args.num_workers)

    archloader = ArchLoader("data/track_200.json")

    for epoch in range(args.epochs):
        train_lu_shun(train_loader, val_loader,  optimizer, scheduler, model,
                      archloader, criterion, soft_criterion, args, args.seed, epoch, writer)

        writer.add_scalar("lr", scheduler.get_last_lr()[0], epoch)

        scheduler.step()
        if (epoch + 1) % args.report_freq == 0:
            top1_val, objs_val = infer(train_loader, val_loader, model, criterion,
                                       archloader, args, epoch)

            if args.local_rank == 0:
                # model
                if writer is not None:
                    writer.add_scalar("Val/loss", objs_val, epoch)
                    writer.add_scalar("Val/acc1", top1_val, epoch)

                save_checkpoint(
                    {'state_dict': model.state_dict(), }, epoch, args.exp_name)


def train_lu_shun(train_dataloader, val_dataloader, optimizer, scheduler, model, archloader, criterion, soft_criterion, args, seed, epoch, writer=None):
    losses_, top1_ = AvgrageMeter(), AvgrageMeter()

    trick = False
    distill_lamda = 2.0
    sample_accumulation_steps = 6
    valid_running = False
    # arch_optimizer = torch.optim.Adam(model.alpha, lr=0.0003, betas=(
    #     0.5, 0.999), weight_decay=0.001)

    model.train()
    widest = [16, 16, 16, 16, 16, 16, 16, 32, 32,
              32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]
    narrowest = [4,  4,  4, 4,  4,  4,  4,  4, 4,
                 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4]

    if args.finetune:
        candidates = archloader.get_arch_list()
    else:
        candidates = [archloader.generate_spos_like_batch().tolist()
                      for i in range(6)]

    train_loader = tqdm(train_dataloader)
    train_loader.set_description(
        '[%s%04d/%04d %s%f]' % ('Epoch:', epoch + 1, args.epochs, 'lr:', scheduler.get_last_lr()[0]))

    optimizer.zero_grad()

    for step, (image, target) in enumerate(train_loader):
        n = image.size(0)

        image = Variable(image, requires_grad=False).cuda(
            args.gpu, non_blocking=True)
        target = Variable(target, requires_grad=False).cuda(
            args.gpu, non_blocking=True)

        # small trick to avoid lr shift
        if trick:
            scheduler.step(max(0, epoch-1))

        optimizer.zero_grad()

        for idx in range(sample_accumulation_steps):
            output = model(image)
            loss = criterion(output, target)

            if args.distill:
                teacher_output = model(image, widest)
                teacher_loss = criterion(teacher_output, target)

                soft_target = torch.nn.functional.softmax(
                    teacher_output, dim=1).detach()
                distill_loss = soft_criterion(output, soft_target)

                loss = 0.5 * (loss + teacher_loss) + \
                    distill_loss * distill_lamda

            loss.backward()

        optimizer.step()

        # statistics show ....
        prec1, prec5 = accuracy(output, target, topk=(1, 5))

        if torch.cuda.device_count() > 1:
            torch.distributed.barrier()

            loss = reduce_mean(loss, args.nprocs)
            prec1 = reduce_mean(prec1, args.nprocs)

        losses_.update(loss.data.item(), n)
        top1_.update(prec1.data.item(), n)

        postfix = {'train_loss': '%.6f' % (
            losses_.avg), 'train_acc1': '%.6f' % top1_.avg}

        train_loader.set_postfix(log=postfix)

        if args.local_rank == 0 and step % 10 == 0 and writer is not None:
            writer.add_scalar("Train/loss", losses_.avg, step +
                              len(train_dataloader) * epoch * args.batch_size)
            writer.add_scalar("Train/acc1", top1_.avg, step +
                              len(train_dataloader) * epoch * args.batch_size)


def train(train_dataloader, val_dataloader, optimizer, scheduler, model, archloader, criterion, soft_criterion, args, seed, epoch, writer=None):
    losses_, top1_ = AvgrageMeter(), AvgrageMeter()
    inplace_distillation = True

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
            candidate_list += [archloader.generate_spos_like_batch().tolist()
                               for i in range(6)]
            candidate_list += [narrowest]

            # archloader.generate_niu_fair_batch(step)
            # 全模型来一遍
            soft_target = model(image, widest)
            soft_loss = criterion(soft_target, target)
            soft_loss.backward()

            # 采样几个子网来一遍
            for arc in candidate_list:
                logits = model(image, arc)
                # loss = soft_criterion(logits, soft_target.cuda(
                #     args.gpu, non_blocking=True))
                if inplace_distillation:
                    T = 1  # temperature in knowledge distillation 2 10 20

                    soft_target = torch.nn.functional.softmax(
                        soft_target/T, dim=1).detach()

                    loss = 0.5 * \
                        torch.mean(soft_criterion(logits, soft_target)
                                   ) + 0.5 * criterion(logits, target)
                else:
                    loss = criterion(logits, target)

                loss.backward()

            prec1, _ = accuracy(logits, target, topk=(1, 5))
            losses_.update(loss.data.item(), n)
            top1_.update(prec1.data.item(), n)

        elif args.model_type == "original":
            logits = model(image)
            loss = criterion(logits, target)
            loss.backward()

            prec1, _ = accuracy(logits, target, topk=(1, 5))
            losses_.update(loss.data.item(), n)
            top1_.update(prec1.data.item(), n)

        if torch.cuda.device_count() > 1:
            torch.distributed.barrier()
            loss = reduce_mean(loss, args.nprocs)
            prec1 = reduce_mean(prec1, args.nprocs)

        optimizer.step()
        optimizer.zero_grad()

        postfix = {'train_loss': '%.6f' % (
            losses_.avg), 'train_acc1': '%.6f' % top1_.avg}

        train_loader.set_postfix(log=postfix)

        if args.local_rank == 0 and step % 10 == 0 and writer is not None:
            writer.add_scalar("Train/loss", losses_.avg, step +
                              len(train_dataloader) * epoch * args.batch_size)
            writer.add_scalar("Train/acc1", top1_.avg, step +
                              len(train_dataloader) * epoch * args.batch_size)


def infer(train_loader, val_loader, model, criterion,  archloader, args, epoch):
    objs_, top1_ = AvgrageMeter(), AvgrageMeter()

    model.eval()
    now = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    # [16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64]
    # .generate_width_to_narrow(epoch, args.epochs)
    fair_arc_list = archloader.generate_spos_like_batch().tolist()

    logging.info('{} |=> Test rng = {}'.format(now, fair_arc_list))  # 只测试最后一个模型

    # if args.model_type == "dynamic":
    #     # BN calibration
    #     retrain_bn(model, train_loader, fair_arc_list, device=0)

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

            top1, _ = accuracy(logits, target, topk=(1, 5))

            if torch.cuda.device_count() > 1:
                torch.distributed.barrier()
                loss = reduce_mean(loss, args.nprocs)
                top1 = reduce_mean(top1, image.size(0))
                top5 = reduce_mean(top5, image.size(0))

            n = image.size(0)
            objs_.update(loss.data.item(), n)
            top1_.update(top1.data.item(), n)

        now = time.strftime('%Y-%m-%d %H:%M:%S',
                            time.localtime(time.time()))
        logging.info('{} |=> valid: step={}, loss={:.2f}, val_acc1={:.2f}, datatime={:.2f}'.format(
            now, step, objs_.avg, top1_.avg, datatime))

    return top1_.avg, objs_.avg


if __name__ == '__main__':
    main()
