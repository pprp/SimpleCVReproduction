import argparse
import os
import sys
import shutil
import time
import random
import glob
import logging
import copy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from resnet20_supernet import resnet20
from utils import *

'''
Namespace(affine=True, 
alpha_type='sample_uniform', 
arch_num=50000, 
arch_start=1, 
batch_size=10000, 
bn_calibrate=False, 
bn_calibrate_batch=10000, 
bn_calibrate_batch_num=1, 
convbn_type='sample_channel', 
eval_json_path='files/Track1_final_archs.json', 
localsep_layers=None, 
localsep_portion=1.0, 
mask_repeat=1, 
model_path='files/supernet.th', 
prob_ratio=1.0, 
r=1.0, 
sameshortcut=True, 
save_dir='eval', 
save_every=1, 
save_file='eval-final', 
seed=2, 
track_running_stats=False, 
train=False, 
train_batch_size=128, 
train_epochs=1, 
train_lr=0.001, 
train_min_lr=0, 
train_momentum=0.9, 
train_print_freq=100, 
train_weight_decay=0.0005, 
workers=4)
'''
parser = argparse.ArgumentParser(
    description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--eval_json_path', help='json file containing archs to evaluete',
                    default='Track1_200_archs.json', type=str)
parser.add_argument('--model_path', default='/home/yanglongxing/project/CVPR2021-NAS-Track1/result/2021.5.11/15/model.th',
                    help='model checkpoint', type=str)
parser.add_argument('--arch_start', default=1, type=int,
                    metavar='N', help='the start index of eval archs')
parser.add_argument('--arch_num', default=50000, type=int,
                    metavar='N', help='the num of eval archs')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--batch_size', default=1024, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--affine', action='store_true', help='BN affine')
parser.add_argument('--save_dir', help='The directory used to save the trained models',
                    default='./checkpoints', type=str)
parser.add_argument('--save_file', help='The file used to save the result',
                    default='eval-1_50000', type=str)
parser.add_argument(
    '--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=1)
parser.add_argument('--convbn_type',
                    default='sample_channel',
                    type=str,
                    help='convbn forward with different mask: mix_channel or random_mix_channel or sample_channel or sample_random_channel or sample_sepmask_channel or sample_sepproject_channel or sample_localfree_channel')
parser.add_argument('--alpha_type', default='sample_uniform', type=str,
                    help='how to cal alpha in forward process: mix, sample_uniform, sample_fair, sample_flops_uniform, sample_flops_fair, sample_sandwich')
parser.add_argument('--mask_repeat', type=int, default=1,
                    help='used in random_mix_channel')
parser.add_argument('--prob_ratio', type=float, default=1.,
                    help='used in sample_flops_uniform or sample_flops_fair')
parser.add_argument('--r', type=int, default=1.,
                    help='used in local sample_localfree_channel')
parser.add_argument('--localsep_layers', default=None,
                    type=str, help='used in sample_localsepmask_channel')
parser.add_argument('--localsep_portion', type=float,
                    default=1., help='used in sample_localsepmask_channel')
parser.add_argument('--sameshortcut', action='store_true',
                    help='same shortcut')
parser.add_argument('--track_running_stats',
                    action='store_true', help='bn track_running_stats')
parser.add_argument('--bn_calibrate', action='store_true', help='bn calibrate')
parser.add_argument('--bn_calibrate_batch', type=int,
                    default=10000, help='bn calibrate batch')
parser.add_argument('--bn_calibrate_batch_num', type=int,
                    default=1, help='bn calibrate batch num')
parser.add_argument('--train', action='store_true', help='train on supernet')
parser.add_argument('--train_batch_size', type=int,
                    default=128, help='train epoch on supernet')
parser.add_argument('--train_epochs', type=int, default=1,
                    help='train epoch on supernet')
parser.add_argument('--train_lr', type=float, default=1e-3,
                    help='train lr on supernet')
parser.add_argument('--train_momentum', type=float,
                    default=0.9, help='train momentum on supernet')
parser.add_argument('--train_min_lr', type=float, default=0,
                    help='train min_lr on supernet')
parser.add_argument('--train_weight_decay', type=float,
                    default=5e-4, help='train wd on supernet')
parser.add_argument('--train_print_freq', type=int, default=100,
                    help='train print freq epoch on supernet')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()
best_prec1 = 0

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if args.bn_calibrate:
    args.track_running_stats = True

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(
    args.save_dir, '{}.txt'.format(args.save_file)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)


def main():
    global args, best_prec1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True

    model = resnet20(args.affine, args.convbn_type, args.mask_repeat, args.alpha_type, localsep_layers=args.localsep_layers,
                     localsep_portion=args.localsep_portion, same_shortcut=args.sameshortcut, track_running_stats=args.track_running_stats)
    model.cuda()
    try:
        model.load_state_dict(torch.load(args.model_path)['state_dict'])
    except:
        print("BN track running stats is False in pt but True in model, so here ignore it")
        model.load_state_dict(torch.load(args.model_path)[
                              'state_dict'], strict=False)

    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409], std=[0.1942, 0.1918, 0.1958])

    if args.bn_calibrate:
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d) and m.track_running_stats is False:
                del m.running_mean
                del m.running_var
                del m.num_batches_tracked
                m.register_buffer('running_mean', torch.zeros(m.num_features))
                m.register_buffer('running_var', torch.ones(m.num_features))
                m.register_buffer('num_batches_tracked',
                                  torch.tensor(0, dtype=torch.long))
        model.cuda()

        calib_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root='./data',
                train=False,
                transform=transforms.Compose([
                    # transforms.RandomCrop(32, 4),
                    # transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)]),
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    normalize,
                ]),
                download=True,
            ),
            batch_size=args.bn_calibrate_batch,
            pin_memory=True,
            shuffle=True,
            num_workers=args.workers,
        )

        calib_loader = get_loader(calib_loader)

    if args.train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(
                root='./data',
                train=True,
                transform=transforms.Compose([
                    transforms.RandomCrop(32, 4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    Cutout(16),
                    normalize,
                ]),
                download=True,
            ),
            batch_size=args.train_batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=args.workers,
        )

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(
            root='./data',
            train=False,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = get_loader(val_loader)

    with open(args.eval_json_path, 'r') as f:
        archs_info = json.load(f)
    sub_archs_info = {}

    if args.train:
        model_origin = model

    for arch_i in range(args.arch_start, min(50001, args.arch_start + args.arch_num)):
        if 'arch{}'.format(arch_i) in archs_info:
            lenlist = get_arch_lenlist(archs_info, arch_i)

            if args.train:
                model = train(train_loader, model_origin, lenlist, args)

            if args.bn_calibrate:
                model = calibrate_bn(calib_loader, model,
                                     lenlist, args.bn_calibrate_batch_num)

            prec1 = validate(val_loader, model, lenlist)

            sub_archs_info['arch{}'.format(arch_i)] = {}
            sub_archs_info['arch{}'.format(arch_i)]['acc'] = prec1
            sub_archs_info['arch{}'.format(
                arch_i)]['arch'] = archs_info['arch{}'.format(arch_i)]['arch']

            logging.info('Arch{}: [acc: {:.5f}][arch: {}]'.format(
                arch_i, prec1, archs_info['arch{}'.format(arch_i)]['arch']))

    save_json = os.path.join(args.save_dir, '{}.json'.format(args.save_file))
    with open(save_json, 'w') as f:
        json.dump(sub_archs_info, f)


def get_arch_lenlist(archs_dict, arch_i):
    arch = archs_dict['arch{}'.format(arch_i)]
    arch_list = arch['arch'].split('-')
    for i, lenth in enumerate(arch_list):
        arch_list[i] = int(lenth)
    return arch_list


def get_loader(loader):
    new_loader = []
    for x, y in loader:
        new_loader.append((x.cuda(), y.cuda()))
    return new_loader


def calibrate_bn(loader, model, lenlist, num):
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.data.fill_(0)
            m.running_var.data.fill_(0)
            m.num_batches_tracked.data.zero_()
            m.momentum = None
    for i, (input, _) in enumerate(loader):
        # input = input.cuda()
        if i < min(len(loader), num):
            model(input, lenlist)
    return model


def train(train_queue, model, lenlist, args):
    model = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(
    ), args.train_lr, momentum=args.train_momentum, weight_decay=args.train_weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.train_epochs, eta_min=args.train_min_lr)

    logging.info('Train arch: {}'.format(lenlist))
    for epoch in range(args.train_epochs):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()

        model.train()

        end = time.time()
        for i, (input, target) in enumerate(train_queue):
            data_time.update(time.time() - end)

            target_var = target.cuda()
            input_var = input.cuda()

            optimizer.zero_grad()  # zero gradient
            output = model(input_var, lenlist)  # compute output
            loss = criterion(output, target_var)  # compute loss
            loss.backward()  # compute gradient
            optimizer.step()  # do SGD step

            output = output.float()
            loss = loss.float()
            # measure accuracy and record loss
            prec1 = accuracy(output.data, target_var.data)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.train_print_freq == 0 or i == len(train_queue) - 1:
                logging.info('\tEpoch: [{0}][{1}/{2}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_queue), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))

        lr_scheduler.step()

    del criterion
    del optimizer
    del lr_scheduler

    return model


def validate(valid_queue, model, lenlist):
    """
    Run evaluation
    """
    # switch to evaluate mode
    top1 = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(valid_queue):
            input_var = input
            target_var = target

            # compute output
            output = model(input_var, lenlist)

            output = output.float()

            # measure accuracy
            prec1 = accuracy(output.data, target)[0]
            top1.update(prec1.item(), input.size(0))

    return top1.avg


if __name__ == '__main__':
    main()
