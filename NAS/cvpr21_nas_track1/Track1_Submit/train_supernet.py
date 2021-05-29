import argparse
import os
import sys
import shutil
import time
import random
import glob
import logging

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import pytorch_warmup as warmup
from resnet20_supernet import resnet20
from utils import *
from auto_argument import *

'''
Namespace(aa=False, affine=True, alpha_type='sample_trackarch',
 arch_lr=0.0003, arch_weight_decay=0.001, 
 batch_size=128, convbn_type='sample_channel', 
 cutout=True, cutout_lenth=16, distill=True, 
 distill_lamda=2.0, drop_path_rate=0.0, 
 dropout=0.0, epochs=20, evaluate=False,
 label_smooth=0.0, linear_dp_rate=False, 
 localsep_layers=None, localsep_portion=1, 
 lr=0.0035, mask_repeat=1, min_distill=False, 
 min_distill_lamda=1.0, min_lr=0.0005, 
 momentum=0.9, print_freq=50, prob_ratio=1.0, 
 r=1.0, resume='train/model.th', sameshortcut=True, 
 sample_accumulation_steps=6, sandwich_N=2, 
 save_alpha=False, save_dir='train', save_every=1, 
 seed=0, start_epoch=0, tauloss=False, tauloss_lamda=1.0, 
 tauloss_noise=0.01, track_file='files/Track1_100_archs.json', 
 track_running_stats=False, train_portion=0.5, 
 warmup=False, warmup_step=2000, weight_decay=0.0, workers=4)
'''

parser = argparse.ArgumentParser(
    description='Propert ResNets for CIFAR10 in pytorch')
parser.add_argument('--workers', default=1, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int,
                    metavar='N', help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--batch_size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--train_portion', default=0.5,
                    type=float, help='train portion')  # 训练一部分数据集
parser.add_argument('--lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--min_lr', default=0.0, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')

parser.add_argument('--arch_lr', default=3e-4, type=float,
                    metavar='ARCH-LR', help='initial arch learning rate')
parser.add_argument('--arch_weight_decay', default=1e-3, type=float,
                    metavar='ARCH-WD', help='arch weight decay (default: 5e-4)')
parser.add_argument('--drop_path_rate', default=0., type=float,
                    metavar='ARCH-LR', help='initial arch learning rate') # 实际上并没有使用droppath

parser.add_argument('--linear_dp_rate', action='store_true', # 未调用
                    help='linearly increase dp rate')
parser.add_argument('--dropout', default=0., type=float,
                    metavar='ARCH-LR', help='initial arch learning rate') # dropout也没有使用
parser.add_argument('--affine', action='store_true', help='BN affine') # True

parser.add_argument('--print_freq', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--save_dir', help='The directory used to save the trained models',
                    default='./checkpoints', type=str)
parser.add_argument(
    '--save_every', help='Saves checkpoints at every specified number of epochs', type=int, default=1)
parser.add_argument( 
    '--convbn_type',
    default='sample_channel',
    type=str,
    help='convbn forward with different mask: mix_channel or random_mix_channel or sample_channel or sample_random_channel or sample_sepmask_channel or sample_sepproject_channel or sample_localfree_channel or sample_localsepmask_channel or sample_localsepadd_channel'
)
'''
指定conv的类型，主要涉及算法有:
mix_channel
random_mix_channel
sample_channel
sample_random_channel
sample_sepmask_channel
sample_sepproject_channel
sample_localfree_channel
sample_localsepmask_channel
sample_localsepadd_channel
'''
parser.add_argument('--alpha_type', default='sample_trackarch', type=str,
                    help='how to cal alpha in forward process: mix, sample_uniform, sample_fair, sample_flops_uniform, sample_flops_fair, sample_sandwich, sample_trackarch')
'''
alpha type: 主要是指定算法训练策略，设计方法有：
mix
sample_uniform: spos
sample_fair: fairnas
sample_flops_uniform: flops
sample_flops_uniform: flops
sample_sandwich: autoslim
sample_trackarch: 根据arch.json进行sample
'''
parser.add_argument('--mask_repeat', type=int, default=1,
                    help='used in random_mix_channel')
parser.add_argument('--prob_ratio', type=float, default=1.,
                    help='used in sample_flops_uniform or sample_flops_fair')

parser.add_argument('--r', type=int, default=1.,
                    help='used in local sample_localfree_channel')
parser.add_argument('--sandwich_N', type=int, default=2,
                    help='used in sample_sandwich')

parser.add_argument('--localsep_layers', default=None,
                    type=str, help='used in sample_localsepmask_channel')
parser.add_argument('--localsep_portion', type=float,
                    default=1, help='used in sample_localsepmask_channel')

parser.add_argument('--track_file', default='Track1_submit/files/Track1_200_archs.json',
                    type=str, help='used in sample_trackarch')
parser.add_argument('--sample_accumulation_steps', type=int,
                    default=3, help='used in sample_based method')
parser.add_argument('--label_smooth', type=float,
                    default=0.0, help='label smoothing')

parser.add_argument('--aa', action='store_true', help='aa')
parser.add_argument('--cutout', action='store_true', help='cutout')
parser.add_argument('--cutout_lenth', type=int,
                    default=16, help='cutout length')
parser.add_argument('--distill', action='store_true', help='distill')
parser.add_argument('--distill_lamda', type=float,
                    default=1.0, help='distill lamda')
parser.add_argument('--min_distill', action='store_true', help='min distill')
parser.add_argument('--min_distill_lamda', type=float,
                    default=1.0, help='min distill lamda')

parser.add_argument('--tauloss', action='store_true', help='tauloss')
parser.add_argument('--tauloss_lamda', type=float,
                    default=1.0, help='tauloss lamda')
parser.add_argument('--tauloss_noise', type=float,
                    default=1e-2, help='tauloss noise')

parser.add_argument('--warmup', action='store_true', help='warmup')
parser.add_argument('--warmup_step', type=int,
                    default=2000, help='warmup step')
parser.add_argument('--sameshortcut', action='store_true',
                    help='same shortcut')
parser.add_argument('--track_running_stats',
                    action='store_true', help='bn track_running_stats')
parser.add_argument('--save_alpha', action='store_true',
                    help='save alpha of all epoch')
parser.add_argument('--seed', type=int, default=2, help='random seed')
args = parser.parse_args()
best_prec1 = 0

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
if args.save_alpha:
    alpha_path = os.path.join(args.save_dir, 'alpha')
    os.makedirs(alpha_path)

create_exp_dir(args.save_dir, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
logging.info(args)


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """

    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss.mean()


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + \
            self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def main():
    global args, best_prec1

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    cudnn.benchmark = False
    cudnn.enabled = True
    cudnn.deterministic = True

    model = resnet20(
        args.affine,
        args.convbn_type,
        args.mask_repeat,
        args.alpha_type,
        args.prob_ratio,
        args.r,
        args.localsep_layers,
        args.localsep_portion,
        args.track_file,
        args.drop_path_rate,
        args.dropout,
        args.sameshortcut,
        args.track_running_stats,
    )
    model.cuda()
    logging.info(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            # logging.info("=> loaded checkpoint '{}' (epoch {})".format(args.evaluate, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    train_queue, valid_queue = get_data_loader(args)

    # define loss function (criterion) and optimizer
    criterion_smooth = CrossEntropyLabelSmooth(100, args.label_smooth).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    soft_criterion = CrossEntropyLossSoft().cuda()

    optimizer = torch.optim.SGD(model.parameters(
    ), args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.min_lr)

    if args.warmup:
        warmup_scheduler = warmup.LinearWarmup(optimizer, args.warmup_step)
    else:
        warmup_scheduler = None

    if 'mix' == args.alpha_type:
        arch_optimizer = torch.optim.Adam(model.alpha, lr=args.arch_lr, betas=(
            0.5, 0.999), weight_decay=args.arch_weight_decay)
    else:
        arch_optimizer = None

    if args.evaluate:
        validate(valid_queue, model, criterion)
        return

    if args.save_alpha:
        with torch.no_grad():
            alpha1, alpha2, alpha3 = model.alpha_cal()
            save_checkpoint({
                'alpha1': alpha1,
                'alpha2': alpha2,
                'alpha3': alpha3,
            }, False, filename=os.path.join(alpha_path, 'epoch_-1.th'))

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        logging.info('current lr {:.5e}'.format(
            optimizer.param_groups[0]['lr']))

        if args.linear_dp_rate and args.drop_path_rate > 0.:
            model.set_drop_path_rate(
                args.drop_path_rate * (epoch - args.start_epoch) / (args.epochs - args.start_epoch))

        train(train_queue, valid_queue if 'mix' == args.alpha_type else None, model, criterion if args.label_smooth ==
              0 else criterion_smooth, soft_criterion, optimizer, arch_optimizer, lr_scheduler, warmup_scheduler, epoch, args)

        if 'mix' == args.alpha_type:
            with torch.no_grad():
                logging.info(model.alpha_cal())
        elif 'fair' in args.alpha_type:
            logging.info((model.counts1, model.counts2, model.counts3))

        # lr_scheduler.step()

        if args.save_alpha:
            with torch.no_grad():
                alpha1, alpha2, alpha3 = model.alpha_cal()
                save_checkpoint({
                    'alpha1': alpha1,
                    'alpha2': alpha2,
                    'alpha3': alpha3,
                }, False, filename=os.path.join(alpha_path, 'epoch_{}.th'.format(epoch)))

        # evaluate on validation set
        prec1 = validate(valid_queue, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        if epoch > 0 and epoch % args.save_every == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
            }, is_best, filename=os.path.join(args.save_dir, 'checkpoint.th'))

        save_checkpoint({
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=os.path.join(args.save_dir, 'model.th'))

    if 'mix' == args.alpha_type:
        alpha1, alpha2, alpha3 = model.alpha_cal()
        generate_result(os.path.join(
            args.save_dir, 'result.json'), alpha1, alpha2, alpha3)


def train(train_queue, valid_queue, model, criterion, soft_criterion, optimizer, arch_optimizer, scheduler, warmup_scheduler, epoch, args):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_queue):
        if valid_queue is not None:
            try:
                input_search, target_search = next(valid_queue_iter)
            except:
                valid_queue_iter = iter(valid_queue)
                input_search, target_search = next(valid_queue_iter)
            input_search_var = input_search.cuda()
            target_search_var = target_search.cuda()

        # measure data loading time
        data_time.update(time.time() - end)

        target_var = target.cuda()
        input_var = input.cuda()

        scheduler.step(max(0, epoch - 1))  # small trick to aviod LR shift
        if warmup_scheduler is not None:
            warmup_scheduler.dampen()

        if 'sample_localfree_channel' == args.convbn_type:
            model.min_min(input_var, target_var, criterion)

        if 'mix' == args.alpha_type:
            output = model(input_var)  # compute output
            loss = criterion(output, target_var)  # compute loss
            optimizer.zero_grad()  # zero gradient
            loss.backward()  # compute gradient
            optimizer.step()  # do SGD step
        elif 'sample_sandwich' == args.alpha_type:
            # sandwich_inplace_distillation
            drop_path_rate = model.drop_path_rate
            optimizer.zero_grad()

            model.alpha_sandwich_type = 'max'
            output = model(input_var)
            loss = criterion(output, target_var)
            loss.backward()
            
            soft_target_var = torch.nn.functional.softmax(
                output, dim=1).detach()
            model.set_drop_path_rate(0.)

            model.alpha_sandwich_type = 'min'
            output = model(input_var)
            loss = soft_criterion(output, soft_target_var)
            loss.backward()
             
            model.alpha_sandwich_type = 'random'
            for _ in range(args.sandwich_N):
                output = model(input_var)
                loss = soft_criterion(output, soft_target_var)
                loss.backward()
            optimizer.step()
            model.set_drop_path_rate(drop_path_rate)
        elif args.tauloss:
            optimizer.zero_grad()  # zero gradient
            for _ in range(args.sample_accumulation_steps):
                model.alpha_hold()
                output = model(input_var)
                loss1 = criterion(output, target_var)
                model.alpha_hold()
                output = model(input_var)
                loss2 = criterion(output, target_var)
                
                new_input_var = input_var + args.tauloss_noise * \
                    2 * (torch.rand(input_var.size()) - 1).cuda()
                model.alpha_pop()
                
                output = model(new_input_var)
                loss3 = criterion(output, target_var)
                model.alpha_pop()
                
                output = model(new_input_var)
                loss4 = criterion(output, target_var)
                loss = 0.25 * (loss1 + loss2 + loss3 + loss4) + 0.5 * args.tauloss_lamda * max(
                    0, -torch.sign(loss1 - loss2) * (loss3 - loss4) - torch.sign(loss3 - loss4) * (loss1 - loss2))
                loss.backward()  # compute gradient
            optimizer.step()  # do SGD step
        else: # 进入这里
            optimizer.zero_grad()  # zero gradient
            for _ in range(args.sample_accumulation_steps): # TODO 为什么要多来几个step呢
                output = model(input_var)  # compute output
                loss = criterion(output, target_var)  # compute loss

                if args.distill:
                    teacher_output = model(input_var, [
                                           16, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64, 64])
                    teacher_loss = criterion(teacher_output, target_var)
                    soft_target_var = torch.nn.functional.softmax(
                        teacher_output, dim=1).detach()
                    distill_loss = soft_criterion(output, soft_target_var)

                    loss = 0.5 * (loss + teacher_loss) + \
                        args.distill_lamda * distill_loss

                    if args.min_distill:
                        min_output = model(
                            input_var, [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
                        min_loss = criterion(min_output, target_var)
                        min_distill_loss = soft_criterion(
                            min_output, soft_target_var)
                        loss = loss + 0.5 * min_loss + args.min_distill_lamda * min_distill_loss
                loss.backward()  # compute gradient
            optimizer.step()  # do SGD step

        if valid_queue is not None: # TODO 在验证集上跑？
            # arch compute output
            output_search = model(input_search_var)
            loss_search = criterion(output_search, target_search_var)

            # arch compute gradient and do ADAM step
            arch_optimizer.zero_grad()
            loss_search.backward()
            arch_optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target_var.data)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(epoch, i, len(train_queue), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))


def validate(valid_queue, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(valid_queue):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()

            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                logging.info('Test: [{0}/{1}]\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                             'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(i, len(valid_queue), batch_time=batch_time, loss=losses, top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))

    return top1.avg


def get_data_loader(args):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409], std=[0.1942, 0.1918, 0.1958])
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        # transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
    ])
    if args.aa:
        train_transform.transforms.append(CIFAR10Policy())
    train_transform.transforms.append(transforms.ToTensor())

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_lenth))
    train_transform.transforms.append(normalize)

    train_data = datasets.CIFAR100(
        root='./data', train=True, transform=train_transform, download=True)
    if 'mix' == args.alpha_type:
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))

        train_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[:split]), pin_memory=True, num_workers=args.workers)
        valid_queue = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, sampler=torch.utils.data.sampler.SubsetRandomSampler(
            indices[split:num_train]), pin_memory=True, num_workers=args.workers)
    else:
        valid_data = datasets.CIFAR100(root='./data', train=False, transform=transforms.Compose([
                                       transforms.ToTensor(), normalize]), download=True)
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.workers)
        valid_queue = torch.utils.data.DataLoader(
            valid_data, batch_size=args.batch_size, pin_memory=True, shuffle=True, num_workers=args.workers)

    return train_queue, valid_queue


if __name__ == '__main__':
    main()
