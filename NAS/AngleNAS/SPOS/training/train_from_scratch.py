from thop import profile
from utils import *
import os
import sys
import numpy as np
import time
import torch
import glob
import random
import logging
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from model import Network
import pickle
from config import config
import apex

sys.path.append("../..")

IMAGENET_TRAINING_SET_SIZE = 1281167
IMAGENET_TEST_SET_SIZE = 50000

parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--local_rank', type=int, default=0,
                    help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float,
                    default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float,
                    default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float,
                    default=100, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=240,
                    help='num of training epochs')
parser.add_argument('--total_iters', type=int,
                    default=300000, help='total iters')
parser.add_argument('--save', type=str, default='SPOS', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float,
                    default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float,
                    default=0.1, help='label smoothing')
parser.add_argument('--model_id', type=str,
                    default='2 2 -1 0 0 -1 -1 0 5 -1 4 4 0 0 4 2 3 4 4 4 3', help='model_id')
parser.add_argument('--train_dir', type=str,
                    default='../../data/train', help='path to training dataset')
parser.add_argument('--test_dir', type=str,
                    default='../../data/test', help='path to test dataset')
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--eval-resume', type=str,
                    default='./checkpoint.pth.tar', help='path for eval model')
args = parser.parse_args()

args.save = 'eval-{}'.format(args.save)
if args.local_rank == 0:
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

time.sleep(1)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,format=log_format, datefmt='%m/%d %I:%M:%S %p')

fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))

logging.getLogger().addHandler(fh)

writer = SummaryWriter(logdir=args.save)

CLASSES = 1000

per_epoch_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size
val_iters = IMAGENET_TEST_SET_SIZE // 200


def main():
    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
        
    num_gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)

    logging.info('gpu device = %d' % args.gpu)
    logging.info("args = %s", args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.batch_size = args.batch_size // args.world_size

    # The network architeture coding
    rngs = [int(id) for id in args.model_id.split(' ')]
    model = Network(rngs)
    op_flops_dict = pickle.load(open(config.flops_lookup_table, 'rb'))

    profile(model, config.model_input_size_imagenet, rngs=rngs)
    flops = get_arch_flops(op_flops_dict, rngs,
                           config.backbone_info, config.blocks_keys)
    params = count_parameters_in_MB(model)
    model = model.cuda(args.gpu)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
    arch = model.module.architecture()
    logging.info('rngs:{}, arch:{}'.format(rngs, arch))
    logging.info("flops = %fMB, param size = %fMB", flops/1e6, params)
    logging.info('batch_size:{}'.format(args.batch_size))

    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
        if p.ndimension() == 4 or 'classifier.0.weight' in pname or 'classifier.0.bias' in pname:
            weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(
        filter(lambda p: id(p) not in weight_parameters_id, all_parameters))

    optimizer = torch.optim.SGD(
        [{'params': other_parameters},
         {'params': weight_parameters, 'weight_decay': args.weight_decay}],
        args.learning_rate,
        momentum=args.momentum,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (1.0-step/args.total_iters), last_epoch=-1)

    # Prepare data
    train_loader = get_train_dataloader(
        args.train_dir, args.batch_size, args.local_rank, args.total_iters)
    train_dataprovider = DataIterator(train_loader)
    val_loader = get_val_dataloader(args.test_dir)
    val_dataprovider = DataIterator(val_loader)

    start_iter = 0
    best_acc_top1 = 0
    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
        logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
        checkpoint = torch.load(checkpoint_tar, map_location={
                                'cuda:0': 'cuda:{}'.format(args.local_rank)})
        start_iter = checkpoint['iters']
        best_acc_top1 = checkpoint['best_acc_top1']
        model.load_state_dict(checkpoint['state_dict'])
        logging.info("loaded checkpoint {} iters = {}" .format(
            checkpoint_tar, checkpoint['iters']))

    for iters in range(start_iter):
        scheduler.step()

    # evaluation mode
    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume)
            model.load_state_dict(checkpoint['state_dict'])
            valid_acc_top1, valid_acc_top5 = infer(
                val_dataprovider, model.module, val_iters)
            print('valid_acc_top1: {}'.format(valid_acc_top1))
        exit(0)

    iters = start_iter
    while iters < args.total_iters:
        train_iters = 10000
        train_acc, train_obj, iters = train(
            iters, train_dataprovider, model, criterion_smooth, optimizer, train_iters, scheduler)
        writer.add_scalar('Train/Loss', train_obj, iters)
        writer.add_scalar('Train/LR', scheduler.get_lr()[0], iters)
        if args.local_rank == 0:
            valid_acc_top1, valid_acc_top5 = infer(
                val_dataprovider, model.module, val_iters)

            is_best = False
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True

            logging.info('valid_acc_top1: %f valid_acc_top5: %f best_acc_top1: %f',
                         valid_acc_top1, valid_acc_top5, best_acc_top1)
            save_checkpoint_({
                'iters': iters,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer': optimizer.state_dict(),
            }, args.save)


def train(iters, train_dataprovider, model, criterion, optimizer, train_iters, scheduler):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.train()

    for i in range(train_iters):
        scheduler.step()
        t0 = time.time()
        input, target = train_dataprovider.next()
        datatime = time.time() - t0

        target = target.cuda(args.gpu)
        input = input.cuda(args.gpu)
        input = Variable(input, requires_grad=False)
        target = Variable(target)
        optimizer.zero_grad()
        logits = model(input)
        loss = criterion(logits, target)
        loss_reduce = reduce_tensor(loss, 0, args.world_size)

        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss_reduce.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)
        iters += 1
        if i % args.report_freq == 0 and args.local_rank == 0:
            logging.info('train iters=%03d %e %f %f %f %f', iters, objs.avg,
                         top1.avg, top5.avg, scheduler.get_lr()[0], float(datatime))
    return top1.avg, objs.avg, iters


def infer(val_dataprovider, model, val_iters):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for i in range(val_iters):
            t0 = time.time()
            input, target = val_dataprovider.next()
            datatime = time.time() - t0
            input = Variable(input).cuda(args.gpu)
            target = Variable(target).cuda(args.gpu)
            logits = model(input)

            prec1, prec5 = accuracy(logits, target, topk=(1, 5))
            n = input.size(0)
            top1.update(prec1.data.item(), n)
            top5.update(prec5.data.item(), n)

            if i % args.report_freq == 0:
                logging.info('valid %03d/%03d %f %f %f', i,
                             val_iters, top1.avg, top5.avg, float(datatime))
    return top1.avg, top5.avg


if __name__ == '__main__':
    main()
