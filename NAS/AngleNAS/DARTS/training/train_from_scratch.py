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
import genotypes
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from model import NetworkImageNet as Network
from tensorboardX import SummaryWriter
import apex

sys.path.append("../..")
from utils import *
from thop import profile

IMAGENET_TRAINING_SET_SIZE = 1281167
IMAGENET_TEST_SET_SIZE = 50000

parser = argparse.ArgumentParser("training imagenet")
parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
parser.add_argument('--workers', type=int, default=32, help='number of workers to load dataset')
parser.add_argument('--batch_size', type=int, default=512, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.25, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='PDARTS_ABS', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PDARTS_ABS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--train_dir', type=str, default='../../data/train', help='path to training dataset')
parser.add_argument('--test_dir', type=str, default='../../data/test', help='path to test dataset')
parser.add_argument('--eval', default=False, action='store_true')
parser.add_argument('--eval-resume', type=str, default='./checkpoint.pth.tar', help='path for eval model')

args, unparsed = parser.parse_known_args()
args.save = 'eval-{}'.format(args.save)
if args.local_rank == 0 and not os.path.exists(args.save):
    create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

time.sleep(1)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
writer = SummaryWriter(logdir=args.save)

CLASSES = 1000
per_epoch_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size
val_iters =  IMAGENET_TEST_SET_SIZE // 200

# Average loss across processes for logging.
def reduce_tensor(tensor, device=0, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, device)
    tensor.div_(world_size)
    return tensor

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)

    num_gpus = torch.cuda.device_count()  
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)

    np.random.seed(args.seed)
    cudnn.benchmark = True
    cudnn.deterministic = True

    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)

    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    args.world_size = torch.distributed.get_world_size()
    args.batch_size = args.batch_size // args.world_size

    genotype = eval("genotypes.%s" % args.arch)
    logging.info('---------Genotype---------')
    logging.info(genotype)
    logging.info('--------------------------') 
    model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model = model.cuda(args.gpu)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    model_profile = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    model_profile = model_profile.cuda(args.gpu)
    model_input_size_imagenet = (1, 3, 224, 224)
    model_profile.drop_path_prob = 0
    flops, _ = profile(model_profile, model_input_size_imagenet)
    logging.info("flops = %fMB, param size = %fMB", flops, count_parameters_in_MB(model))

    criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()

    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    # Prepare data
    total_iters = per_epoch_iters * args.epochs
    train_loader = get_train_dataloader(args.train_dir, args.batch_size, args.local_rank, total_iters)
    train_dataprovider = DataIterator(train_loader)
    val_loader = get_val_dataloader(args.test_dir)
    val_dataprovider = DataIterator(val_loader)
  
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    start_epoch = 0
    best_acc_top1 = 0
    best_acc_top5 = 0
    checkpoint_tar = os.path.join(args.save, 'checkpoint.pth.tar')
    if os.path.exists(checkpoint_tar):
      logging.info('loading checkpoint {} ..........'.format(checkpoint_tar))
      checkpoint = torch.load(checkpoint_tar, map_location={'cuda:0':'cuda:{}'.format(args.local_rank)})
      start_epoch = checkpoint['epoch'] + 1
      model.load_state_dict(checkpoint['state_dict'])
      logging.info("loaded checkpoint {} epoch = {}" .format(checkpoint_tar, checkpoint['epoch']))

    # evaluation mode
    if args.eval:
      if args.eval_resume is not None:
          checkpoint = torch.load(args.eval_resume)
          model.module.drop_path_prob = 0
          model.load_state_dict(checkpoint['state_dict'])
          valid_acc_top1, valid_acc_top5 = infer(val_dataprovider, model.module, val_iters)
          print('valid_acc_top1: {}'.format(valid_acc_top1))
      exit(0)

    for epoch in range(start_epoch, args.epochs):
        if args.lr_scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_lr()[0]
        elif args.lr_scheduler == 'linear':
            current_lr = adjust_lr(optimizer, epoch)
        else:
            logging.info('Wrong lr type, exit')
            sys.exit(1)

        logging.info('Epoch: %d lr %e', epoch, current_lr)
        if epoch < 5 and args.batch_size > 256:
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr * (epoch + 1) / 5.0
            logging.info('Warming-up Epoch: %d, LR: %e', epoch, current_lr * (epoch + 1) / 5.0)
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        epoch_start = time.time()
        train_acc, train_obj = train(train_dataprovider, model, criterion_smooth, optimizer, per_epoch_iters)

        writer.add_scalar('Train/Loss', train_obj, epoch)
        writer.add_scalar('Train/LR', current_lr, epoch)

        if args.local_rank == 0 and (epoch % 5 == 0 or args.epochs - epoch < 10) :
            valid_acc_top1, valid_acc_top5 = infer(val_dataprovider, model.module, val_iters)
            is_best = False
            if valid_acc_top5 > best_acc_top5:
                best_acc_top5 = valid_acc_top5
            if valid_acc_top1 > best_acc_top1:
                best_acc_top1 = valid_acc_top1
                is_best = True
            
            logging.info('Valid_acc_top1: %f', valid_acc_top1)
            logging.info('Valid_acc_top5: %f', valid_acc_top5)
            logging.info('best_acc_top1: %f', best_acc_top1)
            epoch_duration = time.time() - epoch_start
            logging.info('Epoch time: %ds.', epoch_duration)

            save_checkpoint_({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc_top1': best_acc_top1,
                'optimizer' : optimizer.state_dict(),
                }, args.save)
        
def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        

def train(train_dataprovider, model, criterion, optimizer, train_iters):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    batch_time = AvgrageMeter()
    model.train()

    for i in range(train_iters):
        t0 = time.time()
        input, target = train_dataprovider.next()
        datatime = time.time() - t0
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        b_start = time.time()
        optimizer.zero_grad()
        logits, logits_aux = model(input)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss_reduce = reduce_tensor(loss, 0, args.world_size)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        batch_time.update(time.time() - b_start)
        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        objs.update(loss_reduce.data.item(), n)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if i % args.report_freq == 0 and args.local_rank == 0:
            logging.info('TRAIN Step: %03d/%03d Objs: %e R1: %f R5: %f BTime: %.3fs Datatime: %.3f', 
                                    i, train_iters, objs.avg, top1.avg, top5.avg, batch_time.avg, float(datatime))
    return top1.avg, objs.avg


def infer(val_dataprovider, model, val_iters):
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()
    model.eval()

    for i in range(val_iters):
        t0 = time.time()
        input, target = val_dataprovider.next()
        datatime = time.time() - t0
        input = input.cuda()
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)

        prec1, prec5 = accuracy(logits, target, topk=(1, 5))
        n = input.size(0)
        top1.update(prec1.data.item(), n)
        top5.update(prec5.data.item(), n)

        if i % args.report_freq == 0:
            logging.info('VALID Step: %03d/%03d R1: %f R5: %f Datatime: %.3f', i, val_iters, top1.avg, top5.avg, float(datatime))

    return top1.avg, top5.avg


if __name__ == '__main__':
    main() 
