import os
import sys
import time
import numpy as np
import torch
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from super_model import SuperNetwork
from train import *
from search import *
from config import config
import shutil
import functools
print=functools.partial(print,flush=True)
import pickle
import apex

sys.path.append("../..")
from utils import *

IMAGENET_TRAINING_SET_SIZE = 1231167
IMAGENET_TEST_SET_SIZE = 50000

parser = argparse.ArgumentParser("ImageNet")
parser.add_argument('--local_rank', type=int, default=None, help='local rank for distributed training')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=4e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=30, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--total_iters', type=int, default=300000, help='total iters')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--classes', type=int, default=1000, help='number of classes')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--train_dir', type=str, default='../../data/train', help='path to training dataset')
parser.add_argument('--test_dir', type=str, default='../../data/test', help='path to test dataset')
parser.add_argument('--operations_path', type=str, default='../shrinking/shrunk_search_space.pt', help='shrunk search space')
args = parser.parse_args()

per_epoch_iters = IMAGENET_TRAINING_SET_SIZE // args.batch_size
val_iters =  IMAGENET_TEST_SET_SIZE // 200

def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    num_gpus = torch.cuda.device_count() 
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    group_name = 'search'
    print('gpu device = %d' % args.gpu)
    print("args = %s", args)

    torch.distributed.init_process_group(backend='nccl', init_method='env://', group_name = group_name)
    args.world_size = torch.distributed.get_world_size()
    args.batch_size = args.batch_size // args.world_size
    
    criterion_smooth = CrossEntropyLabelSmooth(args.classes, args.label_smooth)
    criterion_smooth = criterion_smooth.cuda()
    model = SuperNetwork()
    args.layers = len(model.features)
    model = model.cuda(args.gpu)
    model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
      if p.ndimension() == 4 or 'classifier.0.weight' in pname or 'classifier.0.bias' in pname:
          weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    optimizer = torch.optim.SGD(
        [{'params' : other_parameters},
        {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
        args.learning_rate,
        momentum=args.momentum,
        )

    operations = pickle.load(open(args.operations_path, 'rb'))
    print('operations={}'.format(operations))
    args.total_iters = args.epochs * per_epoch_iters // len(operations[0])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/args.total_iters), last_epoch=-1)

    # Prepare data
    train_loader = get_train_dataloader(args.train_dir, args.batch_size, args.local_rank, args.total_iters)
    train_dataprovider = DataIterator(train_loader)
    val_loader = get_val_dataloader(args.test_dir)
    val_dataprovider = DataIterator(val_loader)

    train(train_dataprovider, val_dataprovider, optimizer, scheduler, model, criterion_smooth, args, val_iters, args.seed, operations)
    
    if args.local_rank == 0:
        save(model.module, config.net_cache)
        evolution_trainer = EvolutionTrainer()
        topk = evolution_trainer.search(operations)
        now = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        save_checkpoint({
        'topk':topk,
        'state_dict': model.state_dict(),
        }, config.checkpoint_cache)
        topk_str = get_topk_str(topk)
        print('{} |=> topk = {}, topk_str={}'.format(now, topk, topk_str))

if __name__ == '__main__':
  main() 

