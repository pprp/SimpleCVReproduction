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
from tqdm import tqdm
from torch.utils.data.distributed import DistributedSampler
import torch.backends.cudnn as cudnn

from datasets.cifar100_dataset import get_val_dataset, ArchDataSet, get_train_loader
# from model.slimmable_resnet20 import mutableResNet20
from model.dynamic_resnet20 import dynamic_resnet20
from utils.utils import (ArchLoader, AvgrageMeter,
                         CrossEntropyLabelSmooth, DataIterator,
                         accuracy, bn_calibration_init,
                         get_lastest_model, get_parameters, retrain_bn,
                         save_checkpoint)
from utils.angle import generate_angle


def get_args():
    parser = argparse.ArgumentParser("ResNet20-Cifar100-oneshot-Test")

    parser.add_argument('--rank', default=0,
                        help='rank of current process')
    parser.add_argument(
        '--path', default="data/Track1_final_archs.json", help="path for json arch files")
    parser.add_argument('--batch-size', type=int,
                        default=2048, help='batch size')
    parser.add_argument('--workers', type=int,
                        default=6, help='num of workers')
    parser.add_argument('--weights', type=str,
                        default="./weights/checkpoint-latest.pth.tar", help="path for weights loading")

    parser.add_argument('--auto-continue', type=bool,
                        default=True, help='report frequency')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='local rank for distributed training')

    parser.add_argument('--min_lr', type=float,
                        default=5e-4, help='min learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float,
                        default=4e-5, help='weight decay')
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument('--seed', type=int, default=5, help='random seed')
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    num_gpus = torch.cuda.device_count()
    np.random.seed(args.seed)
    args.gpu = args.local_rank % num_gpus
    torch.cuda.set_device(args.gpu)

    # cudnn.benchmark = True
    # cudnn.deterministic = True
    # torch.manual_seed(args.seed)
    # cudnn.enabled = True
    # torch.cuda.manual_seed(args.seed)

    if num_gpus > 1:
        torch.distributed.init_process_group(
            backend='nccl', init_method='env://')
        args.world_size = torch.distributed.get_world_size()
        args.batch_size = args.batch_size // args.world_size

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join(
        'log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    # archLoader
    arch_loader = ArchLoader(args.path)
    arch_dataset = ArchDataSet(args.path)
    arch_sampler = None
    if num_gpus > 1:
        arch_sampler = DistributedSampler(arch_dataset)

    arch_dataloader = torch.utils.data.DataLoader(
        arch_dataset, batch_size=1, shuffle=False, num_workers=3, pin_memory=True, sampler=arch_sampler)

    val_dataset = get_val_dataset()
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=args.workers, pin_memory=True)
    train_loader = get_train_loader(
        batch_size=args.batch_size, local_rank=0, num_workers=args.workers)

    print('load data successfully')

    model = dynamic_resnet20()

    print("load model successfully")

    print('load from latest checkpoint')
    lastest_model = args.weights
    if lastest_model is not None:
        checkpoint = torch.load(
            lastest_model, map_location=None if True else 'cpu')
        model.load_state_dict(checkpoint['state_dict'])

    model = model.cuda(args.gpu)
    if num_gpus > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)

    # 参数设置
    args.val_dataloader = val_loader

    print("start to validate model...")

    validate(model, train_loader, args, arch_loader=arch_dataloader)
    # angle_validate(model, arch_loader, args)


def angle_validate(model, archloader, args):
    arch_dict = archloader.get_arch_dict()
    base_model = dynamic_resnet20().cuda(args.gpu)
    # base_2 = dynamic_resnet20().cuda(args.gpu)

    angle_result_dict = {}

    with torch.no_grad():
        for key, value in arch_dict.items():
            angle = generate_angle(base_model, model, value["arch"])
            tmp_dict = {}
            tmp_dict['arch'] = value['arch']
            tmp_dict['acc'] = angle.item()

            print("angle: ", angle.item())

            angle_result_dict[key] = tmp_dict

    print('\n', "="*10, "RESULTS", "="*10)
    for key, value in angle_result_dict.items():
        print(key, "\t", value)
    print("="*10, "E N D", "="*10)

    with open("angle_result.json", "w") as f:
        json.dump(angle_result_dict, f)


def validate(model, train_loader, args, *, arch_loader=None):
    assert arch_loader is not None

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    val_dataloader = args.val_dataloader

    t1 = time.time()

    result_dict = {}

    arch_loader = tqdm(arch_loader)
    for key, arch in arch_loader:
        arch_list = [int(itm) for itm in arch[0].split('-')]
        # bn calibration
        model.apply(bn_calibration_init)

        model.train()

        t1 = time.time()

        for idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.cuda(args.gpu, non_blocking=True), targets.cuda(
                args.gpu, non_blocking=True)
            outputs = model(inputs, arch_list)
            del inputs, targets, outputs
            if idx > 2:
                break

        # print("bn calibration time:", time.time()-t1)

        t2 = time.time()

        model.eval()

        for data, target in val_dataloader:  # 过一遍数据集
            target = target.type(torch.LongTensor)
            data, target = data.cuda(args.gpu, non_blocking=True), target.cuda(
                args.gpu, non_blocking=True)

            output = model(data, arch_list)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))

            n = data.size(0)

            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

        tmp_dict = {}
        tmp_dict['arch'] = arch[0]
        tmp_dict['acc'] = top1.avg / 100

        # print("val time:", time.time() - t2)

        result_dict[key[0]] = tmp_dict

        post_fix = {"top1": "%.6f" % (top1.avg/100)}
        arch_loader.set_postfix(log=post_fix)

    with open("acc_%s.json" % (args.path.split('/')[1].split('.')[0]), "w") as f:
        json.dump(result_dict, f)


if __name__ == "__main__":
    main()
