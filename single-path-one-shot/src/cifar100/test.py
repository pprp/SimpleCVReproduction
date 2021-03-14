from utils import (AvgrageMeter, CrossEntropyLabelSmooth, accuracy,
                   get_lastest_model, get_parameters, save_checkpoint, ArchLoader)
from slimmable_resnet20 import mutableResNet20
from cifar100_dataset import get_dataset
import argparse
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
import json

torch.autograd.set_detect_anomaly(True)


class OpencvResize(object):

    def __init__(self, size=32):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img)  # (H,W,3) RGB
        img = img[:, :, ::-1]  # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5),
                       self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1]  # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img


class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:, :, ::-1]  # 2 BGR
        img = np.transpose(img, [2, 0, 1])  # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]


def get_args():
    parser = argparse.ArgumentParser("ResNet20-Cifar100-oneshot")
    parser.add_argument('--arch-batch', default=200,
                        type=int, help="arch batch size")
    parser.add_argument(
        '--path', default="Track1_final_archs.json", help="path for json arch files")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str,
                        default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int,
                        default=5120, help='batch size')
    parser.add_argument('--total-iters', type=int,
                        default=15000, help='total iters')
    parser.add_argument('--learning-rate', type=float,
                        default=0.5, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float,
                        default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models',
                        help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float,
                        default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool,
                        default=True, help='report frequency')
    parser.add_argument('--display-interval', type=int,
                        default=20, help='report frequency')
    parser.add_argument('--val-interval', type=int,
                        default=10000, help='report frequency')
    parser.add_argument('--save-interval', type=int,
                        default=10000, help='report frequency')

    parser.add_argument('--train-dir', type=str,
                        default='data/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=str,
                        default='data/val', help='path to validation dataset')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # archLoader
    arch_loader = ArchLoader(args.path)

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

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    train_dataset, val_dataset = get_dataset('cifar100')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                               num_workers=1, pin_memory=use_gpu)
    train_dataprovider = DataIterator(train_loader)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=200, shuffle=False,
                                             num_workers=1, pin_memory=use_gpu)

    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')

    model = mutableResNet20()

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

    model = model.to(device)

    all_iters = 0
    if args.auto_continue:  # 自动进行？？
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(
                lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            print('load from checkpoint')
            for i in range(iters):
                scheduler.step()

    # 参数设置
    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(
                args.eval_resume, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint, strict=True)
            validate(model, device, args, all_iters=all_iters,arch_loader=arch_loader)
        exit(0)

    # while all_iters < args.total_iters:
    #     all_iters = train(model, device, args, val_interval=args.val_interval,
    #                       bn_process=False, all_iters=all_iters, arch_loader=arch_loader, arch_batch=args.arch_batch)
    # all_iters = train(model, device, args, val_interval=int(1280000/args.batch_size), bn_process=True, all_iters=all_iters)
    # save_checkpoint({'state_dict': model.state_dict(),}, args.total_iters, tag='bnps-')

def validate(model, device, args, *, all_iters=None, arch_loader=None):
    assert arch_loader is not None

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    model.eval()
    max_val_iters = 250
    t1 = time.time()

    result_dict = {}

    arch_dict = arch_loader.get_arch_dict()

    with torch.no_grad():
        for key, value in arch_dict.items():
            for _ in range(1, max_val_iters + 1):
                data, target = val_dataprovider.next()
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)

                output = model(data, value["arch"])
                loss = loss_function(output, target)

                prec1, prec5 = accuracy(output, target, topk=(1, 5))
                n = data.size(0)
                objs.update(loss.item(), n)
                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

            result_dict[key] = top1.avg / 100


    print("="*50, "RESULTS", "="*50)
    for key, value in result_dict:
        print(key, "\t", value)
    print("="*50, "E N D", "="*50)


if __name__ == "__main__":
    main()
