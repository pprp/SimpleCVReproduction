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
from model.slimmable_resnet20 import mutableResNet20
from utils.utils import (ArchLoader, AvgrageMeter, CrossEntropyLabelSmooth, accuracy,
                         get_lastest_model, get_parameters, save_checkpoint, bn_calibration_init)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"


def get_args():
    parser = argparse.ArgumentParser("ResNet20-Cifar100-oneshot")
    parser.add_argument(
        '--path', default="Track1_final_archs.json", help="path for json arch files")
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str,
                        default='./snet_detnas.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int,
                        default=163840, help='batch size')

    parser.add_argument('--save', type=str, default='./weights',
                        help='path for saving trained weights')
    parser.add_argument('--label-smooth', type=float,
                        default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', type=bool,
                        default=True, help='report frequency')

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

    _, val_dataset = get_dataset('cifar100')

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=200, shuffle=False,
                                             num_workers=3, pin_memory=use_gpu)

    print('load data successfully')

    model = mutableResNet20()

    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)

    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    model = model.to(device)
    print("load model successfully")

    all_iters = 0
    print('load from latest checkpoint')
    lastest_model, iters = get_lastest_model()
    if lastest_model is not None:
        all_iters = iters
        checkpoint = torch.load(
            lastest_model, map_location=None if use_gpu else 'cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=True)

    # 参数设置
    args.loss_function = loss_function
    args.val_dataloader = val_loader

    print('clear bn statics....')
    model.apply(bn_calibration_init)

    print('train bn with training set (BN sanitize) ....')
    model.train()

    for step in tqdm.tqdm(range(10)):
        # print('train step: {} total: {}'.format(step,max_train_iters))
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device)
        output = model(data, cand)
        del data, target, output

    print("start to validate model")

    validate(model, device, args, all_iters=all_iters, arch_loader=arch_loader)


def validate(model, device, args, *, all_iters=None, arch_loader=None):
    assert arch_loader is not None

    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataloader = args.val_dataloader

    model.eval()
    model.apply(bn_calibration_init)

    max_val_iters = 0
    t1 = time.time()

    result_dict = {}

    arch_dict = arch_loader.get_arch_dict()

    base_model = mutableResNet20().cuda()

    with torch.no_grad():
        for key, value in arch_dict.items():  # 每一个网络
            max_val_iters += 1
            print('\r ', key, ' iter:', max_val_iters, end='')

            for data, target in val_dataloader:  # 过一遍数据集
                target = target.type(torch.LongTensor)
                data, target = data.to(device), target.to(device)

                output = model(data, value["arch"])

                prec1, prec5 = accuracy(output, target, topk=(1, 5))

                n = data.size(0)

                top1.update(prec1.item(), n)
                top5.update(prec5.item(), n)

            print("\t acc1: ", top1.avg)
            tmp_dict = {}
            tmp_dict['arch'] = value['arch']
            tmp_dict['acc'] = top1.avg

            result_dict[key] = tmp_dict

    with open("acc_result.json", "w") as f:
        json.dump(result_dict, f)

    # angle_result_dict = {}

    # with torch.no_grad():
    #     for key, value in arch_dict.items():
    #         angle = generate_angle(base_model, model.module, value["arch"])
    #         tmp_dict = {}
    #         tmp_dict['arch'] = value['arch']
    #         tmp_dict['acc'] = angle.item()

    #         print("angle: ", angle.item())

    #         angle_result_dict[key] = tmp_dict

    # print('\n', "="*10, "RESULTS", "="*10)
    # for key, value in result_dict.items():
    #     print(key, "\t", value)
    # print("="*10, "E N D", "="*10)

    # with open("angle_result.json", "w") as f:
    #     json.dump(angle_result_dict, f)


if __name__ == "__main__":
    main()
