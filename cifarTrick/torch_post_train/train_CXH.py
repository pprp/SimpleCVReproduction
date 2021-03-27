'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
# from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from cutout import Cutout
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn
from sam import SAM

import torchvision
import torchvision.transforms as transforms

from autoaugmentation import CIFAR10Policy
from labelsmooth import CrossEntropyLabelSmooth

import os
import sys
import argparse
import logging
import time
from cxh import *
from utils import progress_bar


import nni

# sys.stdout = Logger()

trainloader = None
testloader = None
net = None
criterion = None
optimizer = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0.0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def prepare(args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global scheduler

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
        Cutout(n_holes=args.n_holes, length=args.length)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=512, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=4)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = CXH()#CXH_Squeeze_Excitation() #CXH()

    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    # criterion = CrossEntropyLabelSmooth(10)
    # optimizer = optim.SGD(net.parameters(), lr=0.1,
    #                       momentum=0.9, weight_decay=5e-4)


    optimizer = SAM(net.parameters(), torch.optim.SGD, lr=0.1,momentum=0.9)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 1, 2)

# Training


def train(epoch, args):
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer
    global scheduler

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iters = len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # inputs, targets_a, targets_b, lam = mixup_data(inputs, targets,args.alpha)
        # inputs, targets_a, targets_b = map(Variable, (inputs,targets_a, targets_b))
        
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        # loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        train_loss += loss.data
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # (lam * predicted.eq(targets_a.data).cpu().sum().float()
        # + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())

        optimizer.zero_grad()
        loss.backward()
        # optimizer.step()
        optimizer.first_step(zero_grad=True)
        

        loss = criterion(net(inputs),targets)
        loss.backward()
        optimizer.second_step()


        scheduler.step(epoch + batch_idx / iters)

        acc = 100.*correct/total
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def test(epoch):
    global best_acc
    global trainloader
    global testloader
    global net
    global criterion
    global optimizer

    # log_dir = os.path.join('tensorboard', 'test')
    # test_writer = SummaryWriter(log_dir=log_dir)

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            acc = 100.*correct/total
            # test_writer.add_scalar('Accuracy', acc, epoch*len(testloader) + batch_idx)
            # test_writer.add_scalar('Loss', test_loss/(batch_idx+1), epoch*len(testloader) + batch_idx)
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
    return acc, best_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument('--alpha', default=1., type=float,
                        help='mixup interpolation coefficient (default: 1)')
    parser.add_argument('--n_holes', type=int, default=1,
                        help='number of holes to cut out from image')
    parser.add_argument('--length', type=int, default=16,
                        help='length of the holes')
    args, _ = parser.parse_known_args()
    prepare(args)
    acc = 0.0
    best_acc = 0.0
    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch, args)
        acc, best_acc = test(epoch)
        nni.report_intermediate_result(acc)
    print('best acc is:', best_acc)
    nni.report_final_result(best_acc)
