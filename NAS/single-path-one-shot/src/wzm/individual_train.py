'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import logging

from mobilenetv2 import MobileNetLike
from utils import progress_bar
from coder import GeneCoder

import nni
import csv

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def cal_similar(list_a,list_b):
    count=0
    for a,b in zip(list_a,list_b):
        if a==b and a!=0:
            count+=1
    return count


# Training
def train(epoch,net,optimizer,trainloader,criterion,scheduler):

    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    iters=len(trainloader)
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + batch_idx / iters)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        if batch_idx % 100==0:
            print('Train : | Loss: %.6f | Acc: %.6f%% (%d/%d)'% (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    print('Epoch Train Loss: %.6f | Epoch Train Acc: %.6f%% (%d/%d)' % (train_loss / len(trainloader), 100. * correct / total, correct, total))

def test(net,valloader,criterion):

    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = 100.*correct/total
            if batch_idx% 100==0:
                print('Test : | Loss: %.6f | Acc: %.6f%% (%d/%d)' % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Epoch Test Loss: %.6f | Epoch Test Acc: %.6f%% (%d/%d)' % ( test_loss / len(valloader), 100. * correct / total, correct, total))
    return acc



def train_arch(cfg, num_epoches, individual_id, xargs,  arch_set):
    best_acc=0
    max_similar=0
    father_param=None
    father_id=None
    print('\n')
    print('==> train and val for the {}-th individual ..'.format(individual_id))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)


    # Model
    print('==> Building model..')
    net = MobileNetLike(cfg)
    net = torch.nn.DataParallel(net)
    net = net.to(device)

    optimizer = optim.SGD(net.parameters(), lr=0.05,
                          momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoches)

    if individual_id > 210:
        for idx,arch in enumerate(arch_set):
            count_similar = cal_similar(arch, cfg)
            if count_similar >= max_similar:
                max_similar = count_similar
                father_param = arch
                father_id = idx+1
        if max_similar != 0:
            print('father param is: ', father_param)
            print('this param is: ', cfg)
            print('similar count is:',max_similar)
            path = os.path.join(xargs.m_path,'{}.pth'.format(father_id))
            save_model = torch.load(path)
            save_state_dict = {k.replace('module.', ''): v for k, v in save_model['net'].items()}
            model_dict = net.state_dict()
            ss = {k: v for k, v in save_state_dict.items() if
                  k in model_dict.keys() and v.size() == model_dict[k].size()}
            model_dict.update(ss)
            net.load_state_dict(model_dict)


    criterion = nn.CrossEntropyLoss()
    for epoch in range(num_epoches):
        train(epoch,net,optimizer,trainloader,criterion,scheduler)
        acc = test(net,testloader,criterion)
        if acc > best_acc:
            best_acc = acc
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'param': cfg,
                'acc': best_acc,
            }
            torch.save(state, os.path.join(xargs.m_path,'{}.pth'.format(individual_id)))

    print('Finished. Best acc of {}-th individual is {}'.format(individual_id,best_acc))
    return best_acc


#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--epochs", type=int, default=100)
#     parser.add_argument("--train_portion", type=float, default=0.8)
#     parser.add_argument("--save_path", type=str, default='save/model_checkpoint')
#     parser.add_argument('--lr', default=0.05, type=float, help='learning rate')
#     args, _ = parser.parse_known_args()
#
#     json_path = "search_space.json"
#     coder_ = GeneCoder(json_path)
#
#     RCV_CONFIG = nni.get_next_parameter()
#     trial_id=nni.get_sequence_id()
#
#     best_acc = 0
#
#     print('==> train and val for the {}-th individual ..'.format(trial_id))
#     transform_train = transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
#
#     trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
#
#     num_train = len(trainset)
#     indices = list(range(num_train))
#     split = int(np.floor(args.train_portion * num_train))
#
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=128,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
#         pin_memory=True, num_workers=4)
#
#     valloader = torch.utils.data.DataLoader(
#         trainset, batch_size=100,
#         sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
#         pin_memory=True, num_workers=4)
#
#     # Model
#     print('==> Building model..')
#     net = MobileNetLike(RCV_CONFIG)
#     net = torch.nn.DataParallel(net)
#     net = net.to(device)
#
#
#
#     optimizer = optim.SGD(net.parameters(), lr=args.lr,
#                           momentum=0.9, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
#
#     criterion = nn.CrossEntropyLoss()
#     for epoch in range(args.epochs):
#         train(epoch, net, optimizer, trainloader, criterion, scheduler)
#         acc = test(net, valloader, criterion)
#         nni.report_intermediate_result(acc)
#         if acc > best_acc:
#             best_acc = acc
#             print('Saving..')
#             state = {
#                 'net': net.state_dict(),
#                 'param': RCV_CONFIG,
#                 'acc': best_acc,
#             }
#             torch.save(state, os.path.join(args.save_path, '{}.pth'.format(trial_id)))
#     print('Finished. Best acc is: ', best_acc)
#     nni.report_final_result(best_acc)
#
#
#
#

