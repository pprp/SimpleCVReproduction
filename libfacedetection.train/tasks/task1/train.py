#!/usr/bin/python3
from __future__ import print_function

import argparse
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data

from config import cfg
from data import FaceRectLMDataset, detection_collate
from multibox_loss import MultiBoxLoss
from prior_box import PriorBox
from yufacedetectnet import YuFaceDetectNet

sys.path.append(os.getcwd() + '/../../src')


parser = argparse.ArgumentParser(description='YuMobileNet Training')
parser.add_argument('--training_face_rect_dir',
                    default='../../data/WIDER_FACE_rect', help='Training dataset directory')
parser.add_argument('--training_face_landmark_dir',
                    default='../../data/WIDER_FACE_landmark', help='Training dataset directory')
parser.add_argument('-b', '--batch_size', default=16,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--gpu_ids', default='0', help='the IDs of GPU')
parser.add_argument('--lr', '--learning-rate', default=1e-3,
                    type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None,
                    help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='resume iter for retraining')
parser.add_argument('-max', '--max_epoch', default=500,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--weight_filename_prefix', default='weights/yunet',
                    help='the prefix of the weight filename')
args = parser.parse_args()


img_dim = 320  # only 1024 is supported
rgb_mean = (0, 0, 0)  # (104, 117, 123) # bgr order
num_classes = 2
gpu_ids = [int(item) for item in args.gpu_ids.split(',')]
num_workers = args.num_workers
batch_size = args.batch_size
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = args.lr
gamma = args.gamma
max_epoch = args.max_epoch
training_face_rect_dir = args.training_face_rect_dir
training_face_landmark_dir = args.training_face_landmark_dir

net = YuFaceDetectNet('train', img_dim)
print("Printing net...")
# print(net)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if len(gpu_ids) > 1:
    net = torch.nn.DataParallel(net, device_ids=gpu_ids)

#device = torch.device(args.device)
device = torch.device('cuda:'+str(gpu_ids[0]))
cudnn.benchmark = True
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=initial_lr,
                      momentum=momentum, weight_decay=weight_decay)
criterion = MultiBoxLoss(num_classes, 0.35, True, 0,
                         True, 3, 0.35, False, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.to(device)


def train():

    net.train()

    # load the two dataset for face rectangles and landmarks respectively
    print('Loading Dataset...')
    dataset_rect = FaceRectLMDataset(training_face_rect_dir, img_dim, rgb_mean)
    dataset_landmark = FaceRectLMDataset(
        training_face_landmark_dir, img_dim, rgb_mean)

    for epoch in range(args.resume_epoch, max_epoch):
        if epoch < 100:
            with_landmark = False
        else:
            with_landmark = (epoch % 2 == 1)

        dataset = dataset_rect
        if with_landmark:
            dataset = dataset_landmark

        epoch_size = math.ceil(len(dataset) / batch_size)
        lr = adjust_learning_rate_poly(optimizer, epoch, max_epoch)

        # for computing average losses in the newest batch_iterator iterations
        # to make the loss to be smooth
        loss_l_epoch = []
        loss_lm_epoch = []
        loss_c_epoch = []
        loss_epoch = []

        # the start time
        load_t0 = time.time()

        # create batch iterator
        batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True,
                                              num_workers=num_workers, collate_fn=detection_collate))
        # for each iteration in this epoch
        for iteration in range(epoch_size):

            # load train data
            images, targets = next(batch_iterator)
            images = images.to(device)
            targets = [anno.to(device) for anno in targets]

            # forward
            out = net(images)

            # backprop
            optimizer.zero_grad()
            loss_l, loss_lm, loss_c = criterion(out, priors, targets)

            loss = 0
            if with_landmark:
                loss = loss_l + loss_lm + loss_c
            else:
                loss = loss_l + loss_c

            loss.backward()
            optimizer.step()

            if (len(loss_l_epoch) >= epoch_size):
                del loss_l_epoch[0]
            if (len(loss_lm_epoch) >= epoch_size):
                del loss_lm_epoch[0]
            if (len(loss_c_epoch) >= epoch_size):
                del loss_c_epoch[0]
            if (len(loss_epoch) >= epoch_size):
                del loss_epoch[0]

            loss_l_epoch.append(loss_l.item())
            mean_l_loss = np.mean(loss_l_epoch)
            loss_lm_epoch.append(loss_lm.item())
            mean_lm_loss = np.mean(loss_lm_epoch)
            loss_c_epoch.append(loss_c.item())
            mean_c_loss = np.mean(loss_c_epoch)
            loss_epoch.append(loss.item())
            mean_loss = np.mean(loss_epoch)

            if (iteration % 10 == 0):
                print('LM:{} || Epoch:{}/{} || Epochiter: {}/{} || L: {:.2f}({:.2f}) LM: {:.2f}({:.2f}) C: {:.2f}({:.2f}) All: {:.2f}({:.2f}) || LR: {:.8f}'.format(with_landmark,
                                                                                                                                                                    epoch, max_epoch, iteration, epoch_size, loss_l.item(), mean_l_loss, loss_lm.item(), mean_lm_loss, loss_c.item(), mean_c_loss, loss.item(), mean_loss, lr))
                #print('time=', (time.time()-load_t0)/60)

        if (epoch % 10 == 1 and epoch > 1):
            torch.save(net.state_dict(), args.weight_filename_prefix +
                       '_epoch_' + str(epoch) + '.pth')

        # the end time
        load_t1 = time.time()
        epoch_time = (load_t1 - load_t0) / 60
        print('Epoch time: {:.2f} minutes; Time left: {:.2f} hours'.format(
            epoch_time, (epoch_time)*(max_epoch-epoch-1)/60))

    torch.save(net.state_dict(), args.weight_filename_prefix + '_final.pth')


def adjust_learning_rate_poly(optimizer, iteration, max_iter):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = initial_lr * (1 - (iteration / max_iter)) * \
        (1 - (iteration / max_iter))
    if (lr < 1.0e-7):
        lr = 1.0e-7
    return lr


if __name__ == '__main__':
    train()
