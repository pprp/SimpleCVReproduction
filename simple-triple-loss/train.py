import argparse
import os
import random
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torch.nn.functional as F

from config import cfg
from data.tripletFolder import TripleFolder
from utils.visdom import Visualizer
from model.model import Res18

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

train_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

val_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_datasets = TripleFolder(os.path.join(cfg.DATA_PATH, 'train'),
                              transforms=train_transforms)
val_datasets = TripleFolder(os.path.join(cfg.DATA_PATH, 'val'),
                            transforms=val_transforms)
train_dataloader = DataLoader(train_datasets,
                              batch_size=cfg.BATCH_SIZE,
                              shuffle=True,
                              num_workers=0,
                              drop_last=True)
test_dataloader = DataLoader(val_datasets,
                             batch_size=cfg.BATCH_SIZE,
                             shuffle=True,
                             num_workers=0)

class_names = train_datasets.classes
# all labels
# path, label = samples
class_vector = [s[1] for s in train_datasets.samples]

use_gpu = torch.cuda.is_available()


def train(model, criterion, optimizer, scheduler, max_epochs=100):
    vis = Visualizer(env='triplet')
    best_acc = 0.0
    margin = 0.0

    for epoch in range(max_epochs):
        model.train()

        running_loss = 0.0
        running_corrects = 0.0
        running_margin = 0.0
        running_reg = 0.0

        for inputs, labels, pos, pos_labels in train_dataloader:
            bs, c, h, w = inputs.shape

            if bs < cfg.BATCH_SIZE:
                continue

            # pos_bs, c, h, w
            pos = pos.view(4 * cfg.BATCH_SIZE, c, h, w)
            pos_labels = pos_labels.repeat(4).reshape(4, cfg.BATCH_SIZE)
            pos_labels = pos_labels.transpose(0, 1).reshape()
            #[7 7 7 7 10 10 10 10]

            if use_gpu:
                inputs = Variable(inputs.cuda())
                pos = Variable(pos.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward L2 feature and classfication results(conf)
            outputs, f = model(inputs)
            _, pf = model(pos)

            # 通过打乱pos labels来生成neg labels
            neg_labels = pos_labels
            nf_data = pf  # pos features = [bs, channel]
            # 选取pool size个图片作为false
            rand = np.random.permutation(4 * cfg.BATCH_SIZE)[0:cfg.POOL_SIZE]
            # 打乱pos的feature得到neg feature, 找到对应的label
            nf_data = nf_data[rand, :]  # poolsize, num
            neg_labels = neg_labels[rand]  # poolsize
            # f.data:[bs, num] * nf_tsps:[num, poolsize] = [bs,poolsize]
            score = torch.mm(f.data, nf_data.transpose(0, 1))
            # score high -> hard: 按照poolsize找，找到最相似的
            score, rank = score.sort(dim=1, descending=True)

            labels_cpu = labels.cpu()
            nf_hard = torch.zeros(f.shape).cuda()  #bs,num

            for k in range(bs):
                # 抽出一张图片对应的rank eg:[0,1,4,3,2]
                hard = rank[k, :]
                for kk in hard:
                    now_label = neg_labels[kk]
                    anchor_label = labels_cpu[k]
                    # 这部分进行筛选，打乱后不能与原label一致
                    if now_label != anchor_label:
                        nf_hard[k, :] = nf_data[kk, :]
                        # 找到一个就可以
                        break
            # hard pos
            pf_hard = torch.zeros(f.shape).cuda()  #bs,num
            for k in range(bs):
                # 得到4张图
                pf_data = pf[4 * k:4 * k + 4, :]
                pf_t = pf_data.transpose(0, 1)  #num, 4
                ff = f.data[k, :].reshape(1, -1)  #1*num
                score = torch.mm(ff, pf_t)  #[1,num]*[num,4]=[1,4]
                # score low -> hard
                score, rank = score.sort(dim=1, descending=False)
                pf_hard[k, :] = pf_data[rank[0][0], :]

            criterion_triplet = nn.MarginRankingLoss(margin=cfg.MARGIN)
            pscore = torch.sum(f * pf_hard, dim=1)  # [bs,num]*[bs,num]
            nscore = torch.sum(f * nf_hard, dim=1)  # [bs,num]*[bs,1]

            y = torch.ones(bs)
            y = Variable(y.cuda())

            # get loss
            _, preds = torch.max(outputs.data, 1)
            reg = torch.sum((1 + nscore)**2) + torch.sum((-1 + pscore)**2)
            loss = torch.sum(F.relu(nscore + cfg.MARGIN - pscore))
            loss_triplet = loss + cfg.ALPHA * reg

            #optimizer and backward
            optimizer.zero_grad()
            loss_triplet.backward()
            optimizer.step()

            running_loss += loss_triplet.item()
            running_corrects += float(torch.sun(pscore > nscore + cfg.MARGIN))
            running_margin += float(torch.sum(pscore - nscore))
            running_reg += reg

        datasize = len(train_datasets) // cfg.BATCH_SIZE * cfg.BATCH_SIZE
        epoch_loss = running_loss / datasize
        epoch_reg = cfg.ALPHA * running_reg / datasize
        epoch_acc = running_corrects / datasize
        epoch_margin = running_margin / datasize

        #if epoch_acc>0.75:
        #    opt.margin = min(opt.margin+0.02, 1.0)
        print(
            'epoch:%d|margin:%.4f|loss:%.4f|reg:%.4f|acc:%.4f|meanMargin:%.4f'
            % (epoch, cfg.MARGIN, epoch_loss, epoch_reg, epoch_acc,
               epoch_margin))

        if epoch % 10 == 0 and epoch != 0:
            model.save(os.path.join(cfg.SAVE_PATH, "%d.pth" % epoch))

if __name__ == "__main__":
    model = Res18(len(class_names))
    if use_gpu:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss()

    ignored_params = list(map(id, model.model.fc.parameters()))
    ignored_params += (
        list(map(id, model.classifier0.parameters())) +
        list(map(id, model.classifier1.parameters())) +
        list(map(id, model.classifier2.parameters())) +
        list(map(id, model.classifier3.parameters())) +
        list(map(id, model.classifier4.parameters())) +
        list(map(id, model.classifier5.parameters()))
        #+list(map(id, model.classifier6.parameters() ))
        #+list(map(id, model.classifier7.parameters() ))
    )
    base_params = filter(lambda p: id(p) not in ignored_params,
                         model.parameters())
    optimizer = optim.SGD(
        [
            {
                'params': base_params,
                'lr': 0.001
            },
            {
                'params': model.model.fc.parameters(),
                'lr': 0.01
            },
            {
                'params': model.classifier0.parameters(),
                'lr': 0.01
            },
            {
                'params': model.classifier1.parameters(),
                'lr': 0.01
            },
            {
                'params': model.classifier2.parameters(),
                'lr': 0.01
            },
            {
                'params': model.classifier3.parameters(),
                'lr': 0.01
            },
            {
                'params': model.classifier4.parameters(),
                'lr': 0.01
            },
            {
                'params': model.classifier5.parameters(),
                'lr': 0.01
            },
            #{'params': model.classifier6.parameters(), 'lr': 0.01},
            #{'params': model.classifier7.parameters(), 'lr': 0.01}
        ],
        weight_decay=5e-4,
        momentum=0.9,
        nesterov=True)
    scheduler = lr_scheduler.MultiStepLR(optimizer,
                                         milestones=[20, 40],
                                         gamma=0.1)

    train(model, criterion, optimizer, scheduler, max_epochs=cfg.TOTAL_EPOCH)
