import argparse
import os
import random
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from config import cfg
from model.models import ResNet18

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

test_transforms = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

gallery_datasets = datasets.ImageFolder(os.path.join(cfg.DATA_PATH, "gallery"),
                                        transform=test_transforms)
gallery_dataloader = DataLoader(gallery_datasets,
                                batch_size=cfg.BATCH_SIZE,
                                drop_last=False,
                                shuffle=False,
                                num_workers=1)
use_gpu = torch.cuda.is_available()
class_names = gallery_datasets.classes


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3) - 1, -1, -1).long()
    img_flip = img.index_select(3, inv_idx)  # flip along w
    return img_flip


def extract_single_image(model, img_path):
    img = cv2.imread(img_path)
    img = torch.FloatTensor(img)
    img = img.unsqueeze(0)
    img = img.permute(0, 3, 1, 2)
    bs, c, h, w = img.size()
    ff = torch.FloatTensor(1, 512).zero_()
    for i in range(2):
        if i == 1:
            img = fliplr(img)
        if use_gpu:
            input_img = Variable(img.cuda())
        else:
            input_img = Variable(img)

        outputs, feature = model(input_img)
        feature = feature.data.cpu()
        ff = ff + feature
    fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
    ff = ff.div(fnorm.expand_as(ff))
    return ff


def extract_features(model, dataloader):
    features = torch.FloatTensor()
    count = 0
    for data in dataloader:
        img, label = data
        bs, c, h, w = img.size()
        count += bs
        ff = torch.FloatTensor(bs, 512).zero_()
        print(count, end='\r')
        sys.stdout.flush()
        # add two features
        for i in range(2):
            if i == 1:
                img = fliplr(img)
            if use_gpu:
                input_img = Variable(img.cuda())
            else:
                input_img = Variable(img)
            outputs, feature = model(input_img)
            feature = feature.data.cpu()
            ff = ff + feature
        # norm features
        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))

        features = torch.cat((features, ff), 0)
    return features


def get_label(img_path):
    labels = []
    for path, _ in img_path:
        filename = os.path.basename(path)
        label = filename.split('_')[0]
        if label[0:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
    return labels


def plot_images(imgs, paths=None, fname='images.jpg'):
    # Plots training images overlaid with targets
    imgs = imgs.cpu().numpy()
    # targets = targets.cpu().numpy()
    # targets = targets[targets[:, 1] == 21]  # plot only one class

    fig = plt.figure(figsize=(10, 10))
    bs, _, h, w = imgs.shape  # batch size, _, height, width
    bs = min(bs, 16)  # limit plot to 16 images
    ns = np.ceil(bs**0.5)  # number of subplots

    for i in range(bs):
        # boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
        # boxes[[0, 2]] *= w
        # boxes[[1, 3]] *= h
        plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
        # plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
        plt.axis('off')
        if paths is not None:
            s = Path(paths[i]).name
            plt.title(s[:min(len(s), 40)],
                      fontdict={'size': 8})  # limit to 40 characters
    fig.tight_layout()
    fig.savefig(fname, dpi=200)
    plt.close()


def topN(qf, gf, gl, n, img_paths):
    query = qf.view(-1, 1)  # query 是一张图
    score = torch.mm(gf, query)  # 计算得分[1, num]
    score = score.squeeze(1).cpu()
    score = score.numpy()
    #predict index
    index = np.argsort(score)
    index = index[::-1]  # index 倒过来
    # 得到前n个
    assert n > 0, "n 必须大于0"
    index = index[0:n]

    container = torch.zeros(n, 512, 512, 3)
    img_paths_list = []
    for cnt, i in enumerate(index):
        print("top%d\tid:%s\tname:%s" %
              (cnt, gl[i], os.path.basename(img_paths[i][0])))
        img_paths_list.append(img_paths[i][0])
        img = cv2.imread(img_paths[i][0])
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.tensor(img)
        img = img / 255.0
        print(img.shape, container.shape)
        container[cnt] = img

    print(container.shape)
    container.transpose(0,3,1,2)

    plot_images(container, img_paths_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('help')
    parser.add_argument('--weight_path', type=str, default="./weights/110.pth")
    parser.add_argument('--img_path',
                        type=str,
                        default="./Market/query/10/10_c1s1_18_01.jpg")
    parser.add_argument('--vis',
                        action="store_true",
                        help='whether show results')
    parser.add_argument('--num', type=int, default=15, help="top N")
    args = parser.parse_args()

    model = ResNet18(len(class_names))
    model.load(args.weight_path)

    model.eval()
    if use_gpu:
        model = model.cuda()

    query_features = extract_single_image(model, args.img_path)
    gallery_features = extract_features(model, gallery_dataloader)

    gallery_label = np.array(get_label(gallery_datasets.imgs))

    if use_gpu:
        gallery_features = gallery_features.cuda()
        query_features = query_features.cuda()

    topN(query_features, gallery_features, gallery_label, args.num,
         gallery_datasets.imgs)
