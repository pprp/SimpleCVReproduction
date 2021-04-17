import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, label, topk=(1,)):
    maxk = max(topk)
    batch_size = label.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(label.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./snapshots"):
        os.makedirs("./snapshots")
    filename = os.path.join("./snapshots/{}_ckpt_{:04}.pth.tar".format(tag, iters))
    torch.save(state, filename)


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def data_transforms(args):
    if args.dataset == 'cifar10':
        MEAN = [0.49139968, 0.48215827, 0.44653124]
        STD = [0.24703233, 0.24348505, 0.26158768]
    elif args.dataset == 'imagenet':
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]

    if args.resize or args.dataset == 'imagenet':  # cifar10 resize or imagenet
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    else:  # cifar10
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])

    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    return train_transform, valid_transform


def random_choice(num_choice, layers):
    return list(np.random.randint(num_choice, size=layers))


def plot_hist(acc_list, min=0, max=101, interval=5, name='search'):
    plt.hist(acc_list, bins=max - min, range=(min, max), histtype='bar')
    plt.xticks(np.arange(min, max, interval))
    img_path = name + '.png'
    plt.savefig(img_path)
    plt.show()


def time_record(start):
    end = time.time()
    duration = end - start
    hour = duration // 3600
    minute = (duration - hour * 3600) // 60
    second = duration - hour * 3600 - minute * 60
    print('Elapsed time: hour: %d, minute: %d, second: %f' % (hour, minute, second))
