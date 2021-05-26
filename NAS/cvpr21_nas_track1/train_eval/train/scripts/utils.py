import os
import torch
import shutil
import math
import numpy as np
import json
import random
from resnet20_supernet import SuperNetSetting


def generate_result(file, alpha1, alpha2, alpha3):
    with open('Track1_final_archs.json', 'r') as f:
        data = json.load(f)

    alpha = alpha1.detach().cpu().numpy().tolist() + alpha2.detach().cpu().numpy().tolist() + alpha3.detach().cpu().numpy().tolist()
    archindex_logprob_list = []
    for i in range(1, 50001):
        archindex = 'arch{}'.format(i)
        logprob = 0
        arch = data[archindex]['arch']
        for j, c in enumerate(arch.split('-')[:-1]):
            index = SuperNetSetting[j].index(int(c))
            logprob += math.log(alpha[j][index])
        archindex_logprob_list.append((archindex, logprob))
    archindex_logprob_list = sorted(archindex_logprob_list, key=lambda x: x[1])

    for i in range(1, 50001):
        key = archindex_logprob_list[i - 1][0]
        data[key]['acc'] = i / 50000

    with open(file, 'w') as json_file:
        json.dump(data, json_file)


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        if not os.path.exists(os.path.join(path, 'scripts')):
            os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


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

        mask[y1:y2, x1:x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
