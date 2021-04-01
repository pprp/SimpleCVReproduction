import os
import re
import torch
import torch.nn as nn
import random
import json
import numpy as np


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


class ArchLoader():
    '''
    load arch from json file
    '''

    def __init__(self, path):
        super(ArchLoader, self).__init__()

        self.arc_list = []
        self.arc_dict = {}
        self.get_arch_list_dict(path)
        random.shuffle(self.arc_list)
        self.idx = -1

        self.level_config = {
            "level1": [4, 8, 12, 16],
            "level2": [4, 8, 12, 16, 20, 24, 28, 32],
            "level3": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64]
        }

    def get_arch_list(self):
        return self.arc_list

    def get_arch_dict(self):
        return self.arc_dict

    def get_random_batch(self, bs):
        return random.sample(self.arc_list, bs)

    def get_part_dict(self):
        keys = list(self.arc_dict.keys())[:10]
        return dict([(key, self.arc_dict[key]) for key in keys])

    def convert_list_arc_str(self, arc_list):
        arc_str = ""
        arc_list = [str(item)+"-" for item in arc_list]
        for item in arc_list:
            arc_str += item

        return arc_str[:-1]

    def __next__(self):
        self.idx += 1
        if self.idx >= len(self.arc_list):
            raise StopIteration
        return self.arc_list[self.idx]

    def __iter__(self):
        return self

    def get_arch_list_dict(self, path):
        with open(path, "r") as f:
            self.arc_dict = json.load(f)

        self.arc_list = []

        for _, v in self.arc_dict.items():
            self.arc_list.append(v["arch"])

    def generate_fair_batch(self):
        rngs = []
        seed = 0
        # level1
        for i in range(0, 7):
            seed += 1
            random.seed(seed)
            rngs.append(random.sample(self.level_config['level1'],
                                      len(self.level_config['level1']))*4)
        # level2
        for i in range(7, 13):
            seed += 1
            random.seed(seed)
            rngs.append(random.sample(self.level_config['level2'],
                                      len(self.level_config['level2']))*2)

        # level3
        for i in range(13, 20):
            seed += 1
            random.seed(seed)
            rngs.append(random.sample(self.level_config['level3'],
                                      len(self.level_config['level3'])))
        return np.transpose(rngs)

# arch_loader = ArchLoader("Track1_final_archs.json")


# print(arch_loader.generate_fair_batch())
# arc_dc = arch_loader.get_random_batch(1000)

# for i, arc in enumerate(arc_dc):
#     print(i, arc)


# cnt = 0
# for i,ac in enumerate(arch_loader):
#     print(i,ac)
#     cnt += 1

# print(cnt)

class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * \
            targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, iters, tag=''):
    if not os.path.exists("./models"):
        os.makedirs("./models")
    filename = os.path.join(
        "./models/{}checkpoint-{:06}.pth.tar".format(tag, iters))
    torch.save(state, filename)
    # latestfilename = os.path.join(
    #     "./models/{}checkpoint-latest.pth.tar".format(tag))
    # torch.save(state, latestfilename)


def get_lastest_model():
    if not os.path.exists('./models'):
        os.mkdir('./models')
    model_list = os.listdir('./models/')
    if model_list == []:
        return None, 0
    model_list.sort()
    lastest_model = model_list[-1]
    iters = re.findall(r'\d+', lastest_model)
    return './models/' + lastest_model, int(iters[0])


def get_parameters(model):
    group_no_weight_decay = []
    group_weight_decay = []
    for pname, p in model.named_parameters():
        if pname.find('weight') >= 0 and len(p.size()) > 1:
            # print('include ', pname, p.size())
            group_weight_decay.append(p)
        else:
            # print('not include ', pname, p.size())
            group_no_weight_decay.append(p)
    assert len(list(model.parameters())) == len(
        group_weight_decay) + len(group_no_weight_decay)
    groups = [dict(params=group_weight_decay), dict(
        params=group_no_weight_decay, weight_decay=0.)]
    return groups


def bn_calibration_init(m):
    """ calculating post-statistics of batch normalization """
    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        if getattr(FLAGS, 'cumulative_bn_stats', False):
            m.momentum = None
