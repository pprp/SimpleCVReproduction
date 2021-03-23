import os
import re
import torch
import torch.nn as nn
import random
import json


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
        self.idx = 0

    def get_arch_list(self):
        return self.arc_list

    def get_arch_dict(self):
        return self.arc_dict

    def __next__(self):

        if self.idx > len(self.arc_list):
            raise StopIteration
        self.idx += 1
        return self.arc_list[self.idx]

    def __iter__(self):
        return self

    def get_arch_list_dict(self, path):
        with open(path, "r") as f:
            self.arc_dict = json.load(f)

        self.arc_list = []

        for _, v in self.arc_dict.items():
            self.arc_list.append(v["arch"])


# arch_loader = ArchLoader("Track1_final_archs.json", batch_size=10)

# for i in arch_loader:
#     print(i)


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
