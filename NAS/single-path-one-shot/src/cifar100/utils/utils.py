import os
import re
import torch
import torch.nn as nn
import random
import json
import numpy as np
from tqdm import tqdm

class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """
    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss
        
def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()


def reduce_tensor(tensor, device=0, world_size=1):
    tensor = tensor.clone()
    torch.distributed.reduce(tensor, device)
    tensor.div_(world_size)
    return tensor


def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= nprocs
    return rt


class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]


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

    def get_arch_list_dict(self, path):
        with open(path, "r") as f:
            self.arc_dict = json.load(f)

        self.arc_list = []

        for _, v in self.arc_dict.items():
            self.arc_list.append(v["arch"])

    def convert_list_arc_str(self, arc_list):
        arc_str = ""
        arc_list = [str(item)+"-" for item in arc_list]
        for item in arc_list:
            arc_str += item
        return arc_str[:-1]

    def convert_str_arc_list(self, arc_str):
        return [int(i) for i in arc_str.split('-')]

    def generate_spos_like_batch(self):
        rngs = []
        for i in range(0, 7):
            rngs += np.random.choice(
                self.level_config["level1"], size=1).tolist()
        for i in range(7, 13):
            rngs += np.random.choice(
                self.level_config['level2'], size=1).tolist()
        for i in range(13, 20):
            rngs += np.random.choice(
                self.level_config['level3'], size=1).tolist()
        return np.array(rngs)

    def generate_niu_fair_batch(self, seed):
        rngs = []
        seed = seed
        random.seed(seed)
        # level1
        for i in range(0, 7):
            tmp_rngs = []
            for _ in range(4):
                tmp_rngs.extend(random.sample(self.level_config['level1'],
                                              len(self.level_config['level1'])))
            rngs.append(tmp_rngs)
        # level2
        for i in range(7, 13):
            tmp_rngs = []
            for _ in range(2):
                tmp_rngs.extend(random.sample(self.level_config['level2'],
                                              len(self.level_config['level2'])))
            rngs.append(tmp_rngs)

        # level3
        for i in range(13, 20):
            rngs.append(random.sample(self.level_config['level3'],
                                      len(self.level_config['level3'])))
        return np.transpose(rngs)

    def generate_width_to_narrow(self, current_epoch, total_epoch):
        current_p = float(current_epoch) / total_epoch
        opposite_p = 1 - current_p
        # print(current_p, opposite_p)
        def p_generator(length):
            rng_list = np.linspace(current_p, opposite_p, length)
            return self.softmax(rng_list)

        rngs = []
        for i in range(0, 7):
            rngs += np.random.choice(self.level_config["level1"], size=1, p=p_generator(
                len(self.level_config['level1']))).tolist()
        for i in range(7, 13):
            rngs += np.random.choice(self.level_config['level2'], size=1, p=p_generator(
                len(self.level_config['level2']))).tolist()
        for i in range(13, 20):
            rngs += np.random.choice(self.level_config['level3'], size=1, p=p_generator(
                len(self.level_config['level3']))).tolist()
        return np.array(rngs)

    @staticmethod
    def softmax(rng_lst):
        if isinstance(rng_lst, list):
            x = np.asarray(rng_lst)
        else:
            x = rng_lst
        x_max = x.max()
        x = x - x_max
        x_exp = np.exp(x)
        x_exp_sum = x_exp.sum(axis=0)
        softmax = x_exp / x_exp_sum
        return softmax


# arch_loader = ArchLoader("data/Track1_final_archs.json")

# for i in range(20):
#     lst= arch_loader.generate_width_to_narrow(i, 20)
#     print(lst, sum(lst))

# print(arch_loader.generate_niu_fair_batch(random.randint(0,100))[-1].tolist())
# for i in range(10):
#     ta = arch_loader.generate_spos_like_batch()
#     print(type(ta),ta)
#     tb = arch_loader.generate_niu_fair_batch(i)[-1]
#     print(type(tb),tb)


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


def save_checkpoint(state, iters, exp, tag=''):
    if not os.path.exists("./weights/{}".format(exp)):
        os.makedirs("./weights/{}".format(exp))
    filename = os.path.join(
        "./weights/{}/{}checkpoint-{:05}.pth.tar".format(exp, tag, iters))

    torch.save(state, filename)
    latestfilename = os.path.join(
        "./weights/{}/{}checkpoint-latest.pth.tar".format(exp, tag))
    torch.save(state, latestfilename)


def get_lastest_model():
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    model_list = os.listdir('./weights/')
    if model_list == []:
        return None, 0
    model_list.sort()
    lastest_model = model_list[-2]
    iters = re.findall(r'\d+', lastest_model)
    return './weights/' + lastest_model, int(iters[0])


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

    cumulative_bn_stats = True

    if getattr(m, 'track_running_stats', False):
        # reset all values for post-statistics
        m.reset_running_stats()
        # set bn in training mode to update post-statistics
        m.training = True
        # if use cumulative moving average
        # if getattr(FLAGS, 'cumulative_bn_stats', False):
        if cumulative_bn_stats:
            m.momentum = None


def retrain_bn(model, dataloader, cand, device=0):
    # from singlepathoneshot Search/tester.py
    with torch.no_grad():
        # print("Clear BN statistics...")
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.running_mean = torch.zeros_like(m.running_mean)
                m.running_var = torch.ones_like(m.running_var)

        # print("Train BN with training set (BN sanitize)...")
        model.train()

        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs, cand[:-1])
            del inputs, targets, outputs
