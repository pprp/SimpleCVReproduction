import json
import logging
import os
import re
import shutil

import numpy as np
import torch
import torch.nn as nn


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1-lam) * criterion(pred, y_b)


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


def generate_result(file, alpha1, alpha2, alpha3):
    with open('Track1_final_archs.json', 'r') as f:
        data = json.load(f)

    alpha = alpha1.detach().cpu().numpy().tolist() + \
        alpha2.detach().cpu().numpy().tolist() + alpha3.detach().cpu().numpy().tolist()
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


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
    """ inplace distillation for image classification """

    def forward(self, output, target):
        output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
        target = target.unsqueeze(1)
        output_log_prob = output_log_prob.unsqueeze(2)
        cross_entropy_loss = -torch.bmm(target, output_log_prob)
        return cross_entropy_loss.mean()


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


# arch_loader = ArchLoader("data/benchmark.json")
# for key , arch in arch_loader:
#     print(key, arch)
# print(arch_loader.get_arch_list())
# for i in range(20):
#     lst= arch_loader.generate_width_to_narrow(i, 20)
#     print(lst, sum(lst))
# print(arch_loader.generate_niu_fair_batch(random.randint(0,100))[-1].tolist())
# for i in range(10):
#     ta = arch_loader.generate_spos_like_batch()
#     print(type(ta),ta)

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


def mixup_accuracy(output, target_a, target_b, lam, topk=(1,)):
    batch_size = target_a.size(0)
    _, pred = torch.max(output.data, 1)

    correct = lam * pred.eq(target_a).cpu().sum().float() + \
        (1-lam) * pred.eq(target_b).cpu().sum().float()

    res = correct * 100.0 / batch_size

    return res


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

def load_checkpoint(path, model, optimizer=None):
    """path: weight path
       model: model object
    """
    if os.path.isfile(path):
        logging.info("=== loading checkpoint '{}' ===".format(path))

        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)

        if optimizer is not None:
            prec = checkpoint["prec"]
            last_epoch = checkpoint["last_epoch"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=== done. also loaded optimizer from "
                + "checkpoint '{}' (epoch {}) ===".format(path, last_epoch + 1)
            )
            return prec, last_epoch



def save_checkpoint(state, iters, exp_name, tag=''):
    '''
    state = state_dict, best_prec, last_epoch, optimizer.state_dict()
    '''
    if not os.path.exists("exp/{}/weights".format(exp_name)):
        os.makedirs("exp/{}/weights".format(exp_name))
    filename = os.path.join(
        "exp/{}/weights/{}model-{:05}.th".format(exp_name, tag, iters))

    torch.save(state, filename)
    latestfilename = os.path.join(
        "exp/{}/weights/{}model-latest.th".format(exp_name, tag))
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
    # with torch.no_grad():
    # print("Clear BN statistics...")
    # for m in model.modules():
    #     if isinstance(m, nn.BatchNorm2d):
    #         m.running_mean = torch.zeros_like(m.running_mean)
    #         m.running_var = torch.ones_like(m.running_var)
    model.apply(bn_calibration_init)

    # print("Train BN with training set (BN sanitize)...")
    model.train()

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs, cand)
        del inputs, targets, outputs
