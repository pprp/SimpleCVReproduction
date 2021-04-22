""" This file stores some common functions for learners """

import torch
import torch.nn as nn
import pdb
import math
import numpy as np
import logging
import os
import torch.distributed as dist
import torch.nn.functional as F
from collections import defaultdict


class CrossEntropyLabelSmooth(nn.Module):
  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
  """ Code taken from torchvision """
  def __init__(self, num_batches, *meters, prefix=""):
    self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
    self.meters = meters
    self.prefix = prefix

  def show(self, batch):
    entries = [self.prefix + self.batch_fmtstr.format(batch)]
    entries += [str(meter) for meter in self.meters]
    logging.info(' '.join(entries))
    # print(' '.join(entries))

  def _get_batch_fmtstr(self, num_batches):
    num_digits = len(str(num_batches // 1))
    fmt = '{:' + str(num_digits) + 'd}'
    return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
  """Computes the accuracy over the k top predictions for the specified values of k
     Code taken from torchvision.
  """
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_seen_unseen(output, target, seen_classes, topk=(1.)):
  """ Compute the acc of seen classes and unseen classes seperately."""
  seen_classes = list(range(seen_classes))
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    seen_ind = torch.zeros(target.shape[0]).byte().cuda()
    unseen_ind = torch.ones(target.shape[0]).byte().cuda()
    for v in seen_classes:
      seen_ind += (target == v)
    unseen_ind -= seen_ind

    seen_correct = pred[:, seen_ind].eq(target[seen_ind].view(1, -1).expand_as(pred[:, seen_ind]))
    unseen_correct = pred[:, unseen_ind].eq(target[unseen_ind].view(1, -1).expand_as(pred[:, unseen_ind]))

    seen_num = seen_correct.shape[1]
    res = []
    seen_accs = []
    unseen_accs = []
    for k in topk:
        seen_correct_k = seen_correct[:k].view(-1).float().sum(0, keepdim=True)
        seen_accs.append(seen_correct_k.mul_(100.0 / (seen_num+1e-10)))
        unseen_correct_k = unseen_correct[:k].view(-1).float().sum(0, keepdim=True)
        unseen_accs.append(unseen_correct_k.mul_(100.0 / (batch_size-seen_num+1e-10)))
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res, seen_accs, unseen_accs, seen_num

def is_last_layer(layer):
  W = layer.weight
  if W.ndimension() == 2 and (W.shape[0] == 10 or W.shape[0] == 100 or W.shape[0] == 1000):
    return True
  else:
    return False


def is_first_layer(layer):
  if isinstance(layer, nn.Conv2d):
    W = layer.weight
    if W.ndimension() == 4 and (W.shape[1] == 3 or W.shape[1] == 1):
      return True
    else:
      return False
  else:
    return False


def reinitialize_conv_weights(model, init_first=False):
  """ Only re-initialize the kernels for conv layers """
  print("re-intializing conv weights done. evaluating...")
  for W in model.parameters():
    if W.ndimension() == 4:
      if W.shape[1] == 3 and not init_first:
        continue # do not init first conv layer
      nn.init.kaiming_uniform_(W, a = math.sqrt(5))


def weights_init(m):
  """ default param initializer in pytorch. """
  if isinstance(m, nn.Conv2d):
    n = m.in_channels
    for k in m.kernel_size:
      n *= k
    stdv = 1. / math.sqrt(n)
    m.weight.data.uniform_(-stdv, stdv)
    if m.bias is not None:
        m.bias.data.uniform_(-stdv, stdv)
  elif isinstance(m, nn.Linear):
    stdv = 1. / math.sqrt(m.weight.size(1))
    m.weight.data.uniform_(-stdv, stdv)
    if m.bias is not None:
        m.bias.data.uniform_(-stdv, stdv)


class keydefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        else:
            ret = self[key] = self.default_factory(key)
            return ret


class WarmUpCosineLRScheduler:
    """
    update lr every step
    """
    def __init__(self, optimizer, T_max, eta_min, base_lr, warmup_lr, warmup_steps, last_iter=-1):
        # Attach optimizer
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer

        # Initialize step and base learning rates
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_iter = last_iter

        assert warmup_steps < T_max
        self.T_max = T_max
        self.eta_min = eta_min

        # warmup settings
        self.base_lr = base_lr
        self.warmup_lr = warmup_lr
        self.warmup_step = warmup_steps
        self.warmup_k = None

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        return list(map(lambda group: group['lr'], self.optimizer.param_groups))

    def step(self, this_iter=None):
        if this_iter is None:
            this_iter = self.last_iter + 1
        self.last_iter = this_iter

        # get lr during warmup stage
        if self.warmup_step > 0 and this_iter < self.warmup_step:
            if self.warmup_k is None:
                self.warmup_k = (self.warmup_lr - self.base_lr) / self.warmup_step
            scale = (self.warmup_k * this_iter + self.base_lr) / self.base_lr
        # get lr during cosine annealing
        else:
            step_ratio = (this_iter - self.warmup_step) / (self.T_max - self.warmup_step)
            scale = self.eta_min + (self.warmup_lr - self.eta_min) * (1 + math.cos(math.pi * step_ratio)) / 2
            scale /= self.base_lr

        values = [scale * lr for lr in self.base_lrs]
        for param_group, lr in zip(self.optimizer.param_groups, values):
            param_group['lr'] = lr
