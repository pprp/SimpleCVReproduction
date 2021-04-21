import torch
import torch.nn as nn
import torch.optim as optim
import logging
import os
import torch.distributed as dist
from tensorboardX import SummaryWriter
from utils.utils import *
from utils.dist_utils import *
from timeit import default_timer as timer
import time
import numpy as np
import copy
from pdb import set_trace as br


class TrickLearner(object):
  """ Performs vanilla training with tricks:
    1. Label Smooth;
    2. Mixup;
    3. SE-module is deployed in nets/decode models.
  """
  def __init__(self, model, loaders, args, device):
    self.args = args
    self.device = device
    self.model = model
    self.__build_path()
    self.train_loader, self.test_loader = loaders
    self.setup_optim()
    self.criterion = nn.CrossEntropyLoss().cuda()
    if self.args.label_smooth:
      classes = 10 if args.dataset == 'cifar10' else (1000 if args.dataset == 'ilsvrc_12' else 100)
      self.criterion_smooth = CrossEntropyLabelSmooth(classes, args.label_smooth_eps).cuda()

    if self.check_is_primary():
      self.writer = SummaryWriter(os.path.dirname(self.save_path))
      # self.add_graph()

  def train(self, train_sampler=None):
    for epoch in range(self.args.epochs):
      if self.args.distributed:
        assert train_sampler != None
        train_sampler.set_epoch(epoch)

      self.model.train()
      if self.check_is_primary():
        logging.info("Training at Epoch: %d" % epoch)
      train_acc, train_loss = self.epoch(True)

      if self.check_is_primary():
        self.writer.add_scalar('train_acc', train_acc, epoch)
        self.writer.add_scalar('train_loss', train_loss, epoch)

      if (epoch+1) % self.args.eval_epoch == 0:
        # evaluate every GPU, but we only show the results on a single one.!
        if self.check_is_primary():
          logging.info("Evaluation at Epoch: %d" % epoch)
        self.evaluate(True, epoch)

        if self.check_is_primary():
          self.save_model()

  def evaluate(self, is_train=False, epoch=None):
    self.model.eval()
    # NOTE: syncronizing the BN statistics
    if self.args.distributed:
      sync_bn_stat(self.model, self.args.world_size)

    if not is_train:
      self.load_model()

    with torch.no_grad():
      test_acc, test_loss = self.epoch(False)

    if is_train and epoch and self.check_is_primary():
      self.writer.add_scalar('test_acc', test_acc, epoch)
      self.writer.add_scalar('test_loss', test_loss, epoch)
    return test_acc, test_loss

  def finetune(self, train_sampler):
    self.load_model()
    self.evaluate()

    for epoch in range(self.args.epochs):
      if self.args.distributed:
        assert train_sampler != None
        train_sampler.set_epoch(epoch)

      self.model.train()

      # NOTE: use the preset learning rate for all epochs.
      ft_acc, ft_loss = self.epoch(True)

      if self.check_is_primary():
        self.writer.add_scalar('ft_acc', ft_acc, epoch)
        self.writer.add_scalar('ft_loss', ft_loss, epoch)

      # evaluate every k step
      if (epoch+1) % self.args.eval_epoch == 0:
        if self.check_is_primary():
          logging.info("Evaluation at Epoch: %d" % epoch)
        self.evaluate(True, epoch)

        # save the model
      if self.check_is_primary():
        self.save_model()

  def misc(self):
    raise NotImplementedError("Misc functions are implemented in sub classes")

  def epoch(self, is_train):
    """ Rewrite this function if necessary in the sub-classes. """

    loader = self.train_loader if is_train else self.test_loader

    # setup statistics
    batch_time = AverageMeter('Time', ':3.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    lrs = AverageMeter('Lr', ':.4e')
    top1 = AverageMeter('Acc@1', ':3.3f')
    top5 = AverageMeter('Acc@5', ':3.3f')
    metrics = [batch_time, lrs, top1, top5, losses]

    loader_len = len(loader)
    progress = ProgressMeter(loader_len, *metrics, prefix='Job id: %s, ' % self.args.job_id)
    end = time.time()

    for idx, (X, y) in enumerate(loader):

      # data_time.update(time.time() - end)
      criterion = self.criterion_smooth if is_train and self.args.label_smooth else self.criterion
      X, y = X.to(self.device), y.to(self.device)

      if is_train and self.args.mixup:
        mixed_X, y_a, y_b, lam = self.mixup_data(X, y)
        mixed_yp = self.model(mixed_X)
        loss = (lam * criterion(mixed_yp, y_a) + (1.0 - lam) * criterion(mixed_yp, y_b)) / self.args.world_size
        acc1_a, acc5_a = accuracy(mixed_yp, y_a, topk=(1, 5))
        acc1_b, acc5_b = accuracy(mixed_yp, y_b, topk=(1, 5))
        acc1, acc5 = lam * acc1_a + (1 - lam) * acc1_b, lam * acc5_a + (1 - lam) * acc5_b
      else:
        yp = self.model(X)
        loss = criterion(yp, y) / self.args.world_size
        acc1, acc5 = accuracy(yp, y, topk=(1, 5))

      reduced_loss = loss.data.clone()
      reduced_acc1 = acc1.clone() / self.args.world_size
      reduced_acc5 = acc5.clone() / self.args.world_size

      if self.args.distributed:
        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_acc1)
        dist.all_reduce(reduced_acc5)

      if is_train:
        self.opt.zero_grad()
        loss.backward()
        if self.args.distributed:
          average_gradients(self.model) # NOTE: important
        self.opt.step()

      if self.lr_scheduler:
        self.lr_scheduler.step()

      # update statistics
      top1.update(reduced_acc1[0].item(), X.shape[0])
      top5.update(reduced_acc5[0].item(), X.shape[0])
      losses.update(reduced_loss.item(), X.shape[0])
      lrs.update(self.lr_scheduler.get_lr()[0])
      batch_time.update(time.time() - end)
      end = time.time()

      # show the training/evaluating statistics
      if self.check_is_primary() and ((idx % self.args.print_freq == 0) or (idx+1) % loader_len == 0):
        progress.show(idx)

    return top1.avg, losses.avg

  def setup_optim(self):
    max_iter = len(self.train_loader) * self.args.epochs

    if self.args.model_type.startswith('model_'):
      self.opt = optim.SGD(self.model.parameters(), lr=self.args.lr, \
          momentum=self.args.momentum, nesterov=self.args.nesterov, \
          weight_decay=self.args.weight_decay)
      self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[int(max_iter*0.5), int(max_iter*0.75)])

    elif self.args.model_type.startswith('resnet_'):
      self.opt = optim.SGD(self.model.parameters(), lr=self.args.lr, \
          momentum=self.args.momentum, nesterov=self.args.nesterov, \
          weight_decay=self.args.weight_decay)
      if self.args.lr_decy_type == 'cosine':
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, max_iter, eta_min=0)
      elif self.args.lr_decy_type == 'multi_step':
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.opt, milestones=[int(max_iter*0.5), int(max_iter*0.75)])
      else:
        raise ValueError("Unknown learning rate decay type")

    elif self.args.model_type.startswith('mobilenet_v2'):
      # default on 8-gpu: 250 epochs, 2e-1 lr with cosine, 4e-5 wd, no dropout, warmup to 8e-1, nestrov, no wd for BN and bias
      param_groups = self.get_param_group()
      self.opt = optim.SGD(param_groups, lr=self.args.lr, \
          momentum=self.args.momentum, nesterov=self.args.nesterov, \
          weight_decay=self.args.weight_decay)
      # self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.epochs, eta_min=0)
      warmup_lr = 4 * self.args.lr
      warmup_steps = 1250
      self.lr_scheduler = WarmUpCosineLRScheduler(self.opt, max_iter, self.args.lr_min, self.args.lr, warmup_lr, warmup_steps, last_iter=-1)

    else:
      raise ValueError("Unknown model, failed to initalize optim")

  def add_graph(self):
    # create dummy input
    x = torch.randn(self.args.batch_size, 3, 32, 32)
    with self.writer:
      self.writer.add_graph(self.model, (x,))

  def __build_path(self):
    if self.args.exec_mode == 'finetune':
      self.load_path = self.args.load_path
      self.save_path = os.path.join(os.path.dirname(self.load_path), 'model_ft.pt')
    elif self.args.exec_mode == 'train':
      self.save_path = os.path.join(self.args.save_path, '_'.join([self.args.model_type, self.args.learner]), self.args.job_id, 'model.pt')
      self.load_path = self.save_path
    else:
      self.load_path = self.args.load_path
      self.save_path = self.load_path

  def check_is_primary(self):
    if (self.args.distributed and self.args.rank == 0) or \
        not self.args.distributed:
      return True
    else:
      return False

  def save_model(self):
    state = {'state_dict': self.model.state_dict(), \
        'optimizer': self.opt.state_dict()}
    torch.save(state, self.save_path)
    logging.info("Model stored at: " + self.save_path)

  def load_model(self):
    if self.args.distributed:
      # read parameters to each GPU seperately
      loc = 'cuda:{}'.format(torch.cuda.current_device())
      checkpoint = torch.load(self.load_path, map_location=loc)
    else:
      checkpoint = torch.load(self.load_path)

    self.model.load_state_dict(checkpoint['state_dict'])
    self.opt.load_state_dict(checkpoint['optimizer'])
    logging.info("Model succesfully restored from %s" % self.load_path)

    if self.args.distributed:
      broadcast_params(self.model)

  def mixup_data(self, X, y):
    batch_size = X.size()[0]
    alpha = self.args.mixup_alpha
    if alpha > 0:
      lam = np.random.beta(alpha, alpha)
    else:
      lam = 1
    index = torch.randperm(batch_size).to(self.device)
    mixed_X = lam * X + (1 - lam) * X[index, :]
    y_a, y_b =  y, y[index]
    return mixed_X, y_a, y_b, lam

  def get_param_group(self):
    param_group_no_wd = []
    names_no_wd = []
    param_group_normal = []

    for name, m in self.model.named_modules():
        if isinstance(m, nn.Conv2d):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
        elif isinstance(m, nn.Linear):
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if m.weight is not None:
                param_group_no_wd.append(m.weight)
                names_no_wd.append(name + '.weight')
            if m.bias is not None:
                param_group_no_wd.append(m.bias)
                names_no_wd.append(name + '.bias')

    for name, p in self.model.named_parameters():
        if name not in names_no_wd:
            param_group_normal.append(p)
    return [{'params': param_group_normal}, {'params': param_group_no_wd, 'weight_decay': 0.0}]