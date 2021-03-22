import os
import shutil
import numpy as np
import torch
import torch.nn as nn
import math
import joblib
from torch.autograd import Variable
from collections import defaultdict
import torch.distributed as dist
import copy

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

def broadcast(obj, src, group=torch.distributed.group.WORLD, async_op=False):
    obj_tensor = torch.from_numpy(np.array(obj)).cuda()
    torch.distributed.broadcast(obj_tensor, src, group, async_op)
    obj = obj_tensor.cpu().numpy()
    return obj

# Average loss across processes for logging
def reduce_tensor(tensor, device=0, world_size=1):
    tensor = tensor.clone()
    dist.reduce(tensor, device)
    tensor.div_(world_size)
    return tensor

def accuracy(output, target, topk=(1,)):
  maxk = max(topk)
  batch_size = target.size(0)

  _, pred = output.topk(maxk, 1, True, True)
  pred = pred.t()
  correct = pred.eq(target.view(1, -1).expand_as(pred))

  res = []
  for k in topk:
    correct_k = correct[:k].view(-1).float().sum(0)
    res.append(correct_k.mul_(100.0/batch_size))
  return res

def save_checkpoint(state, save):
  torch.save(state, save)
  print('save checkpoint....')

def save_checkpoint_(state, save):
  if not os.path.exists(save):
    os.makedirs(save)
  filename = os.path.join(save, 'checkpoint.pth.tar')
  torch.save(state, filename)
  print('save checkpoint....')

def save(model, model_path):
  torch.save(model.state_dict(), model_path)
  print('Save SuperNet....')

def load(model, model_path):
  model.load_state_dict(torch.load(model_path))
  print('Load SuperNet....')

def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  script_path = os.path.join(path, 'scripts')
  if scripts_to_save is not None and not os.path.exists(script_path):
    os.mkdir(script_path)
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

def get_topk_str(rngs):
    cand = ''
    for r in rngs[0]:
      cand += str(r)
      cand += ' '
    cand = cand[:-1]
    return cand

# shrinking
def get_location(s, key):
    d = defaultdict(list)
    for k,va in [(v,i) for i,v in enumerate(s)]:
        d[k].append(va)
    return d[key]

def list_substract(list1, list2):
    list1 = [item for item in list1 if item not in set(list2)]
    return list1

# Following DARTS, make sure each node connects only two predecessor nodes
def check_cand(cand, operations, n_edges):
    cand = np.reshape(cand, [-1, n_edges])
    offset, cell_cand = 0, cand[0]
    for j in range(4):
        edges = cell_cand[offset:offset+j+2]
        edges_ops = operations[offset:offset+j+2]
        none_idxs = get_location(edges, 0)
        if len(none_idxs) < j:
            general_idxs = list_substract(range(j+2), none_idxs)
            num = min(j-len(none_idxs), len(general_idxs))
            general_idxs = np.random.choice(general_idxs, size=num, replace=False, p=None)
            for k in general_idxs:
                edges[k] = 0
        elif len(none_idxs) > j:
            none_idxs = np.random.choice(none_idxs, size=len(none_idxs)-j, replace=False, p=None)
            for k in none_idxs:
                if len(edges_ops[k]) > 1:
                    l = np.random.randint(len(edges_ops[k])-1)
                    edges[k] = edges_ops[k][l+1]
        offset += len(edges)
    for i in range(1,len(cand)):
        cand[i] = copy.deepcopy(cell_cand)
    return cand.tolist()

def merge_ops(rngs):
    cand = []
    for rng in rngs:
        for r in rng:
          if isinstance(r, list):
            cand += r
            cand += [-3]
          else:
            cand.append(r)
        cand += [-2]
    cand = cand[:-1]
    return cand

def split_ops_(cand_):
  if -3 in cand_:
    idxs_ = get_location(cand_, -3)
    idxs_ = idxs_[:-1]
    tmp = []
    last_idx_ = 0
    for idx_ in idxs_:
      tmp.append(cand_[last_idx_:idx_])
      last_idx_ = idx_ + 1
    tmp.append(cand_[last_idx_:-1])
    cand_ = tmp
  return cand_

def split_ops(cand):
    cand = list(cand)
    ops = []
    idxs = get_location(cand, -2)
    last_idx = 0
    for idx in idxs:
        cand_ = cand[last_idx:idx]
        cand_ = split_ops_(cand_)
        ops.append(cand_)
        last_idx = idx+1
    cand_ = cand[last_idx:]
    cand_ = split_ops_(cand_)
    ops.append(cand_)
    return ops

def get_search_space_size(operations):
  comb_num = 1
  for j in range(len(operations)):
      comb_num *= len(operations[j])
  return comb_num

# Get flops by lookup table
def get_arch_flops(op_flops_dict, cand, backbone_info, blocks_keys):
    assert len(cand) == len(backbone_info) - 2
    preprocessing_flops = op_flops_dict['PreProcessing'][backbone_info[0]]
    postprocessing_flops = op_flops_dict['PostProcessing'][backbone_info[-1]]
    total_flops = preprocessing_flops + postprocessing_flops
    for i in range(len(cand)):
        inp, oup, img_h, img_w, stride = backbone_info[i+1]
        op_id = cand[i]
        if op_id >= 0:
          key = blocks_keys[op_id]
          total_flops += op_flops_dict[key][(inp, oup, img_h, img_w, stride)]
    return total_flops

# cifar
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

def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
  CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def _data_transforms_cifar100(args):
  CIFAR_MEAN = [0.5071, 0.4867, 0.4408]
  CIFAR_STD = [0.2675, 0.2565, 0.2761]

  train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
    x.div_(keep_prob)
    x.mul_(mask)
  return x
  
def get_optimizer_schedule(model, args, total_iters=None):

    all_parameters = model.parameters()
    weight_parameters = []
    for pname, p in model.named_parameters():
      if p.ndimension() == 4 or 'classifier.0.weight' in pname or 'classifier.0.bias' in pname:
          weight_parameters.append(p)
    weight_parameters_id = list(map(id, weight_parameters))
    other_parameters = list(filter(lambda p: id(p) not in weight_parameters_id, all_parameters))
    optimizer = torch.optim.SGD(
        [{'params' : other_parameters},
        {'params' : weight_parameters, 'weight_decay' : args.weight_decay}],
        args.learning_rate,
        momentum=args.momentum,
        )

    if total_iters is None:
      total_iters = args.total_iters  
    print('total_iters={}'.format(total_iters))
    delta_iters = total_iters / (1.-args.min_lr / args.learning_rate)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step : (1.0-step/delta_iters), last_epoch=-1)
    return optimizer, scheduler

def recalculate_bn(net, rngs, train_dataprovider, data_arr=None, data_for_bn='./train_20000_images_for_BN.pkl', batchsize=64):
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean = torch.zeros_like(m.running_mean)
            m.running_var = torch.ones_like(m.running_var)

    if not os.path.exists(data_for_bn):
      img_num = 0
      for step in range(1000):
          image, _ = train_dataprovider.next()
          if data_arr is None:
              data_arr = image
          else:
              data_arr = np.concatenate((data_arr, image), 0)
          img_num = data_arr.shape[0]
          if img_num > 20000:
            break
      data_arr = data_arr[:20000, :, :, :]

      f = open(data_for_bn, 'wb')
      joblib.dump(data_arr, f)
      f.close()
    if data_arr is None:
        data_arr = joblib.load(open(data_for_bn, 'rb'))

    print('Compute BN, rng={}'.format(rngs))
    net.train()
    with torch.no_grad():
        for i in range(0, data_arr.shape[0], batchsize):
            data = data_arr[i:i+batchsize]
            data = torch.from_numpy(data).cuda()
            raw_logits = net(data, rngs)
            del raw_logits, data