import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from torch.autograd import Variable
import torch.optim as optim

# try to import tensorboard as logging tool
# see https://github.com/dmlc/tensorboard
use_tensorboard = False
try:
    import tensorboard
except ImportError:
    use_tensorboard = False

from warpctc_pytorch import CTCLoss

from model import StackedRNN
from dataset import CaptchaDataset
from utils import to_gpu, tensor_to_variable, get_prediction, DATASET_PATH

input_size, output_size = 180, 11
hidden_size = 512
number_layer = 2

if use_tensorboard:
    if not os.path.exists('./log'):
        os.mkdir('./log')
    logger = tensorboard.SummaryWriter('./log')

use_cuda = torch.cuda.is_available()

## get model
def get_model():
    return StackedRNN(input_size, output_size, hidden_size, number_layer)

model = get_model()

if use_cuda:
    model = model.cuda()

## get dataset
root_dir = DATASET_PATH

normalized = True
if normalized:
    from dataset import mean, std
else:
    mean = [0. for _ in range(3)]
    std = [1. for _ in range(3)]

def collate_fn(batch):
    imgs = torch.stack([x[0] for x in batch], dim=0)
    labels = [x[1] for x in batch]
    return imgs, labels

training_dataset = CaptchaDataset(os.path.join(root_dir, 'train'), 
    mean=mean, std=std)
testing_dataset = CaptchaDataset(os.path.join(root_dir, 'test'),
    mean=mean, std=std)

print('data has been loaded.')

traing_bsz = 64
testing_bsz = 64

num_workers = 4
training_loader = DataLoader(training_dataset, batch_size=traing_bsz, 
    shuffle=True, 
    collate_fn=collate_fn, pin_memory=True)

testing_loader = DataLoader(testing_dataset, batch_size=testing_bsz, 
    shuffle=False, 
    collate_fn=collate_fn, pin_memory=True)

def preprocess_data(x):
    """ Preprocess data

        :param x: `Tensor.FloatTensor` with size `N x C x H x W`
    """
    n, c, h, w = x.size()
    x = x.permute(3, 0, 2, 1).contiguous().view((w, n, -1))
    return x

def preprocess_target(target):
    """ Preprocess targets.

        :param target: list of `torch.IntTensor`
    """
    lengths = [len(t) for t in target]
    lengths = torch.IntTensor(lengths)
    
    flatten_target = torch.cat([t for t in target])
    return flatten_target, lengths

def get_seq_length(x):
    """ Get sequence lengths of batch of data
        :param x: batch data
    """
    bsz, length = x.size(1), x.size(0)
    lengths = torch.IntTensor(bsz).fill_(length)
    return lengths


class AverageMeter(object):
    """ Average meter.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset items.
        """
        self.n = 0
        self.val = 0.
        self.sum = 0.
        self.avg = 0.

    def update(self, val, n=1):
        """ Update
        """
        self.n += n
        self.val = val
        self.sum += val * n
        self.avg = self.sum / self.n

def get_accuracy(output, targets, prob=True):
    """ Get accuracy given output and targets
    """
    pred, _ = get_prediction(output, prob)
    cnt = 0
    for batch_ind, target in enumerate(targets):
        target = [v for v in target]
        if target == pred[batch_ind]:     
            cnt += 1
    return float(cnt) / len(targets)

criterion = CTCLoss()

solver = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train(epoch, max_epoch):
    """ train model
    """
    if epoch % 10 == 0:
        for param_group in solver.param_groups:
            param_group['lr'] *= 0.1

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for ind, (x, target) in enumerate(training_loader):
        # x is NxCxHxW => WxNx(HxC)
        x = preprocess_data(x)
        act_lengths = get_seq_length(x)
        # target is a list of `torch.InTensor` with `bsz` size.
        flatten_target, target_lengths = preprocess_target(target)

        if use_cuda:
            x = to_gpu(x)
        
        x, act_lengths, flatten_target, target_lengths = tensor_to_variable(
            (x, act_lengths, flatten_target, target_lengths), volatile=False)
        
        bsz = x.size(1)
        hidden = model.init_hidden(bsz)
        if use_cuda:
            hidden = to_gpu(hidden)

        output, _ = model(x, hidden)

        loss = criterion(output, flatten_target, act_lengths, target_lengths)

        solver.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(model.parameters(), 10)
        solver.step()

        if use_tensorboard:
            logger.add_scalar('train_loss', loss.data[0])

        loss_meter.update(loss.data[0])

        acc = get_accuracy(output, target)
        acc_meter.update(acc)
        if use_tensorboard:
            logger.add_scalar('train_acc', acc)

        if (ind+1) % 100 == 0 or (ind+1) == len(training_loader):
            print('train:\t[{:03d}/{:03d}],\t'
                  '[{:02d}/{:02d}]\t'
                  'loss: {loss.avg:.4f}({loss.val:.4f})\t'
                  'accuracy: {acc.avg:.4f}({acc.val:.4f})'.format(epoch, max_epoch,
                  ind+1, len(training_loader), loss=loss_meter, acc=acc_meter))

def test(epoch, max_epoch):
    """ Test model
    """
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for ind, (x, target) in enumerate(testing_loader):
        # x is NxCxHxW => WxNx(HxC)
        x = preprocess_data(x)
        act_lengths = get_seq_length(x)
        # target is a list of `torch.InTensor` with `bsz` size.
        flatten_target, target_lengths = preprocess_target(target)

        if use_cuda:
            x = to_gpu(x)
        
        x, act_lengths, flatten_target, target_lengths = tensor_to_variable(
            (x, act_lengths, flatten_target, target_lengths), volatile=True)
        
        bsz = x.size(1)
        hidden = model.init_hidden(bsz, volatile=True)
        if use_cuda:
            hidden = to_gpu(hidden)

        output, _ = model(x, hidden)

        acc = get_accuracy(output, target)
        acc_meter.update(acc)
        if use_tensorboard:
            logger.add_scalar('test_acc', acc)

    print('test:\t[{:03d}/{:03d}],\t'
          'accuracy: {acc.avg:.4f}({acc.val:.4f})'.format(epoch, max_epoch,
          acc=acc_meter))  
    return acc_meter.avg

def main():
    max_epoch = 30
    if not os.path.exists('pretrained'):
        os.mkdir('pretrained')
    for epoch in range(1, max_epoch+1):
        train(epoch, max_epoch)
        acc = test(epoch, max_epoch)
        if epoch % 10 == 0:
            torch.save({'state_dict': model.state_dict(),
                        'accuracy': acc}, 
                        os.path.join('pretrained', 
                        'model-{:02d}.pth.tar'.format(epoch)))


if __name__ == '__main__':
    main()
    


