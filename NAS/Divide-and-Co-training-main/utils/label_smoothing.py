# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F


class label_smoothing_CE(nn.Module):

    def __init__(self, e=0.1, reduction='mean'):
        super(label_smoothing_CE, self).__init__()

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.e = e
        self.reduction = reduction

    def forward(self, x, target):

        if x.size(0) != target.size(0):
            raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
                    .format(x.size(0), target.size(0)))

        if x.dim() < 2:
            raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
                    .format(x.size(0)))

        num_classes = x.size(-1)
        one_hot = F.one_hot(target, num_classes=num_classes).type_as(x)
        # simply adding because (e / num_classes) is very small in most cases
        smoothed_target = one_hot * (1 - self.e) + self.e / num_classes
        # negative log likelihood
        log_probs = self.log_softmax(x)
        loss = torch.sum(- log_probs * smoothed_target, dim=-1)

        if self.reduction == 'none':
            return loss

        elif self.reduction == 'sum':
            return torch.sum(loss)
        
        elif self.reduction == 'mean':
            return torch.mean(loss)
        
        else:
            raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')
