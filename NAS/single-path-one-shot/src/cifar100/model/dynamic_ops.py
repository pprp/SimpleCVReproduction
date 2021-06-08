import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

def get_same_padding(kernel_size):
    if isinstance(kernel_size, tuple):
        assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
        p1 = get_same_padding(kernel_size[0])
        p2 = get_same_padding(kernel_size[1])
        return p1, p2
    assert isinstance(
        kernel_size, int), 'kernel size should be either `int` or `tuple`'
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


class BaseBN(nn.Module):
    def __init__(self, indims, outdims):
        super.__init__()
        self.indims = indims
        self.outdims = outdims
        self.max = max(outdims)
    
    def forward(self, x, indim, outdim):
        raise NotImplementedError()


class IndependentBN(BaseBN):
    """SBN in MixPath"""
    def __init__(self, indims, outdims):
        super().__init__(indims, outdims)
        for cin in indims:
            for cout in outdims:
                setattr(self, f"bn-{cin}-{cout}", nn.BatchNorm2d(cout))

    def forward(self, x, indim, outdim):
        x = x[:,:outdim]
        return getattr(self, f"bn-{indim}-{outdim}")(x)

class FrontShareBN()
        
