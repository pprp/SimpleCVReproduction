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


def pad(tensor, max_dim):
    """padding with zeros"""
    c = tensor.shape[1]  # channels
    if c != max_dim:
        # padd
        tmp_shape = tensor.shape
        tmp_shape[1] = max_dim - c
        pad_zero = torch.zeros(tmp_shape)
        return torch.cat([tensor, pad_zero], dim=1)
    return tensor


# BN PART

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
        x = x[:, :outdim]
        return getattr(self, f"bn-{indim}-{outdim}")(x)


class FrontShareBN(BaseBN):
    def __init__(self, indims, outdims):
        super().__init__(indims, outdims)
        for cin in indims:
            setattr(self, f"bn-{cin}", nn.BatchNorm2d(self.max))

    def forward(self, x, indim, outdim):
        x = getattr(self, f"bin-{indim}")(pad(x, self.max))
        return pad(x[:, :outdim], self.max)


class EndShareBN(BaseBN):
    def __init__(self, indims, outdims):
        super().__init__(indims, outdims)
        for cout in outdims:
            setattr(self, f"bn-{cout}", nn.BatchNorm2d(cout))

    def forward(self, x, indim, outdim):
        x = x[:, :outdim]
        return getattr(self, f"bn-{outdim}")(x)


class FullBN(BaseBN):
    def __init__(self, indims, outdims):
        super().__init__(indims, outdims)
        self.bn = nn.BatchNorm2d(self.max)

    def forward(self, x, indim, outdim):
        x = self.bn(pad(x, self.max))
        x = pad(x[:, :outdim], self.max)
        return x

# CONV PART


class BaseConv(nn.Module):
    def __init__(self, indims, outdims, stride=1, down=False):
        super(BaseConv, self).__init__()
        self.indims = indims
        self.outdims = outdims
        self.max_in = max(self.indims)
        self.max_out = max(self.outdims)
        self.stride = stride
        self.down = down
        self.kernel_size = 1 if down else 3

    def _make_conv(self, indim, outdim):
        """TODO: is bias matter?"""
        return nn.Conv2d(indim, outdim, self.kernel_size, self.stride, get_same_padding(self.kernel_size), bias=False)

    def forward(self, x, indim, outdim):
        raise NotImplementedError()


class IndependentConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        for cin in indims:
            for cout in outdims:
                setattr(self, f"conv-{cin}-{cout}", self._make_conv(cin, cout))

    def forward(self, x, indim, outdim):
        x = x[:, :indim]
        return getattr(self, f"conv-{indim}-{outdim}")(x)


class FrontShareConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        for cin in indims:
            setattr(self, f"conv-{cin}", self._make_conv(cin, self.max_out))

    def forward(self, x, indim, outdim):
        x = x[:, :indim]
        return getattr(self, f"conv-{indim}")(x)


class EndShareConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        for cout in outdims:
            setattr(self, f"conv-{cout}", self._make_conv(self.max_in, cout))

    def forward(self, x, indim, outdim):
        return getattr(self, f"conv-{outdim}")(pad(x[:, :indim], self.max_in))


class FullConv(BaseConv):
    def __init__(self, indims, outdims, stride=1, down=False):
        super().__init__(indims, outdims, stride, down)
        self.conv = nn.Conv2d(self.max_in, self.max_out, self.kernel_size,
                              self.stride, get_same_padding(self.kernel_size), bias=False)

    def forward(self, x, indim, outdim):
        return self.conv(pad(x[:, :indim], self.max_in))

# FC PART


class BaseFC(nn.Module):
    def __init__(self, indims, outdim):
        super(BaseFC, self).__init__()
        self.indims = indims
        self.outdim = outdim
        self.max = max(self.indims)

    def forward(self, x, indim):
        raise NotImplementedError()


class IndependentFC(BaseFC):
    def __init__(self, indims, outdim):
        super().__init__(indims, outdim)
        for cin in indims:
            setattr(self, f"fc-{cin}", nn.Linear(cin, self.outdim))

    def forward(self, x, indim):
        return getattr(self, f'fc-{indim}')(x[:, :indim])


class FullFC(BaseFC):
    def __init__(self, indims, outdim):
        super().__init__(indims, outdim)
        self.fc = nn.Linear(self.max, self.outdim)
        nn.init.kaiming_normal_(
            self.fc.weight, nonlinearity='relu', mode='fan_out')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x, indim):
        x = pad(x[:, :indim], self.max)
        return self.fc(x)
