""" Activations
A collection of activations fn and modules with a common interface so that they can
easily be swapped. All have an `inplace` arg even if not used.

Reference:
	[1] https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/activations.py
"""

import torch
from torch import nn as nn
from torch.nn import functional as F


def get_act(act, inplace=False, memory_efficient=False):
	"""get the activation functions"""
	if act == 'relu':
		return nn.ReLU(inplace=inplace)

	elif act == 'leakyrelu':
		return nn.LeakyReLU(0.01, inplace=inplace)
	
	elif act == 'swish':
		if memory_efficient:
			return MemoryEfficientSwish()
		else:
			return Swish(inplace=inplace)
	
	elif act == 'hardswish':
		return HardSwish(inplace=inplace)
	
	else:
		raise NotImplementedError


class Swish(nn.Module):
	"""
	Swish: Swish Activation Function
	Described in: https://arxiv.org/abs/1710.05941
	"""
	def __init__(self, inplace=True):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class SwishImplementation(torch.autograd.Function):
	"""
	A memory-efficient implementation of Swish function from
	https://github.com/lukemelas/EfficientNet-PyTorch
	"""

	@staticmethod
	def forward(ctx, i):
		result = i * torch.sigmoid(i)
		ctx.save_for_backward(i)
		return result

	@staticmethod
	def backward(ctx, grad_output):
		i = ctx.saved_tensors[0]
		sigmoid_i = torch.sigmoid(i)
		return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
	def forward(self, x):
		return SwishImplementation.apply(x)


class HardSwish(nn.Module):
	"""
	PyTorch has this, but not with a consistent inplace argmument interface.

	Searching for MobileNetV3`:
		https://arxiv.org/abs/1905.02244

	"""
	def __init__(self, inplace: bool = False):
		super(HardSwish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		return F.hardswish(x, inplace=self.inplace)
