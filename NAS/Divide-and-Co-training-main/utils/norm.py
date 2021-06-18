# coding=utf-8

import torch
import torch.nn as nn


class Identity(nn.Module):
	__constants__ = []

	def __init__(self, **kwargs):
		super(Identity, self).__init__()
	
	def forward(self, x):
		return x


class LayerNorm2d(nn.Module):
	__constants__ = ['weight', 'bias']

	def __init__(self, eps=1e-05, weight=True, bias=True, **kwargs):
		super(LayerNorm2d, self).__init__()

		self.eps = eps
		self.weight = weight
		self.bias = bias

	def forward(self, x):
		if (not isinstance(self.weight, nn.parameter.Parameter) and
			not isinstance(self.bias, nn.parameter.Parameter) and
			(self.weight == True or self.bias == True)):
			self.init_affine(x)
		return nn.functional.layer_norm(x, x.shape[1:],
										weight=self.weight, bias=self.bias,
										eps=self.eps)

	def init_affine(self, x):
		# Unlike Batch Normalization and Instance Normalization, which applies
		# scalar scale and bias for each entire channel/plane with the affine
		# option, Layer Normalization applies per-element scale and bias
		N, C, H, W = x.shape
		s = [C, H, W]
		if self.weight:
			self.weight = nn.Parameter(torch.ones(s),
										requires_grad=True)
		else:
			self.register_parameter('weight', None)
		if self.bias:
			self.bias = nn.Parameter(torch.zeros(s),
										requires_grad=True)
		else:
			self.register_parameter('bias', None)


class norm(nn.Module):

	def __init__(self, mode='batch', eps=1e-05, momentum=0.1,
					weight=True, bias=True, track_running_stats=True, gn_num_groups=32,
					**kwargs):
		"""
		Function which instantiates a normalization scheme based on mode

		Arguments:
			num_features: :math:`C` from an expected input of size
				:math:`(N, C, H, W)`
			mode: Option to select normalization method (Default: None)
			eps: a value added to the denominator for numerical stability.
				Default: 1e-5
			momentum: the value used for the running_mean and running_var
				computation. Can be set to ``None`` for cumulative moving average
				(i.e. simple average). Default: 0.1
			weight: a boolean value that when set to ``True``, this module has
				learnable linear parameters. Default: ``True``
			bias: a boolean value that when set to ``True``, this module has
				learnable bias parameters. Default: ``True``
			track_running_stats: a boolean value that when set to ``True``, this
				module tracks the running mean and variance, and when set to ``False``,
				this module does not track such statistics and always uses batch
				statistics in both training and eval modes. Argument valid when
				using batch norm. Default: ``True``

		Note:
			1. When using BN affine = weight & bias
		"""
		super(norm, self).__init__()

		self.mode = mode
		if self.mode not in ['batch', 'group', 'layer', 'instance', 'none']:
			raise KeyError('mode options include: "batch" | "group" | "layer" | '
							'"instance" | "none"')
		else:
			print('INFO:PyTorch: Normalizer is {}'.format(self.mode))
		self.eps = eps
		self.momentum = momentum
		self.weight = weight
		self.bias = bias
		self.affine = self.weight and self.bias
		if not self.affine:
			print('affine not used in norm layer')
		self.track_running_stats = track_running_stats
		self.gn_num_groups = gn_num_groups

	def forward(self, num_features):

		if self.mode == 'batch':
			normalizer = nn.BatchNorm2d(num_features=num_features,
											eps=self.eps,
											momentum=self.momentum,
											affine=self.affine,
											track_running_stats=self.track_running_stats)
		elif self.mode == 'group':
			normalizer = nn.GroupNorm(self.gn_num_groups, num_features,
										eps=self.eps, affine=self.affine)

		elif self.mode == 'layer':
			normalizer = LayerNorm2d(eps=self.eps, weight=self.weight, bias=self.bias)

		elif self.mode == 'instance':
			normalizer = nn.InstanceNorm2d(num_features, eps=self.eps, affine=self.affine)
		else:
			normalizer = Identity()

		return normalizer
