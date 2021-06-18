# coding=utf-8
"""
PyTorch implementation for EfficientNet

Class:
		> Swish
		> SEBlock
		> MBConvBlock
		> EfficientNet

Reference:
	[1] LightNet++: Boosted Light-weighted Networks for Real-time Semantic Segmentation.

	[2] Huijun Liu M.Sc. https://github.com/ansleliu/EfficientNet.PyTorch. 08.02.2020.

	[3] https://github.com/lukemelas/EfficientNet-PyTorch.
"""

import math
from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F


__all__ = ['get_efficientnet']


class Swish(nn.Module):
	"""
	Swish: Swish Activation Function
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


class ConvBlock(nn.Module):
	def __init__(self, in_planes, out_planes, kernel_size, stride=1,
					groups=1, dilate=1, memory_efficient=False):

		super(ConvBlock, self).__init__()
		dilate = 1 if stride > 1 else dilate
		padding = ((kernel_size - 1) // 2) * dilate

		self.conv_block = nn.Sequential(OrderedDict([
			("conv", nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
								kernel_size=kernel_size, stride=stride, padding=padding,
								dilation=dilate, groups=groups, bias=False)),
			("norm", nn.BatchNorm2d(num_features=out_planes,
									eps=1e-3, momentum=0.01)),
			("act", Swish(inplace=True) if not memory_efficient else MemoryEfficientSwish())
		]))

	def forward(self, x):
		return self.conv_block(x)


class SEBlock(nn.Module):
	"""
	SEBlock: Squeeze & Excitation (SCSE), namely, Channel-wise Attention
	"""
	def __init__(self, in_planes, reduced_dim, memory_efficient=False):
		super(SEBlock, self).__init__()
		self.channel_se = nn.Sequential(OrderedDict([
			("linear1", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
			("act", Swish(inplace=True) if not memory_efficient else MemoryEfficientSwish()),
			("linear2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
		]))

	def forward(self, x):
		x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x, output_size=(1, 1))))
		return torch.mul(x, x_se)


class MBConvBlock(nn.Module):
	"""
	MBConvBlock: MBConvBlock for EfficientNet
	"""
	def __init__(self, in_planes, out_planes,
					expand_ratio, kernel_size, stride, dilate,
					reduction_ratio=4, dropout_rate=0.2,
					memory_efficient=False):
		super(MBConvBlock, self).__init__()
		self.dropout_rate = dropout_rate
		self.expand_ratio = expand_ratio
		self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
		self.use_residual = in_planes == out_planes and stride == 1

		assert stride in [1, 2]
		assert kernel_size in [3, 5]
		dilate = 1 if stride > 1 else dilate
		hidden_dim = in_planes * expand_ratio
		reduced_dim = max(1, int(in_planes / reduction_ratio))

		# step 1. Expansion phase/Point-wise convolution
		if expand_ratio != 1:
			self.expansion = ConvBlock(in_planes, hidden_dim, 1, memory_efficient=memory_efficient)

		# step 2. Depth-wise convolution phase
		self.depth_wise = ConvBlock(hidden_dim, hidden_dim, kernel_size,
									stride=stride, groups=hidden_dim, dilate=dilate,
									memory_efficient=memory_efficient)
		# step 3. Squeeze and Excitation
		if self.use_se:
			self.se_block = SEBlock(hidden_dim, reduced_dim, memory_efficient=memory_efficient)

		# step 4. Point-wise convolution phase
		self.point_wise = nn.Sequential(OrderedDict([
			("conv", nn.Conv2d(in_channels=hidden_dim,
								out_channels=out_planes, kernel_size=1,
								stride=1, padding=0, dilation=1, groups=1, bias=False)),
			("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01))
		]))

	def forward(self, x):
		res = x

		# step 1. Expansion phase/Point-wise convolution
		if self.expand_ratio != 1:
			x = self.expansion(x)

		# step 2. Depth-wise convolution phase
		x = self.depth_wise(x)

		# step 3. Squeeze and Excitation
		if self.use_se:
			x = self.se_block(x)

		# step 4. Point-wise convolution phase
		x = self.point_wise(x)

		# step 5. Skip connection and drop connect
		if self.use_residual:
			if self.training and (self.dropout_rate is not None):
				x = F.dropout2d(input=x, p=self.dropout_rate,
								training=self.training, inplace=True)
			x = x + res

		return x


def get_efficientnet(arch='efficientnetb0', **kwargs):
	return EfficientNet(arch=arch[-2:], **kwargs)


class EfficientNet(nn.Module):
	"""EfficientNet: EfficientNet Implementation"""
	def __init__(self, arch="bo", num_classes=10, dataset='cifar10', split_factor=1,
					is_dropout=True, is_efficientnet_user_crop=False, crop_size=224,
					memory_efficient=False):
		super(EfficientNet, self).__init__()

		if split_factor == 1:
			settings = [
				# t, c,  n, k, s, d
				# t --- channel expand ratio of MBConvBlock
				# c --- number of channels
				# n --- (depth) number of layers in certain block
				# k --- the size of conv kernel
				# s --- stride of the conv
				# d --- dilation of the conv
				[1, 16, 1, 3, 1, 1],   # 3x3, 112 -> 112
				[6, 24, 2, 3, 2, 1],   # 3x3, 112 ->  56
				[6, 40, 2, 5, 2, 1],   # 5x5, 56  ->  28
				[6, 80, 3, 3, 2, 1],   # 3x3, 28  ->  14
				[6, 112, 3, 5, 1, 1],  # 5x5, 14  ->  14
				[6, 192, 4, 5, 2, 1],  # 5x5, 14  ->   7
				[6, 320, 1, 3, 1, 1],  # 3x3, 7   ->   7
			]
			self.last_ch = 1280
			divisor = 8
			out_channels_mod1 = 32
		elif split_factor == 2:
			settings = [
				# t, c,  n, k, s, d
				[1, 12, 1, 3, 1, 1],   	# 3x3, 112 -> 112
				[6, 16, 2, 3, 2, 1],   	# 3x3, 112 ->  56
				[6, 24, 2, 5, 2, 1],   	# 5x5, 56  ->  28
				[6, 56, 3, 3, 2, 1],   	# 3x3, 28  ->  14
				[6, 80, 3, 5, 1, 1],  	# 5x5, 14  ->  14
				[6, 136, 4, 5, 2, 1],  	# 5x5, 14  ->   7
				[6, 224, 1, 3, 1, 1],  	# 3x3, 7   ->   7
			]
			self.last_ch = 920
			divisor = 4
			out_channels_mod1 = 24
		elif split_factor == 4:
			settings = [
				# t, c,  n, k, s, d
				[1, 12, 1, 3, 1, 1],   	# 3x3, 112 -> 112
				[6, 16, 2, 3, 2, 1],   	# 3x3, 112 ->  56
				[6, 20, 2, 5, 2, 1],   	# 5x5, 56  ->  28
				[6, 40, 3, 3, 2, 1],   	# 3x3, 28  ->  14
				[6, 56, 3, 5, 1, 1],  	# 5x5, 14  ->  14
				[6, 96, 4, 5, 2, 1], 	# 5x5, 14  ->   7
				[6, 160, 1, 3, 1, 1],  	# 3x3, 7   ->   7
			]
			self.last_ch = 640
			divisor = 4
			out_channels_mod1 = 16
		else:
			raise NotImplementedError

		if dataset == 'imagenet':
			arch_params = {
				# arch width_multi depth_multi input_h dropout_rate
				'b0': (1.0, 1.0, 224, 0.2),
				'b1': (1.0, 1.1, 240, 0.2),
				'b2': (1.1, 1.2, 260, 0.3),
				'b3': (1.2, 1.4, 300, 0.3),
				'b4': (1.4, 1.8, 380, 0.4),
				'b5': (1.6, 2.2, 456, 0.4),
				'b6': (1.8, 2.6, 528, 0.5),
				'b7': (2.0, 3.1, 600, 0.5),
				'b8': (2.2, 3.6, 672, 0.5),
				'l2': (4.3, 5.3, 800, 0.5),
			}
		elif dataset in ['cifar10', 'cifar100', 'svhn']:
			arch_params = {
				# arch width_multi depth_multi input_h dropout_rate
				'b0': (1.0, 1.0, 32, 0.2),
				'b1': (1.0, 1.1, 32, 0.2),
				'b2': (1.1, 1.2, 32, 0.3),
				'b3': (1.2, 1.4, 32, 0.3),
				'b4': (1.4, 1.8, 32, 0.4),
				'b5': (1.6, 2.2, 32, 0.4),
				'b6': (1.8, 2.6, 32, 0.5),
				'b7': (2.0, 3.1, 32, 0.5),
			}
			settings[2][4] = 1
			settings[5][4] = 1
			
		else:
			raise NotImplementedError

		width_multi, depth_multi, net_h, dropout_rate = arch_params[arch]
		# dropout ratio also change with the split factor
		self.dropout_rate = dropout_rate / (split_factor ** 0.5) if is_dropout else None
		# To save memory, use user crop size (e.g., 224) for efficientnet
		if is_efficientnet_user_crop:
			net_h = crop_size

		# first conv
		in_channels = 3
		out_channels = self._round_filters(out_channels_mod1, width_multi, divisor=divisor)
		first_conv_stride = 2 if dataset == 'imagenet' else 1
		print('INFO:PyTorch: The stride of the fist conv layer is {}.'.format(first_conv_stride))
		if memory_efficient:
			print('INFO:PyTorch: Using memory efficient Swish function in EfficientNet.')
		self.mod1 = ConvBlock(in_channels, out_channels,
								kernel_size=3, stride=first_conv_stride,
								groups=1, dilate=1, memory_efficient=memory_efficient)

		in_channels = out_channels
		drop_rate = self.dropout_rate
		mod_id = 0
		for t, c, n, k, s, d in settings:
			out_channels = self._round_filters(c, width_multi, divisor=divisor)
			repeats = self._round_repeats(n, depth_multi)

			if self.dropout_rate:
				drop_rate = self.dropout_rate * float(mod_id + 1) / len(settings)

			# Create blocks for module
			blocks = []
			for block_id in range(repeats):
				stride = s if block_id == 0 else 1
				dilate = d if stride == 1 else 1

				blocks.append(("block%d" % (block_id + 1),
								MBConvBlock(in_channels, out_channels,
												expand_ratio=t, kernel_size=k,
												stride=stride, dilate=dilate,
												dropout_rate=drop_rate,
												memory_efficient=memory_efficient)
								))

				in_channels = out_channels
			self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
			mod_id += 1

		self.last_channels = self._round_filters(self.last_ch, width_multi, divisor=divisor)
		self.last_feat = ConvBlock(in_channels, self.last_channels, 1,
									memory_efficient=memory_efficient)
		self.classifier = nn.Linear(self.last_channels, num_classes)

		self._initialize_weights()

	def _initialize_weights(self):
		# weight initialization
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# nn.init.xavier_uniform_(m.weight)
				nn.init.kaiming_normal_(m.weight, mode='fan_out')
				if m.bias is not None:
					nn.init.zeros_(m.bias)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
				nn.init.ones_(m.weight)
				nn.init.zeros_(m.bias)
			elif isinstance(m, nn.Linear):
				fan_out = m.weight.size(0)
				init_range = 1.0 / math.sqrt(fan_out)
				nn.init.uniform_(m.weight, -init_range, init_range)
				if m.bias is not None:
					nn.init.zeros_(m.bias)

	@staticmethod
	def _make_divisible(value, divisor=8):
		new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
		if new_value < 0.9 * value:
			new_value += divisor
		return new_value

	def _round_filters(self, filters, width_multi, divisor=8):
		if width_multi == 1.0:
			return filters
		return int(self._make_divisible(filters * width_multi, divisor=divisor))

	@staticmethod
	def _round_repeats(repeats, depth_multi):
		if depth_multi == 1.0:
			return repeats
		return int(math.ceil(depth_multi * repeats))

	def forward(self, x):
		x = self.mod2(self.mod1(x))   # (N, 16,   H/2,  W/2)
		x = self.mod3(x)              # (N, 24,   H/4,  W/4)
		x = self.mod4(x)              # (N, 32,   H/8,  W/8)
		x = self.mod6(self.mod5(x))   # (N, 96,   H/16, W/16)
		x = self.mod8(self.mod7(x))   # (N, 320,  H/32, W/32)
		x = self.last_feat(x)

		x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.last_channels)
		if self.training and (self.dropout_rate is not None):
			x = F.dropout(input=x, p=self.dropout_rate,
							training=self.training, inplace=True)
		x = self.classifier(x)
		return x


if __name__ == "__main__":
	'''
	import os
	import time
	from torchstat import stat
	from pytorch_memlab import MemReporter

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	arch = "b6"
	img_preparam = {"b0": (224, 0.875),
					"b1": (240, 0.882),
					"b2": (260, 0.890),
					"b3": (300, 0.904),
					"b4": (380, 0.922),
					"b5": (456, 0.934),
					"b6": (528, 0.942),
					"b7": (600, 0.949)}
	net_h = img_preparam[arch][0]
	model = EfficientNet(arch=arch, num_classes=1000)
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-1,
								momentum=0.90, weight_decay=1.0e-4, nesterov=True)
	
	# stat(model, (3, net_h, net_h))

	model = model.cuda().train()
	loss_func = nn.CrossEntropyLoss().cuda()
	dummy_in = torch.randn(2, 3, net_h, net_h).cuda().requires_grad_()
	dummy_target = torch.ones(2).cuda().long().cuda()
	reporter = MemReporter(model)
	
	optimizer.zero_grad()
	dummy_out = model(dummy_in)
	loss = loss_func(dummy_out, dummy_target)
	print('========================================== before backward ===========================================')
	reporter.report()
	
	loss.backward()
	optimizer.step()
	print('========================================== after backward =============================================')
	reporter.report()
	'''
	kwargs = {'memory_efficient': True}
	model = get_efficientnet("efficientnetb1", **kwargs)
	print(model.mod1)
