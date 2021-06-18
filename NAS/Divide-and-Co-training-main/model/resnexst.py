# coding=utf-8
"""
ResNeXSt:
A combination of ResNeXt and ResNeSt.

Reference:
	[1] https://github.com/Cadene/pretrained-models.pytorch
	[2] https://github.com/zhanghang1989/ResNeSt
	[3] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

"""

from __future__ import print_function, division, absolute_import
from collections import OrderedDict

import torch.nn as nn
from .layers import get_drop, get_act, SplitAttnConv2d


_act_inplace = True			# use inplace activation or not
_drop_inplace = True		# use inplace drop layer or not


__all__ = ['resnexst50_4x16d', 'resnexst50_8x16d', 'resnexst50_4x32d', 'resnexst101_8x32d']


class ResNeXStBottleneck(nn.Module):
	"""
	ResNeXSt Bottleneck.
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, radix=1,
					reduction=4, stride=1, downsample=None, base_width=4,
					act='relu', drop_p=None, dilation=1, first_dilation=None,
					drop_type='dropout', dropblock_size=0, gamma_scale=1.0,
					conv1_group=True, avd=False, avd_first=False, is_first=False,
					se_reduction=16):
		super(ResNeXStBottleneck, self).__init__()
		width = int(planes * (base_width / 64)) * groups
		first_dilation = first_dilation or dilation

		self.radix = radix
		self.avd = avd and (stride > 1 or is_first)
		self.avd_first = avd_first
		if self.avd:
			self.avd_layer = nn.AvgPool2d(3, stride, padding=1)
			stride = 1

		self.drop_p = drop_p
		self.drop_type = drop_type

		self.conv1_group = 1		# min(max(inplanes // 256, 1), 8)
		self.conv2_group = groups
		self.conv3_group = 1

		# conv 1x1
		self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
								stride=1, groups=self.conv1_group)
		self.bn1 = nn.BatchNorm2d(width)
		self.act1 = get_act(act, inplace=_act_inplace)

		# split attention with conv.and fc. layers
		if self.radix >= 1:
			self.conv2 = SplitAttnConv2d(
				width, width, kernel_size=3, stride=stride,
				padding=first_dilation, dilation=first_dilation,
				groups=self.conv2_group, radix=radix, reduction_factor=reduction,
				norm_layer=nn.BatchNorm2d, drop_block=None)
			self.bn2 = None  # FIXME revisit, here to satisfy current torchscript fussyness
			self.act2 = None
		else:
			self.conv2 = nn.Conv2d(
				width, width, kernel_size=3, stride=stride, padding=first_dilation,
				dilation=first_dilation, groups=self.conv2_group, bias=False)
			self.bn2 = nn.BatchNorm2d(width)
			self.act2 = get_act(act, inplace=_act_inplace)

		# conv 1x1
		self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False,
								groups=self.conv3_group)
		self.bn3 = nn.BatchNorm2d(planes * 4)

		self.downsample = downsample
		self.stride = stride
		self.drop = None
		if self.drop_p:
			# self.drop = nn.Dropout(drop_p, inplace=_drop_inplace)
			self.drop = get_drop(drop_type, self.drop_p, inplace=_drop_inplace,
									block_size=dropblock_size, gamma_scale=gamma_scale)
		self.act3 = get_act(act, inplace=_act_inplace)

	def forward(self, x):
		residual = x

		# step 1. conv 1x1
		out = self.conv1(x)
		out = self.bn1(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)
		out = self.act1(out)

		if self.avd and self.avd_first:
			out = self.avd_layer(out)

		# step 2. split attention convolution
		out = self.conv2(out)
		if self.bn2 is not None:
			out = self.bn2(out)
			if self.drop is not None and self.drop_type == 'dropblock':
				out = self.drop(out)
			out = self.act2(out)

		# step 3. average downsampling
		if self.avd and not self.avd_first:
			out = self.avd_layer(out)

		# step 4. conv 1x1
		out = self.conv3(out)
		out = self.bn3(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)

		# step 5. skip connection
		if self.downsample is not None:
			residual = self.downsample(x)

		if self.drop is not None and self.drop_type != 'dropblock':
			out = self.drop(out)

		out = out + residual

		out = self.act3(out)

		return out


class ResNeXSt(nn.Module):

	def __init__(self, arch, block, layers, radix, groups, reduction=4, dropout_p=0.2,
					base_width=4, inplanes=128, downsample_kernel_size=1,
					num_classes=1000, zero_init_residual=True,
					dataset='imagenet', split_factor=1, output_stride=8,
					act='relu', mix_act=False, mix_act_block=2,
					block_drop=False, block_drop_p=0.1,
					drop_type='dropout', crop_size=224,
					avd=False, avd_first=False
					):
		"""
		Parameters
		----------
		block (nn.Module): Bottleneck class.

		layers (list of ints): Number of residual blocks for 4 layers of the
			network (layer1...layer4).
		groups (int): Number of groups for the 3x3 convolution in each

		reduction (int): Reduction factor for Split Attention modules.

		dropout_p (float or None): Drop probability for the Dropout layer.
			If `None` the Dropout layer is not used.

		inplanes (int):  Number of input channels for layer1.

		downsample_kernel_size (int): Kernel size for downsampling convolutions
			in layer2, layer3 and layer4.

		downsample_padding (int): Padding for downsampling convolutions in
			layer2, layer3 and layer4.

		num_classes (int): Number of outputs in `last_linear` layer.
			- For all models: 1000
		
		dataset (str): 'imagenet', 'cifar10', 'cifar100'

		split_factor (int): divide the network into {split_factor} small networks
		
		mix_act (bool): whether use mixed activations, {ReLU and HardSwish}
		
		mix_act_block (int): the last (4 - {mix_act_block}) blocks use HardSwish act function
		
		block_drop (bool): whether use block drop layer or not
		
		drop_type (int): 'dropout', 'dropblock', 'droppath'
		
		block_drop_p (folat): drop probablity in drop layers
			- For dropout, 0.2 or 0.3
			- For dropblock, 0.1
			- For droppath, 0.1 or 0.2
		"""
		super(ResNeXSt, self).__init__()
		self.base_width = base_width
		self.dataset = dataset
		self.radix = radix
		self.avd = avd
		self.avd_first = avd_first

		# modification of activations
		self.act = act
		self.mix_act = mix_act
		self.mix_act_block = mix_act_block if self.mix_act else len(layers) + 1
		if self.mix_act_block < 4:
			print('INFO:PyTorch: last {} block(s) use'
					' hardswish activation function'.format(4 - self.mix_act_block))

		# modification of drop blocks
		self.crop_size = crop_size
		self.block_drop = block_drop
		dropblock_size = [0, 0, 3, 3]
		self.gamma_scales = [0, 0, 1.0, 1.0]
		self.dropblock_size = [int(x * crop_size / 224) for x in dropblock_size]
		self.drop_type = drop_type
		if self.block_drop:
			# add dropout or other drop layers within each block
			print('INFO:PyTorch: Using {} within blocks'.format(self.drop_type))
			block_drop_p = block_drop_p / (split_factor ** 0.5)
			n = sum(layers)
			if self.drop_type in ['dropout', 'droppath']:
				self.block_drop_ps = [block_drop_p * (i + 1) / (n + 1) for i in range(n)]
			else:
				block_drop_flag = [False, False, True, True]
				self.block_drop_ps = [block_drop_p] * n
				# a mixed drop manner
				j = 0
				for i in range(len(block_drop_flag)):
					if not block_drop_flag[i]:
						for k in range(j, j + layers[i]):
							self.block_drop_ps[k] = 0
						j += layers[i]

		# inplanes and base width of the bottleneck
		if groups == 1:
			self.groups = groups
			inplanes_dict = {'imagenet': {1: 64, 2: 44, 4: 32, 8: 24},
								'cifar10': {1: 16, 2: 12, 4: 8},
								'cifar100': {1: 16, 2: 12, 4: 8},
								'svhn': {1: 16, 2: 12, 4: 8},
							}

			self.inplanes = inplanes_dict[dataset][split_factor]

		elif groups in [4, 8, 16, 32, 64, 128]:
			# For resnext, just divide groups
			self.groups = groups
			if split_factor > 1:
				self.groups = int(groups / split_factor)
				print("INFO:PyTorch: Dividing {}, change groups from {} "
						"to {}.".format(arch, groups, self.groups))
			self.inplanes = 64
		
		else:
			raise NotImplementedError

		self.layer0_inplanes = self.inplanes

		if inplanes == 128:
			self.inplanes = self.inplanes * 2
		print('INFO:PyTorch: The initial inplanes of ResNeXSt is {}'.format(self.inplanes))
		print('INFO:PyTorch: The reduction of ResNeXSt is {}'.format(reduction))

		if 'imagenet' in dataset:
			layer0_modules = [
					('conv1', nn.Conv2d(3, self.layer0_inplanes, 3, stride=2, padding=1,
										bias=False)),
					('bn1', nn.BatchNorm2d(self.layer0_inplanes)),
					('relu1', get_act(act, inplace=_act_inplace)),
					('conv2', nn.Conv2d(self.layer0_inplanes, self.layer0_inplanes, 3, stride=1, padding=1,
										bias=False)),
					('bn2', nn.BatchNorm2d(self.layer0_inplanes)),
					('relu2', get_act(act, inplace=_act_inplace)),
					('conv3', nn.Conv2d(self.layer0_inplanes, self.inplanes, 3, stride=1, padding=1,
										bias=False)),
					('bn3', nn.BatchNorm2d(self.inplanes)),
					('relu3', get_act(act, inplace=_act_inplace)),
					('pool', nn.MaxPool2d(3, stride=2, ceil_mode=True)),
						]
			strides = [1, 2, 2, 2]

		elif 'cifar' in dataset or 'svhn' in dataset:
			layer0_modules = [
					('conv1', nn.Conv2d(3, self.inplanes, 3, stride=1, padding=1, bias=False)),
					('bn1', nn.BatchNorm2d(self.inplanes)),
					('relu1', get_act(act, inplace=_act_inplace)),
						]
			strides = [1, 2, 2, 1]
		else:
			raise NotImplementedError
		
		self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
		
		# For CIFAR/SVHN, layer1 - layer3, channels - [16, 32, 64]
		# For ImageNet, layer1 - layer4, channels - [64, 128, 256, 512]
		inplanes_origin = self.layer0_inplanes

		self.layer1 = self._make_layer(
			block,
			planes=inplanes_origin,
			blocks=layers[0],
			stride=strides[0],
			groups=self.groups,
			reduction=reduction,
			downsample_kernel_size=1,
			act=act,
			dropblock_size=self.dropblock_size[0],
			gamma_scale=self.gamma_scales[0],
			is_first=False
		)
		self.layer2 = self._make_layer(
			block,
			planes=inplanes_origin * 2,
			blocks=layers[1],
			stride=strides[1],
			groups=self.groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			act='hardswish' if self.mix_act_block < 2 else act,
			dropblock_size=self.dropblock_size[1],
			gamma_scale=self.gamma_scales[1]
		)
		self.layer3 = self._make_layer(
			block,
			planes=inplanes_origin * 4,
			blocks=layers[2],
			stride=strides[2],
			groups=self.groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			act='hardswish' if self.mix_act_block < 3 else act,
			dropblock_size=self.dropblock_size[2],
			gamma_scale=self.gamma_scales[2]
		)
		inplanes_now = inplanes_origin * 4
		
		self.layer4 = None
		if 'imagenet' in dataset:
			print('INFO:PyTorch: Using layer4 for ImageNet Training')
			self.layer4 = self._make_layer(
				block,
				planes=inplanes_origin * 8,
				blocks=layers[3],
				stride=strides[3],
				groups=self.groups,
				reduction=reduction,
				downsample_kernel_size=downsample_kernel_size,
				act='hardswish' if self.mix_act_block < 4 else act,
				dropblock_size=self.dropblock_size[3],
				gamma_scale=self.gamma_scales[3]
			)
			inplanes_now = inplanes_origin * 8
		
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		
		self.dropout = None
		if dropout_p is not None:
			dropout_p = dropout_p / (split_factor ** 0.5)
			# You can also use the below code.
			# dropout_p = dropout_p / split_factor
			print('INFO:PyTorch: Using dropout before last fc layer with ratio {}'.format(dropout_p))
			self.dropout = nn.Dropout(dropout_p, inplace=True)

		self.last_linear = nn.Linear(inplanes_now * block.expansion, num_classes)

		# initialize the parameters
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				# nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, std=1e-3)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)

	def _make_layer(self, block, planes, blocks, groups, reduction, stride=1,
					downsample_kernel_size=1,
					act='relu', dropblock_size=0, gamma_scale=1.0,
					is_first=True):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			if downsample_kernel_size != 1:
				# using conv3x3 to downsample
				downsample = nn.Sequential(
						nn.Conv2d(self.inplanes, planes * block.expansion,
									kernel_size=3, stride=stride, padding=1, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
				)
			else:
				# otherwise, using 2x2 average pooling to reserve information
				if stride == 1:
					downsample = nn.Sequential(
						nn.Conv2d(self.inplanes, planes * block.expansion,
									kernel_size=1, stride=stride, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
					)
				else:
					downsample = nn.Sequential(
						nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True,
										padding=0, count_include_pad=False),
						nn.Conv2d(self.inplanes, planes * block.expansion,
										kernel_size=1, stride=1, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
					)

		layers = []

		layers.append(block(self.inplanes, planes, groups,
							radix=self.radix, reduction=reduction,
							stride=stride, downsample=downsample,
							base_width=self.base_width, act=act,
							drop_p=self.block_drop_ps.pop(0) if self.block_drop else None,
							drop_type=self.drop_type,
							dropblock_size=dropblock_size,
							gamma_scale=gamma_scale,
							avd=self.avd, avd_first=self.avd_first,
							is_first=is_first))
		self.inplanes = planes * block.expansion

		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups,
							radix=self.radix, reduction=reduction,
							base_width=self.base_width, act=act,
							drop_p=self.block_drop_ps.pop(0) if self.block_drop else None,
							drop_type=self.drop_type,
							dropblock_size=dropblock_size,
							gamma_scale=gamma_scale,
							avd=self.avd, avd_first=self.avd_first,
							is_first=is_first))

		return nn.Sequential(*layers)

	def features(self, x):
		x = self.layer0(x)
		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		if self.layer4 is not None:
			x = self.layer4(x)
		return x

	def logits(self, x):
		x = self.avg_pool(x)
		if self.dropout is not None:
			x = self.dropout(x)
		x = x.view(x.size(0), -1)
		x = self.last_linear(x)
		return x

	def forward(self, x):
		x = self.features(x)
		x = self.logits(x)
		return x


def resnexst50_4x16d(num_classes=1000, pretrained=None, **kwargs):
	model = ResNeXSt('resnexst50_4x16d', ResNeXStBottleneck, [3, 4, 6, 3],
					radix=2, base_width=16, groups=4,
					dropout_p=0.0, num_classes=num_classes,
					avd=True, avd_first=False, **kwargs)
	if pretrained is not None:
		raise NotImplementedError
	return model


def resnexst50_8x16d(num_classes=1000, pretrained=None, **kwargs):
	model = ResNeXSt('resnexst50_8x16d', ResNeXStBottleneck, [3, 4, 6, 3],
					radix=2, base_width=16, groups=8,
					dropout_p=0.0, num_classes=num_classes,
					avd=True, avd_first=False, **kwargs)
	if pretrained is not None:
		raise NotImplementedError
	return model


def resnexst50_4x32d(num_classes=1000, pretrained=None, **kwargs):
	model = ResNeXSt('resnexst50_4x32d', ResNeXStBottleneck, [3, 4, 6, 3],
					radix=2, base_width=32, groups=4,
					dropout_p=0.0, num_classes=num_classes,
					avd=True, avd_first=False, **kwargs)
	if pretrained is not None:
		raise NotImplementedError
	return model


def resnexst101_8x32d(num_classes=1000, pretrained=None, **kwargs):
	model = ResNeXSt('resnexst101_8x32d', ResNeXStBottleneck, [3, 4, 23, 3],
					radix=2, base_width=32, groups=8,
					dropout_p=0.1, num_classes=num_classes,
					avd=True, avd_first=False, **kwargs)
	if pretrained is not None:
		raise NotImplementedError
	return model
