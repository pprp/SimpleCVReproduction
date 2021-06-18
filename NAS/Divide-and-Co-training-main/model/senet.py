# coding=utf-8
"""
Reference:
	[1] https://github.com/Cadene/pretrained-models.pytorch

	[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""
from __future__ import print_function, division, absolute_import
from collections import OrderedDict
import math

import torch.nn as nn
from torch.utils import model_zoo
from .layers import get_drop, get_act


__all__ = ['SENet', 'senet154', 'se_resnet50', 'se_resnet50_B',
			'se_resnet101', 'se_resnet152',
			'se_resnet110', 'se_resnet164',
			'se_resnext50_32x4d', 'se_resnext101_32x4d',
			'se_resnext101_64x4d', 'se_resnext101_64x4d_B',
			'senet113']


pretrained_settings = {
	'senet154': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/senet154-c7b49a05.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnet50': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet50-ce0d4300.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnet101': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet101-7e38fcc6.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnet152': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnet152-d17c99b7.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnext50_32x4d': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext50_32x4d-a260b3a4.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
	'se_resnext101_32x4d': {
		'imagenet': {
			'url': 'http://data.lip6.fr/cadene/pretrainedmodels/se_resnext101_32x4d-3b2fe3d8.pth',
			'input_space': 'RGB',
			'input_size': [3, 224, 224],
			'input_range': [0, 1],
			'mean': [0.485, 0.456, 0.406],
			'std': [0.229, 0.224, 0.225],
			'num_classes': 1000
		}
	},
}


_act_inplace = True			# use inplace activation or not
_drop_inplace = True		# use inplace drop layer or not


class SEModule(nn.Module):

	def __init__(self, channels, reduction, act='relu'):
		super(SEModule, self).__init__()
		self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
								padding=0)
		
		# self.relu = nn.ReLU(inplace=True)
		self.relu = get_act(act, inplace=_act_inplace)
		
		self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
								padding=0)
		self.sigmoid = nn.Sigmoid()

	def forward(self, x):
		module_input = x
		x = self.avg_pool(x)
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)
		x = self.sigmoid(x)
		return module_input * x


class Bottleneck(nn.Module):
	"""
	Base class for bottlenecks that implements `forward()` method.
	"""
	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)
		if self.drop is not None and self.drop_type == 'dropblock':
			out = self.drop(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out = self.se_module(out)

		# ResNeSt use dropblock afer each BN (&downsample)
		# https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/gluon/resnet.py
		if self.drop is not None and self.drop_type != 'dropblock':
			out = self.drop(out)

		out = out + residual

		out = self.relu(out)

		return out


class SEBottleneck(Bottleneck):
	"""
	Bottleneck for SENet154.
	Ref:
	https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/senet.py#L134
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, reduction, stride=1,
					downsample=None, drop_p=None, act='relu',
					drop_type='dropout', dropblock_size=0, gamma_scale=1.0):
		super(SEBottleneck, self).__init__()
		# For senet154, the default number of groups is 64
		groups_ratio = 1.0 * groups / 64.0
		width_1 = int(2 * groups_ratio * planes)
		width_2 = int(4 * groups_ratio * planes)
		self.drop_p = drop_p
		self.drop_type = drop_type

		# inplanes -> planes * 2 (conv1x1)
		# planes * 2  -> planes * 4 (conv3x3)
		# planes * 4  -> planes * 4 (conv3x3)
		self.conv1 = nn.Conv2d(inplanes, width_1, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(width_1)
		self.conv2 = nn.Conv2d(width_1, width_2, kernel_size=3,
								stride=stride, padding=1, groups=groups,
								bias=False)
		self.bn2 = nn.BatchNorm2d(width_2)
		self.conv3 = nn.Conv2d(width_2, planes * 4, kernel_size=1,
								bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		
		# self.relu = nn.ReLU(inplace=True)
		self.relu = get_act(act, inplace=_act_inplace)
		
		self.se_module = SEModule(planes * 4, reduction=reduction, act=act)
		self.downsample = downsample
		self.stride = stride
		self.drop = None
		if self.drop_p:
			# self.drop = nn.Dropout(drop_p, inplace=_drop_inplace)
			self.drop = get_drop(drop_type, self.drop_p, inplace=_drop_inplace,
									block_size=dropblock_size, gamma_scale=gamma_scale)


class SEResNetBottleneck(Bottleneck):
	"""
	ResNet bottleneck with a Squeeze-and-Excitation module. It follows Caffe
	implementation and uses `stride=stride` in `conv1` and not in `conv2`
	(the latter is used in the torchvision implementation of ResNet).
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, reduction, stride=1,
					downsample=None, drop_p=None, act='relu',
					drop_type='dropout', dropblock_size=0, gamma_scale=1.0):
		super(SEResNetBottleneck, self).__init__()
		self.drop_p = drop_p
		self.drop_type = drop_type

		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False,
								stride=1)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
								groups=groups, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		
		# self.relu = nn.ReLU(inplace=True)
		self.relu = get_act(act, inplace=_act_inplace)

		self.se_module = SEModule(planes * 4, reduction=reduction, act=act)
		self.downsample = downsample
		self.stride = stride
		self.drop = None
		if self.drop_p:
			# self.drop = nn.Dropout(drop_p, inplace=_drop_inplace)
			self.drop = get_drop(drop_type, self.drop_p, inplace=_drop_inplace,
									block_size=dropblock_size, gamma_scale=gamma_scale)


class SEResNeXtBottleneck(Bottleneck):
	"""
	ResNeXt bottleneck type C with a Squeeze-and-Excitation module.
	"""
	expansion = 4

	def __init__(self, inplanes, planes, groups, reduction, stride=1,
					downsample=None, base_width=4,
					act='relu', drop_p=None,
					drop_type='dropout', dropblock_size=0, gamma_scale=1.0):
		super(SEResNeXtBottleneck, self).__init__()
		width = math.floor(planes * (base_width / 64)) * groups
		self.drop_p = drop_p
		self.drop_type = drop_type

		self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False,
								stride=1)
		self.bn1 = nn.BatchNorm2d(width)
		self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
								padding=1, groups=groups, bias=False)
		self.bn2 = nn.BatchNorm2d(width)
		self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		
		# self.relu = nn.ReLU(inplace=True)
		self.relu = get_act(act, inplace=_act_inplace)

		self.se_module = SEModule(planes * 4, reduction=reduction, act=act)
		self.downsample = downsample
		self.stride = stride
		self.drop = None
		if self.drop_p:
			# self.drop = nn.Dropout(drop_p, inplace=_drop_inplace)
			self.drop = get_drop(drop_type, self.drop_p, inplace=_drop_inplace,
									block_size=dropblock_size, gamma_scale=gamma_scale)


class SENet(nn.Module):

	def __init__(self, arch, block, layers, groups, reduction, dropout_p=0.2,
					inplanes=128, input_3x3=True, downsample_kernel_size=3,
					downsample_padding=1, num_classes=1000, zero_init_residual=True,
					dataset='imagenet', split_factor=1, output_stride=8,
					act='relu', mix_act=False, mix_act_block=2,
					block_drop=False, block_drop_p=0.5,
					drop_type='dropout', crop_size=224
					):
		"""
		Parameters
		----------
		block (nn.Module): Bottleneck class.
			- For SENet154: SEBottleneck
			- For SE-ResNet models: SEResNetBottleneck
			- For SE-ResNeXt models:  SEResNeXtBottleneck
		layers (list of ints): Number of residual blocks for 4 layers of the
			network (layer1...layer4).
		groups (int): Number of groups for the 3x3 convolution in each
			bottleneck block.
			- For SENet154: 64
			- For SE-ResNet models: 1
			- For SE-ResNeXt models:  32
		reduction (int): Reduction ratio for Squeeze-and-Excitation modules.
			- For all models: 16
		dropout_p (float or None): Drop probability for the Dropout layer.
			If `None` the Dropout layer is not used.
			- For SENet154: 0.2
			- For SE-ResNet models: None
			- For SE-ResNeXt models: None
		inplanes (int):  Number of input channels for layer1.
			- For SENet154: 128
			- For SE-ResNet models: 64
			- For SE-ResNeXt models: 64
		input_3x3 (bool): If `True`, use three 3x3 convolutions instead of
			a single 7x7 convolution in layer0.
			- For SENet154: True
			- For SE-ResNet models: False
			- For SE-ResNeXt models: False
		downsample_kernel_size (int): Kernel size for downsampling convolutions
			in layer2, layer3 and layer4.
			- For SENet154: 3
			- For SE-ResNet models: 1
			- For SE-ResNeXt models: 1
		downsample_padding (int): Padding for downsampling convolutions in
			layer2, layer3 and layer4.
			- For SENet154: 1
			- For SE-ResNet models: 0
			- For SE-ResNeXt models: 0
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
		super(SENet, self).__init__()
		self.dataset = dataset

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

			if 'cifar' in dataset or 'svhn' in dataset:
				reduction = 4

		elif groups in [8, 16, 32, 64, 128]:
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
		print('INFO:PyTorch: The initial inplanes of SENet is {}'.format(self.inplanes))
		print('INFO:PyTorch: The reduction of SENet is {}'.format(reduction))

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
			downsample_padding=0,
			act=act,
			dropblock_size=self.dropblock_size[0],
			gamma_scale=self.gamma_scales[0]
		)
		self.layer2 = self._make_layer(
			block,
			planes=inplanes_origin * 2,
			blocks=layers[1],
			stride=strides[1],
			groups=self.groups,
			reduction=reduction,
			downsample_kernel_size=downsample_kernel_size,
			downsample_padding=downsample_padding,
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
			downsample_padding=downsample_padding,
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
				downsample_padding=downsample_padding,
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
					downsample_kernel_size=1, downsample_padding=0,
					act='relu', dropblock_size=0, gamma_scale=1.0):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			if downsample_kernel_size != 1:
				# using conv3x3 to reserve information in SENet154
				downsample = nn.Sequential(
						nn.Conv2d(self.inplanes, planes * block.expansion,
									kernel_size=downsample_kernel_size, stride=stride,
									padding=downsample_padding, bias=False),
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
						# Ref:
						# Bag of Tricks for Image Classification with Convolutional Neural Networks, 2018
						# https://arxiv.org/abs/1812.01187
						# https://github.com/rwightman/pytorch-image-models/blob
						# /5966654052b24d99e4bfbcf1b59faae8a75db1fd/timm/models/resnet.py#L293
						# nn.AvgPool2d(kernel_size=3, stride=stride, padding=1),
						nn.AvgPool2d(kernel_size=2, stride=stride, ceil_mode=True,
										padding=0, count_include_pad=False),
						nn.Conv2d(self.inplanes, planes * block.expansion,
										kernel_size=1, stride=1, bias=False),
						nn.BatchNorm2d(planes * block.expansion),
					)

		layers = []
		layers.append(block(self.inplanes, planes, groups, reduction, stride,
							downsample, act=act,
							drop_p=self.block_drop_ps.pop(0) if self.block_drop else None,
							drop_type=self.drop_type,
							dropblock_size=dropblock_size,
							gamma_scale=gamma_scale))
		self.inplanes = planes * block.expansion

		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes, groups, reduction,
							act=act,
							drop_p=self.block_drop_ps.pop(0) if self.block_drop else None,
							drop_type=self.drop_type,
							dropblock_size=dropblock_size,
							gamma_scale=gamma_scale))

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


def initialize_pretrained_model(model, num_classes, settings):
	assert num_classes == settings['num_classes'], \
		'num_classes should be {}, but is {}'.format(
			settings['num_classes'], num_classes)
	model.load_state_dict(model_zoo.load_url(settings['url']))
	model.input_space = settings['input_space']
	model.input_size = settings['input_size']
	model.input_range = settings['input_range']
	model.mean = settings['mean']
	model.std = settings['std']


def senet154(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('senet154', SEBottleneck, [3, 8, 36, 3], groups=64, reduction=16,
					dropout_p=0.2, num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['senet154'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnet50(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnet50', SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
					dropout_p=0.2, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet50'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnet50_B(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnet50', SEResNetBottleneck, [3, 4, 6, 3], groups=1, reduction=16,
					dropout_p=0.2, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes,
					block_drop=True, block_drop_p=0.1,
					drop_type='dropblock', **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet50'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnet101(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnet101', SEResNetBottleneck, [3, 4, 23, 3], groups=1, reduction=16,
					dropout_p=0.2, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet101'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnet152(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnet152', SEResNetBottleneck, [3, 8, 36, 3], groups=1, reduction=16,
					dropout_p=0.2, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet152'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


# Note: This is a model for CIFAR dataset
def se_resnet110(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnet110', SEResNetBottleneck, [12, 12, 12, 12], groups=1, reduction=16,
					dropout_p=None, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet110'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


# Note: This is a model for CIFAR dataset
def se_resnet164(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnet164', SEResNetBottleneck, [18, 18, 18, 18], groups=1, reduction=16,
					dropout_p=None, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnet110'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnext50_32x4d(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnext50_32x4d', SEResNeXtBottleneck, [3, 4, 6, 3], groups=32, reduction=16,
					dropout_p=0.2, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnext50_32x4d'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnext101_32x4d(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnext101_32x4d', SEResNeXtBottleneck, [3, 4, 23, 3],
					groups=32, reduction=16,
					dropout_p=0.2, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	if pretrained is not None:
		settings = pretrained_settings['se_resnext101_32x4d'][pretrained]
		initialize_pretrained_model(model, num_classes, settings)
	return model


def se_resnext101_64x4d(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnext101_64x4d', SEResNeXtBottleneck, [3, 4, 23, 3],
					groups=64, reduction=16,
					dropout_p=0.2, inplanes=64, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes, **kwargs)
	return model


def se_resnext101_64x4d_B(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('se_resnext101_64x4d_B', SEResNeXtBottleneck, [3, 4, 23, 3],
					groups=64, reduction=16,
					dropout_p=0.2, inplanes=128, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes,
					block_drop=False, block_drop_p=0.1,
					drop_type='droppath',
					**kwargs)
	return model


def senet113(num_classes=1000, pretrained='imagenet', **kwargs):
	model = SENet('senet113', SEBottleneck, [3, 8, 23, 3],
					groups=64, reduction=16,
					dropout_p=0.2, inplanes=128, input_3x3=True,
					downsample_kernel_size=1, downsample_padding=0,
					num_classes=num_classes,
					block_drop=False, block_drop_p=0.1,
					drop_type='droppath',
					**kwargs)
	return model
