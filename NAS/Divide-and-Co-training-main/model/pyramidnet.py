"""
PyramidNet with shakedrop

Ref:
[1] https://github.com/kakaobrain/fast-autoaugment/blob/master/FastAutoAugment/networks/pyramidnet.py

[2] https://github.com/ecs-vlc/FMix

[3] Dongyoon Han, Jiwhan Kim, Junmo Kim. Deep Pyramidal Residual Networks, 2017.
	https://arxiv.org/abs/2002.12047.

[4] Yoshihiro Yamada, Masakazu Iwamura, Takuya Akiba, Koichi Kise.
	ShakeDrop Regularization for Deep Residual Learning, 2018.
	https://arxiv.org/abs/1802.02375.
"""

import math

import torch
import torch.nn as nn
from torch.autograd import Variable

__all__ = ['pyramidnet272', 'pyramidnet164']

_inplace_flag = True


"""
splitnet/model/pyramidnet.py:175:
UserWarning: Output 0 of ShakeDropFunctionBackward is a view and is being modified inplace.
This view was created inside a custom Function (or because an input was returned as-is)
and the autograd logic to handle view+inplace would override
the custom backward associated with the custom Function,
leading to incorrect gradients.
This behavior is deprecated and will be forbidden starting version 1.6.
You can remove this warning by cloning the output of the custom Function.
(Triggered internally at  /opt/conda/conda-bld/pytorch_1595629403081/work/torch/csrc/autograd/variable.cpp:464.)
"""


class ShakeDropFunction(torch.autograd.Function):

	@staticmethod
	def forward(ctx, x, training=True, p_drop=0.5, alpha_range=[-1, 1]):
		if training:
			gate = torch.cuda.FloatTensor([0]).bernoulli_(1 - p_drop)
			ctx.save_for_backward(gate)
			if gate.item() == 0:
				alpha = torch.cuda.FloatTensor(x.size(0)).uniform_(*alpha_range)
				alpha = alpha.view(alpha.size(0), 1, 1, 1).expand_as(x)
				return alpha * x
			else:
				return x
		else:
			return (1 - p_drop) * x

	@staticmethod
	def backward(ctx, grad_output):
		gate = ctx.saved_tensors[0]
		if gate.item() == 0:
			beta = torch.cuda.FloatTensor(grad_output.size(0)).uniform_(0, 1)
			beta = beta.view(beta.size(0), 1, 1, 1).expand_as(grad_output)
			beta = Variable(beta)
			return beta * grad_output, None, None, None
		else:
			return grad_output, None, None, None


class ShakeDrop(nn.Module):

	def __init__(self, p_drop=0.5, alpha_range=[-1, 1]):
		super(ShakeDrop, self).__init__()
		self.p_drop = p_drop
		self.alpha_range = alpha_range

	def forward(self, x):
		return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


def conv3x3(in_planes, out_planes, stride=1):
	"""
	3x3 convolution with padding
	"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
	outchannel_ratio = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None, p_shakedrop=1.0):
		super(BasicBlock, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes)
		self.bn3 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=_inplace_flag)
		self.downsample = downsample
		self.stride = stride
		self.shake_drop = ShakeDrop(p_shakedrop)

	def forward(self, x):

		out = self.bn1(x)
		out = self.conv1(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn3(out)

		out = self.shake_drop(out)

		if self.downsample is not None:
			shortcut = self.downsample(x)
			featuremap_size = shortcut.size()[2:4]
		else:
			shortcut = x
			featuremap_size = out.size()[2:4]

		batch_size = out.size()[0]
		residual_channel = out.size()[1]
		shortcut_channel = shortcut.size()[1]

		if residual_channel != shortcut_channel:
			padding = torch.autograd.Variable(
				torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
										featuremap_size[1]).fill_(0))
			out += torch.cat((shortcut, padding), 1)
		else:
			out += shortcut

		return out


class Bottleneck(nn.Module):
	outchannel_ratio = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None, p_shakedrop=1.0):
		super(Bottleneck, self).__init__()
		self.bn1 = nn.BatchNorm2d(inplanes)
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, (planes * 1), kernel_size=3, stride=stride,
								padding=1, bias=False)
		self.bn3 = nn.BatchNorm2d((planes * 1))
		self.conv3 = nn.Conv2d((planes * 1), planes * Bottleneck.outchannel_ratio, kernel_size=1, bias=False)
		self.bn4 = nn.BatchNorm2d(planes * Bottleneck.outchannel_ratio)
		self.relu = nn.ReLU(inplace=_inplace_flag)
		self.downsample = downsample
		self.stride = stride
		self.shake_drop = ShakeDrop(p_shakedrop)

	def forward(self, x):

		out = self.bn1(x)
		out = self.conv1(out)

		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv2(out)

		out = self.bn3(out)
		out = self.relu(out)
		out = self.conv3(out)

		out = self.bn4(out)

		out = self.shake_drop(out)

		if self.downsample is not None:
			shortcut = self.downsample(x)
			featuremap_size = shortcut.size()[2:4]
		else:
			shortcut = x
			featuremap_size = out.size()[2:4]

		batch_size = out.size()[0]
		residual_channel = out.size()[1]
		shortcut_channel = shortcut.size()[1]

		if residual_channel != shortcut_channel:
			padding = torch.autograd.Variable(
				torch.cuda.FloatTensor(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
											featuremap_size[1]).fill_(0))
			# for summary parameters and FLOPs
			#padding = torch.autograd.Variable(
			#			torch.zeros(batch_size, residual_channel - shortcut_channel, featuremap_size[0],
			#							featuremap_size[1]).type_as(shortcut))
			out += torch.cat((shortcut, padding), 1)
		else:
			out += shortcut

		return out


class PyramidNet(nn.Module):
	def __init__(self, dataset='cifar10',
						bottleneck=True,
						depth=272,
						alpha=200,
						num_classes=10,
						split_factor=1):
		super(PyramidNet, self).__init__()
		"""
		inplanes_dict = {'imagenet': {1: 64, 2: 44, 4: 32, 8: 24},
							'cifar10': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
							'cifar100': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
							'svhn': {1: 16, 2: 12, 4: 8, 8: 6, 16: 4},
						}
		"""

		self.dataset = dataset
		if self.dataset in ['cifar10', 'cifar100', 'svhn']:
			# self.inplanes = inplanes_dict[dataset][split_factor]
			self.inplanes = 16

			if bottleneck:
				n = int((depth - 2) / 9)
				block = Bottleneck
			else:
				n = int((depth - 2) / 6)
				block = BasicBlock

			# self.addrate = alpha / (3 * n * 1.0)
			self.addrate = alpha / (3 * n * (split_factor ** 0.5))
			self.final_shake_p = 0.5 / (split_factor ** 0.5)
			print('INFO:PyTorch: PyramidNet: The add rate is {}, '
				'the final shake p is {}'.format(self.addrate, self.final_shake_p))

			self.ps_shakedrop = [1. - (1.0 - (self.final_shake_p / (3 * n)) * (i + 1)) for i in range(3 * n)]

			self.input_featuremap_dim = self.inplanes
			self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=3, stride=1, padding=1, bias=False)
			self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)

			self.featuremap_dim = self.input_featuremap_dim
			self.layer1 = self.pyramidal_make_layer(block, n)
			self.layer2 = self.pyramidal_make_layer(block, n, stride=2)
			self.layer3 = self.pyramidal_make_layer(block, n, stride=2)

			self.final_featuremap_dim = self.input_featuremap_dim
			self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
			self.relu_final = nn.ReLU(inplace=_inplace_flag)
			#self.avgpool = nn.AvgPool2d(8)
			self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
			self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

		elif dataset == 'imagenet':
			raise NotImplementedError

			blocks = {18: BasicBlock, 34: BasicBlock, 50: Bottleneck, 101: Bottleneck, 152: Bottleneck, 200: Bottleneck}
			layers = {18: [2, 2, 2, 2],
						34: [3, 4, 6, 3],
						50: [3, 4, 6, 3],
						101: [3, 4, 23, 3],
						152: [3, 8, 36, 3],
						200: [3, 24, 36, 3]}

			if layers.get(depth) is None:
				if bottleneck is True:
					blocks[depth] = Bottleneck
					temp_cfg = int((depth - 2) / 12)
				else:
					blocks[depth] = BasicBlock
					temp_cfg = int((depth - 2) / 8)

				layers[depth] = [temp_cfg, temp_cfg, temp_cfg, temp_cfg]
				print('=> the layer configuration for each stage is set to', layers[depth])

			self.inplanes = 64
			self.addrate = alpha / (sum(layers[depth]) * 1.0)

			self.input_featuremap_dim = self.inplanes
			self.conv1 = nn.Conv2d(3, self.input_featuremap_dim, kernel_size=7, stride=2, padding=3, bias=False)
			self.bn1 = nn.BatchNorm2d(self.input_featuremap_dim)
			self.relu = nn.ReLU(inplace=_inplace_flag)
			self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

			self.featuremap_dim = self.input_featuremap_dim
			self.layer1 = self.pyramidal_make_layer(blocks[depth], layers[depth][0])
			self.layer2 = self.pyramidal_make_layer(blocks[depth], layers[depth][1], stride=2)
			self.layer3 = self.pyramidal_make_layer(blocks[depth], layers[depth][2], stride=2)
			self.layer4 = self.pyramidal_make_layer(blocks[depth], layers[depth][3], stride=2)

			self.final_featuremap_dim = self.input_featuremap_dim
			self.bn_final = nn.BatchNorm2d(self.final_featuremap_dim)
			self.relu_final = nn.ReLU(inplace=_inplace_flag)
			# self.avgpool = nn.AvgPool2d(7)
			self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
			self.fc = nn.Linear(self.final_featuremap_dim, num_classes)

		else:
			raise NotImplementedError

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

		assert len(self.ps_shakedrop) == 0, self.ps_shakedrop

	def pyramidal_make_layer(self, block, block_depth, stride=1):
		downsample = None
		if stride != 1:  # or self.inplanes != int(round(featuremap_dim_1st)) * block.outchannel_ratio:
			downsample = nn.AvgPool2d((2, 2), stride=(2, 2), ceil_mode=True)

		layers = []
		self.featuremap_dim = self.featuremap_dim + self.addrate
		layers.append(block(self.input_featuremap_dim, int(round(self.featuremap_dim)), stride, downsample,
							p_shakedrop=self.ps_shakedrop.pop(0)))
		for i in range(1, block_depth):
			temp_featuremap_dim = self.featuremap_dim + self.addrate
			layers.append(
				block(int(round(self.featuremap_dim)) * block.outchannel_ratio, int(round(temp_featuremap_dim)), 1,
						p_shakedrop=self.ps_shakedrop.pop(0)))
			self.featuremap_dim = temp_featuremap_dim
		self.input_featuremap_dim = int(round(self.featuremap_dim)) * block.outchannel_ratio

		return nn.Sequential(*layers)

	def forward(self, x):
		if self.dataset == 'cifar10' or self.dataset == 'cifar100':
			x = self.conv1(x)
			x = self.bn1(x)

			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)

			x = self.bn_final(x)
			x = self.relu_final(x)
			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc(x)

		elif self.dataset == 'imagenet':
			x = self.conv1(x)
			x = self.bn1(x)
			x = self.relu(x)
			x = self.maxpool(x)

			x = self.layer1(x)
			x = self.layer2(x)
			x = self.layer3(x)
			x = self.layer4(x)

			x = self.bn_final(x)
			x = self.relu_final(x)
			x = self.avgpool(x)
			x = x.view(x.size(0), -1)
			x = self.fc(x)

		return x


def pyramidnet164(bottleneck=True, **kwargs):
	"""PyramidNet164 for CIFAR and SVHN"""
	return PyramidNet(bottleneck=bottleneck, depth=164, alpha=270, **kwargs)


def pyramidnet272(bottleneck=True, **kwargs):
	"""PyramidNet272 for CIFAR and SVHN"""
	return PyramidNet(bottleneck=bottleneck, depth=272, alpha=200, **kwargs)
