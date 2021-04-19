import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

class DynamicBatchNorm2d(nn.Module):
	SET_RUNNING_STATISTICS = False

	def __init__(self, max_feature_dim):
		super(DynamicBatchNorm2d, self).__init__()

		self.max_feature_dim = max_feature_dim
		self.bn = nn.BatchNorm2d(self.max_feature_dim)

	@staticmethod
	def bn_forward(x, bn: nn.BatchNorm2d, feature_dim):
		if bn.num_features == feature_dim or DynamicBatchNorm2d.SET_RUNNING_STATISTICS:
			return bn(x)
		else:
			exponential_average_factor = 0.0

			if bn.training and bn.track_running_stats:
				if bn.num_batches_tracked is not None:
					bn.num_batches_tracked += 1
					if bn.momentum is None:  # use cumulative moving average
						exponential_average_factor = 1.0 / float(bn.num_batches_tracked)
					else:  # use exponential moving average
						exponential_average_factor = bn.momentum
			return F.batch_norm(
				x, bn.running_mean[:feature_dim], bn.running_var[:feature_dim], bn.weight[:feature_dim],
				bn.bias[:feature_dim], bn.training or not bn.track_running_stats,
				exponential_average_factor, bn.eps,
			)

	def forward(self, x):
		feature_dim = x.size(1)
		y = self.bn_forward(x, self.bn, feature_dim)
		return y



class DynamicConv2d(nn.Module):
	def __init__(self, max_in_channels, max_out_channels, kernel_size=1, stride=1, dilation=1):
		super(DynamicConv2d, self).__init__()

		self.max_in_channels = max_in_channels
		self.max_out_channels = max_out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.dilation = dilation

		self.conv = nn.Conv2d(
			self.max_in_channels, self.max_out_channels, self.kernel_size, stride=self.stride, bias=False,
		)

		self.active_out_channel = self.max_out_channels

	def get_active_filter(self, out_channel, in_channel):
		return self.conv.weight[:out_channel, :in_channel, :, :]

	def forward(self, x, out_channel=None):
		if out_channel is None:
			out_channel = self.active_out_channel
		in_channel = x.size(1)
		filters = self.get_active_filter(out_channel, in_channel).contiguous()

		padding = get_same_padding(self.kernel_size)
		filters = self.conv.weight_standardization(filters) if isinstance(self.conv, MyConv2d) else filters
		y = F.conv2d(x, filters, None, self.stride, padding, self.dilation, 1)
		return y
