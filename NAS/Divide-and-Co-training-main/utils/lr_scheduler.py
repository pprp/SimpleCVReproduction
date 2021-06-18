# coding=utf-8
"""
some training utils.
reference:
	https://github.com/ZJULearning/RMI/blob/master/utils/train_utils.py
	https://github.com/zhanghang1989/PyTorch-Encoding

Contact: zhaoshuaimcc@gmail.com
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math


class lr_scheduler(object):
	"""learning rate scheduler
	step mode: 			```lr = init_lr * 0.1 ^ {floor(epoch-1 / lr_step)}```
	cosine mode: 		```lr = init_lr * 0.5 * (1 + cos(iter/maxiter))```
	poly mode: 			```lr = init_lr * (1 - iter/maxiter) ^ 0.9```
	HTD mode:			```lr = init_lr * 0.5 * (1 - tanh(low + (up - low) * iter/maxiter)```
							https://arxiv.org/pdf/1806.01593.pdf
	exponential mode:	```decayed_learning_rate = learning_rate *
							decay_rate ^ (global_step / decay_steps)```

	Args:
		init_lr:			initial learnig rate.
		mode:				['cos', 'poly', 'HTD', 'step', 'exponential'].
		num_epochs:			the number of epochs.
		iters_per_epoch:	iterations per epochs.
		lr_milestones:		lr milestones used for 'step' lr mode
		lr_step:			lr step used for 'step' lr mode.
							It only works when lr_milestones is None.
		lr_step_multiplier: lr multiplier for 'step' lr mode.
		
		multiplier:			lr multiplier for params group in optimizer.
							It only works for {3rd, 4th..} groups
		end_lr:				minimal learning rate.
		
		lower_bound,
		upper_bound:		bound of HTD learning rate strategy.

		decay_factor:		lr decay factor for exponential lr.
		decay_epochs: 		lr decay epochs for exponetital lr.
		staircase:			staircase or not for exponetital lr.
	"""
	def __init__(self, mode='cos',
						init_lr=0.1,
						num_epochs=100,
						iters_per_epoch=300,
						lr_milestones=None,
						lr_step=100,
						lr_step_multiplier=0.1,
						slow_start_epochs=0,
						slow_start_lr=1e-4,
						end_lr=1e-3,
						multiplier=1.0,
						lower_bound=-6.0,
						upper_bound=3.0,
						decay_factor=0.97,
						decay_epochs=0.8,
						staircase=True,
						):

		assert mode in ['cos', 'poly', 'HTD', 'step', 'exponential']
		self.init_lr = init_lr
		self.now_lr = self.init_lr
		self.end_lr = end_lr
		self.mode = mode

		self.num_epochs = num_epochs
		self.iters_per_epoch = iters_per_epoch
		
		self.slow_start_iters = slow_start_epochs * iters_per_epoch
		self.slow_start_lr = slow_start_lr
		self.total_iters = (num_epochs - slow_start_epochs) * iters_per_epoch
		
		self.multiplier = multiplier
		
		# step mode
		self.lr_step = lr_step
		self.lr_milestones = lr_milestones
		self.lr_step_multiplier = lr_step_multiplier
		
		# hyperparameters for HTD
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound

		# exponetital lr
		self.decay_factor = decay_factor
		self.decay_steps = decay_epochs * iters_per_epoch
		self.staircase = staircase

		# log info
		print("INFO:PyTorch: Using {} learning rate scheduler with"
				" warm-up epochs of {}!".format(self.mode, slow_start_epochs))

	def __call__(self, optimizer, i, epoch):
		"""call method"""
		T = epoch * self.iters_per_epoch + i

		if self.slow_start_iters > 0 and T <= self.slow_start_iters:
			# slow start strategy -- warm up
			# see 	https://arxiv.org/pdf/1812.01187.pdf
			# 	Bag of Tricks for Image Classification with Convolutional Neural Networks
			# for details.
			lr = (1.0 * T / self.slow_start_iters) * (self.init_lr - self.slow_start_lr)
			lr = min(lr + self.slow_start_lr, self.init_lr)
		
		elif self.mode == 'cos':
			T = T - self.slow_start_iters
			lr = 0.5 * self.init_lr * (1.0 + math.cos(1.0 * T / self.total_iters * math.pi))
		
		elif self.mode == 'poly':
			T = T - self.slow_start_iters
			lr = self.init_lr * pow(1.0 - 1.0 * T / self.total_iters, 0.9)
			
		elif self.mode == 'HTD':
			"""
			Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification.
			https://arxiv.org/pdf/1806.01593.pdf
			"""
			T = T - self.slow_start_iters
			ratio = 1.0 * T / self.total_iters
			lr = 0.5 * self.init_lr * (1.0 - math.tanh(
							self.lower_bound + (self.upper_bound - self.lower_bound) * ratio))
			
		elif self.mode == 'step':
			T = T - self.slow_start_iters

			if self.lr_milestones is None:
				lr = self.init_lr * (self.lr_step_multiplier ** (epoch // self.lr_step))
			else:
				j = 0
				for mile in self.lr_milestones:
					if epoch < mile:
						continue
					else:
						j += 1
				lr = self.init_lr * (self.lr_step_multiplier ** j)
		
		elif self.mode == 'exponential':
			T = T - self.slow_start_iters

			if self.staircase:
				power = 1.0 * math.floor(T / self.decay_steps)
			else:
				power = 1.0 * T / self.decay_steps

			lr = self.init_lr * (self.decay_factor ** power)
		
		else:
			raise NotImplementedError

		lr = max(lr, self.end_lr)
		self.now_lr = lr

		# adjust learning rate
		self._adjust_learning_rate(optimizer, lr)

	def _adjust_learning_rate(self, optimizer, lr):
		"""adjust the leaning rate"""
		if len(optimizer.param_groups) == 1:
			optimizer.param_groups[0]['lr'] = lr
		else:
			# BE CAREFUL HERE!!!
			# 0 -- the backbone conv weights with weight decay
			# 1 -- the bn params and bias of backbone without weight decay
			# 2 -- the weights of other layers with weight decay
			# 3 -- the bn params and bias of other layers without weigth decay
			optimizer.param_groups[0]['lr'] = lr
			optimizer.param_groups[1]['lr'] = lr
			for i in range(2, len(optimizer.param_groups)):
				optimizer.param_groups[i]['lr'] = lr * self.multiplier


def scale_lr_and_momentum(args):
	"""
	Scale hyperparameters given the adjusted batch_size from input
	hyperparameters and batch size

	Arguements:
		args: holds the script arguments
	"""
	print('=> adjusting learning rate and momentum. '
			'Original lr: {args.lr}, Original momentum: {args.momentum}')
	if 'cifar' in args.dataset:
		std_b_size = 128
	elif 'imagenet' in args.dataset:
		std_b_size = 256
	else:
		raise NotImplementedError

	old_momentum = args.momentum
	args.momentum = old_momentum ** (args.batch_size / std_b_size)
	# args.lr = args.lr * (args.batch_size / std_b_size *
	#                     (1 - args.momentum) / (1 - old_momentum))
	#
	args.lr = args.lr * (args.batch_size / std_b_size)
	print(f'lr adjusted to: {args.lr}, momentum adjusted to: {args.momentum}')

	return args


def get_parameter_groups(model, norm_weight_decay=0):
	"""
	Separate model parameters from scale and bias parameters following norm if
	training imagenet
	"""
	model_params = []
	norm_params = []

	for name, p in model.named_parameters():
		if p.requires_grad:
			# if 'fc' not in name and ('norm' in name or 'bias' in name):
			if 'norm' in name or 'bias' in name:
				norm_params += [p]
			else:
				model_params += [p]

	return [{'params': model_params},
			{'params': norm_params,
				'weight_decay': norm_weight_decay}]
