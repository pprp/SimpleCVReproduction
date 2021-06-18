""" RMSProp modified to behave like Tensorflow impl
Originally cut & paste from PyTorch RMSProp
https://github.com/pytorch/pytorch/blob/063946d2b3f3f1e953a2a3b54e0b34f1393de295/torch/optim/rmsprop.py
Licensed under BSD-Clause 3 (ish), https://github.com/pytorch/pytorch/blob/master/LICENSE
Modifications Copyright 2020 Ross Wightman

Commits:
07/10/2020
Change some old add_() apis.
Change decayed weight to (lr x wd).

Contact: zhaoshuaimcc@gmail.com
"""

import torch
from torch.optim import Optimizer


class RMSpropTF(Optimizer):
	r"""Implements RMSprop algorithm (TensorFlow style epsilon).
	
	Noteworthy changes include:
	1. Epsilon applied inside square-root
	2. square_avg initialized to ones
	3. LR scaling of update accumulated in momentum buffer

	Proposed by G. Hinton in his
	`course <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_.

	The centered version first appears in `Generating Sequences
	With Recurrent Neural Networks <https://arxiv.org/pdf/1308.0850v5.pdf>`_.

	The implementation here takes the square root of the gradient average before
	adding epsilon (note that TensorFlow interchanges these two operations). The effective
	learning rate is thus :math:`\alpha/(\sqrt{v} + \epsilon)` where :math:`\alpha`
	is the scheduled learning rate and :math:`v` is the weighted moving average
	of the squared gradient.

	Arguments:
		params (iterable): iterable of parameters to optimize or dicts defining
			parameter groups
		lr (float, optional): learning rate (default: 1e-2)
		momentum (float, optional): momentum factor (default: 0)
		alpha (float, optional): smoothing constant (default: 0.99)
		eps (float, optional): term added to the denominator to improve
			numerical stability (default: 1e-8)
		centered (bool, optional) : if ``True``, compute the centered RMSProp,
			the gradient is normalized by an estimation of its variance
		weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

	"""

	def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
					momentum=0, centered=False,
					decoupled_decay=False, lr_in_momentum=True):
		if not 0.0 <= lr:
			raise ValueError("Invalid learning rate: {}".format(lr))
		if not 0.0 <= eps:
			raise ValueError("Invalid epsilon value: {}".format(eps))
		if not 0.0 <= momentum:
			raise ValueError("Invalid momentum value: {}".format(momentum))
		if not 0.0 <= weight_decay:
			raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
		if not 0.0 <= alpha:
			raise ValueError("Invalid alpha value: {}".format(alpha))

		defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
							centered=centered, weight_decay=weight_decay,
							decoupled_decay=decoupled_decay, lr_in_momentum=lr_in_momentum)
		super(RMSpropTF, self).__init__(params, defaults)

	def __setstate__(self, state):
		super(RMSpropTF, self).__setstate__(state)
		for group in self.param_groups:
			group.setdefault('momentum', 0)
			group.setdefault('centered', False)

	@torch.no_grad()
	def step(self, closure=None):
		"""Performs a single optimization step.

		Arguments:
			closure (callable, optional): A closure that reevaluates the model
				and returns the loss.
		"""
		loss = None
		if closure is not None:
			with torch.enable_grad():
				loss = closure()

		for group in self.param_groups:
			for p in group['params']:
				if p.grad is None:
					continue
				grad = p.grad
				if grad.is_sparse:
					raise RuntimeError('RMSprop does not support sparse gradients')
				state = self.state[p]

				# State initialization
				if len(state) == 0:
					state['step'] = 0
					# PyTorch inits to zero
					# state['square_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
					state['square_avg'] = torch.ones_like(p, memory_format=torch.preserve_format)
					if group['momentum'] > 0:
						state['momentum_buffer'] = torch.zeros_like(p, memory_format=torch.preserve_format)
					if group['centered']:
						state['grad_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

				square_avg = state['square_avg']
				# alpha = group['alpha']
				one_minus_alpha = 1. - group['alpha']

				state['step'] += 1

				if group['weight_decay'] != 0:
					if 'decoupled_decay' in group and group['decoupled_decay']:
						# perform step weight decay
						p.mul_(1.0 - group['lr'] * group['weight_decay'])
						# p.add_(p.data, alpha=-group['weight_decay'])
					else:
						grad = grad.add(p, alpha=group['weight_decay'])

				# Tensorflow order of ops for updating squared avg
				# old api
				# square_avg.add_(one_minus_alpha, grad.pow(2) - square_avg)
				# new api
				square_avg.add_(grad.pow(2) - square_avg, alpha=one_minus_alpha)
				# Pytorch origin
				# square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

				if group['centered']:
					grad_avg = state['grad_avg']
					# PyTorch original
					# grad_avg.mul_(alpha).add_(grad, alpha=one_minus_alpha)
					grad_avg.add_(grad - grad_avg, alpha=one_minus_alpha)
					# PyTorch original
					# avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_().add_(group['eps'])
					# eps moved in sqrt
					avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).add(group['eps']).sqrt_()
				else:
					# PyTorch original
					# avg = square_avg.sqrt().add_(group['eps'])
					# eps moved in sqrt
					avg = square_avg.add(group['eps']).sqrt_()

				if group['momentum'] > 0:
					buf = state['momentum_buffer']
					if 'lr_in_momentum' in group and group['lr_in_momentum']:
						# Tensorflow accumulates the LR scaling in the momentum buffer
						buf.mul_(group['momentum']).addcdiv_(grad, avg, value=group['lr'])
						p.add_(-buf)
					else:
						# PyTorch scales the param update by LR
						buf.mul_(group['momentum']).addcdiv_(grad, avg)
						p.add_(buf, alpha=-group['lr'])
				else:
					p.addcdiv_(grad, avg, value=-group['lr'])

		return loss
