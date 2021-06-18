# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import json
import math
#import argparse

import torch


def add_parser_params(parser):
	"""add argument to the parser"""

	parser.add_argument('--data', type=str, metavar='DIR', default='./data_dir',
						help='path to dataset')
	
	parser.add_argument('--model_dir', type=str, default='./model_dir',
						help='dir to which model is saved (default: ./model_dir)')

	# model
	parser.add_argument('--arch', type=str, default='resnet50',
							choices=['resnet34', 'resnet50', 'resnet101',
									'resnet152', 'resnet200',
									'resnet110', 'resnet164',
									'wide_resnet16_8', 'wide_resnet16_12',
									'wide_resnet28_10', 'wide_resnet40_10',
									'wide_resnet52_8',
									'wide_resnet50_2', 'wide_resnet50_3', 'wide_resnet101_2',
									'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
									'resnext29_8x64d', 'resnext29_16x64d',
									'se_resnet110', 'se_resnet164',
									'se_resnet50', 'se_resnet50_B',
									'se_resnet101', 'se_resnet152',
									'se_resnext101_64x4d',
									'se_resnext101_64x4d_B',
									'senet154', 'senet113',
									'shake_resnet26_2x96d',
									'densenet190',
									'pyramidnet164', 'pyramidnet272',
									'efficientnetb0', 'efficientnetb1',
									'efficientnetb2', 'efficientnetb3',
									'efficientnetb4', 'efficientnetb5',
									'efficientnetb6', 'efficientnetb7',
									'efficientnetb8', 'efficientnetl2',
									'resnest101', 'resnest200', 'resnest269',
									'resnexst50_4x16d', 'resnexst50_8x16d',
									'resnexst50_4x32d',
									'resnexst101_8x32d'],
							help='The name of the neural architecture (default: resnet50)')

	parser.add_argument('--norm_mode', type=str, default='batch',
							choices=['batch', 'group', 'layer', 'instance', 'none'],
							help='The style of the batchnormalization (default: batch)')

	parser.add_argument('--is_syncbn', default=0, type=int,
							help='use nn.SyncBatchNorm or not')

	parser.add_argument('--workers', default=8, type=int, metavar='N',
						help='number of data loading workers (default: 8)')
	
	parser.add_argument('--epochs', default=300, type=int, metavar='N',
						help='number of total epochs to run (default: 300)')
	
	parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
						help='manual epoch number (useful on restarts, default: 0)')

	parser.add_argument('--eval_per_epoch', default=1, type=int,
						help='run evaluation per eval_per_epoch')

	# data augmentation
	parser.add_argument('--batch_size', default=128, type=int, metavar='N',
						help='mini-batch size (default: 128), this is the total '
						'batch size of all GPUs on the current node when '
						'using Data Parallel or Distributed Data Parallel')

	parser.add_argument('--eval_batch_size', default=100, type=int, metavar='N',
						help='mini-batch size (default: 100), this is will not be divided by'
						'the number of gpus.')

	parser.add_argument('--crop_size', default=32, type=int, metavar='N',
						help='crop size (default: 32)')

	parser.add_argument('--output_stride', default=8, type=int,
							help='output_stride = (resolution of input) / (resolution of output)'
							'(before global pooling layer)')

	parser.add_argument('--padding', default=4, type=int, metavar='N',
						help='padding size (default: 4)')
	# learning rate
	parser.add_argument('--lr_mode', type=str,
							choices=['cos', 'step', 'poly', 'HTD', 'exponential'],
							default='cos',
							help='strategy of the learning rate')

	parser.add_argument('--lr', '--learning_rate', default=0.1, type=float,
						metavar='LR', help='initial learning rate (default: 0.1)',
						dest='lr')

	parser.add_argument('--optimizer', type=str, default='SGD',
							choices=['SGD', 'AdamW', 'RMSprop', 'RMSpropTF'],
							help='The optimizer.')

	parser.add_argument('--lr_milestones', nargs='+', type=int,
							default=[100, 150],
							help='epochs at which we take a learning-rate step '
							'(default: [100, 150])')
	
	parser.add_argument('--lr_step_multiplier', default=0.1, type=float, metavar='M',
						help='lr multiplier at lr_milestones (default: 0.1)')

	parser.add_argument('--lr_multiplier', type=float, default=1.0,
						help='Learning rate multiplier for the unpretrained model.')

	parser.add_argument('--slow_start_lr', type=float, default=5e-3,
						help='Learning rate employed during slow start.')

	parser.add_argument('--end_lr', type=float, default=1e-4,
						help='The ending learning rate.')

	parser.add_argument('--slow_start_epochs', type=int, default=10,
						help='Training model with small learning rate for few epochs.')

	# parameters of the optimizer
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
						help='optimizer momentum (default: 0.9)')

	parser.add_argument('--is_nesterov', default=1, type=int,
							help='using Nesterov accelerated gradient or not')

	parser.add_argument('--print_freq', default=20, type=int,
						metavar='N', help='print frequency (default: 20)')
	
	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')
	
	parser.add_argument('--evaluate', dest='evaluate', action='store_true',
						help='evaluate model on validation set')
	
	parser.add_argument('--pretrained', dest='pretrained', action='store_true',
						help='use pre-trained model')

	parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')

	parser.add_argument('--gn_num_groups', default=8, type=int,
						help='number of groups in group norm if using group norm '
								'as normalization method (default: 8)')
	# datatset
	parser.add_argument('--dataset', type=str, default='cifar10',
							choices=['cifar10', 'cifar100', 'imagenet', 'svhn'],
							help='dataset name (default: pascal)')
	
	parser.add_argument('--is_cutout', default=1, type=int,
							help='using cutout ot not when training')

	parser.add_argument('--erase_p', default=0.5, type=float,
							help='the probability of random earasing (cutout)')

	parser.add_argument('--is_autoaugment', default=0, type=int,
							help='using auto augmentation not when training')

	parser.add_argument('--randaa', type=str, default=None, metavar='NAME',
							help='Use AutoAugment policy. "v0" or "original". (default: None)'
							'This will disable autoaugment'),

	parser.add_argument('--num_classes', default=10, type=int, metavar='N',
							help='The number of classes.')

	parser.add_argument('--is_label_smoothing', default=0, type=int,
							help='using label smoothing or not')
	# mix up
	parser.add_argument('--mix_alpha', default=0.2, type=float,
							help='mixup interpolation coefficient (default: 0.2)')

	parser.add_argument('--is_mixup', default=1, type=int,
							help='using mixup or not')

	# process info
	parser.add_argument('--proc_name', type=str, default='splitnet',
							help='The name of the process.')

	# distributed training setting
	parser.add_argument('--gpu', default=None, type=int,
							help='GPU id to use.')

	# parser.add_argument('--is_distributed', type=bool, default=False,
	# 						help='whether use distributed training or not.')

	parser.add_argument('--no_cuda', action='store_true', default=False,
							help='disables CUDA training')

	parser.add_argument('--gpu_ids', type=str, default='0',
							help='use which gpus to train, must be a comma-separated list of integers only (default=0)')
	
	parser.add_argument('--dist_backend', type=str, default='nccl',
							choices=['nccl', 'gloo'],
							help='Name of the backend to use.')

	parser.add_argument('--world_size', default=1, type=int,
							help='number of nodes for distributed training')

	parser.add_argument('--rank', default=0, type=int,
							help='node rank for distributed training')
	
	parser.add_argument('--dist_url', type=str, default='env://',
							help='specifying how to initialize the package.')

	parser.add_argument('--multiprocessing_distributed', action='store_true',
							help='Use multi-processing distributed training to launch '
								'N processes per node, which has N GPUs. This is the '
								'fastest way to use PyTorch for either single node or '
								'multi node data parallel training')

	# split factor
	parser.add_argument('--split_factor', default=1, type=int,
							help='split one big network into split_factor small networks')

	parser.add_argument('--is_identical_init', default=0, type=int,
							help='initialize the small networks identically or not')

	parser.add_argument('--is_diff_data_train', default=1, type=int,
							help='using different data augmentation for different networks or not when training')

	#parser.add_argument('--is_diff_flip', default=0, type=int,
	#						help='using different fliped image for small networks')

	# ensembled loss
	parser.add_argument('--is_ensembled_loss', default=0, type=int,
							help='calculate loss between ensembled outputs and ground truth'
							'This does not work in practice, corresponding code is removed,'
							'so it is useless.')

	parser.add_argument('--ensembled_loss_weight', default=0.5, type=float,
							help='the weight factor of the ensembled loss (default: 0.5)')

	parser.add_argument('--is_ensembled_after_softmax', default=0, type=int,
							help='whether ensemble the output after softmax')

	parser.add_argument('--is_max_ensemble', default=0, type=int,
							help='use max ensemble rather than simple averaging')

	# co-training loss
	parser.add_argument('--is_cot_loss', default=1, type=int,
							help='calculate co-training loss between outputs of small networks or not')

	parser.add_argument('--cot_loss_choose', type=str, default='js_divergence',
							choices=['kl_seperate', 'kl_mean', 'mse_seperate', 'smooth_l1_seperate',
							'mse_mean', 'smooth_l1_mean', 'js_divergence'],
							help='loss type of co-training loss (default: js_divergence)')

	parser.add_argument('--cot_weight', default=0.5, type=float,
							help='the weight factor of the co-training loss (default: 0.5)')

	parser.add_argument('--is_cot_weight_warm_up', default=1, type=int,
							help='For cot_weight, use warm-up or not')

	parser.add_argument('--cot_weight_warm_up_epochs', default=40, type=int,
							help='The warm up epoch for cot_weight')

	#parser.add_argument('--kl_temperature', default=2.0, type=float,
	#						help='the temperature of the KL loss')

	parser.add_argument('--is_linear_lr', default=0, type=int,
							help='using linear scaling lr with batch_size strategy or not')

	parser.add_argument('--is_summary', default=0, type=int,
							help='only get the Params and FLOPs of the model.')

	parser.add_argument('--is_train_sep', default=0, type=int,
							help='Train small models seperately.')

	# setting about the weight decay
	parser.add_argument('--weight_decay', default=1e-4, type=float,
						metavar='W', help='weight decay (default: 1e-4)',
						dest='weight_decay')

	parser.add_argument('--is_wd_test', default=0, type=int,
						help='test mode for weight_decay hyperparameter')

	parser.add_argument('--is_div_wd', default=1, type=int,
							help='divide the weight_decay or not.')

	parser.add_argument('--is_wd_all', default=0, type=int,
							help='apply weight to all learnable in the model, otherwise, only weights parameters.')

	parser.add_argument('--div_wd_den', default=20.0, type=float,
							help='the denominator when dividing the weight_decay.')

	parser.add_argument('--max_ckpt_nums', default=5, type=int,
							help='maximum number of ckpts.')

	# AMP training
	parser.add_argument('--is_apex_amp', default=0, type=int,
							help='Using NVIDIA APEX Automatic Mixed Precision (AMP)')

	parser.add_argument('--amp_opt_level', type=str, default='O1',
							help='optimization level of apex amp.')

	parser.add_argument('--is_amp', default=1, type=int,
							help='Using PyTorch Automatic Mixed Precision (AMP)')

	# gradient accumulate
	parser.add_argument('--iters_to_accumulate', default=1, type=int,
							help="Gradient accumulation adds gradients "
							"over an effective batch of size batch_per_iter * iters_to_accumulate")

	# (memory efficient) densenet
	parser.add_argument('--is_efficient_densenet', default=0, type=int,
							help='Whether use efficient densenet or not.')

	parser.add_argument('--is_official_densenet', default=1, type=int,
							help='Whether use official densenet implementation or not.')
	
	parser.add_argument('--densenet_p_shakedrop', default=0.0, type=float,
							help='final shake drop probability of shake drop layers in densenet.')

	# setting of efficientnet
	parser.add_argument('--is_efficientnet_user_crop', default=0, type=int,
							help='To save memory, one can use small crop size.')

	parser.add_argument('--is_lukemelas_efficientnet', default=0, type=int,
							help='If True, use the implementation of '
								'https://github.com/lukemelas/EfficientNet-PyTorch.')
	
	parser.add_argument('--is_memory_efficient_swish', default=1, type=int,
							help='Whether use memory-efficient Swish activation or not')
	
	parser.add_argument('--decay_factor', default=0.97, type=float,
							help='decay factor of exponetital lr')

	parser.add_argument('--decay_epochs', default=0.8, type=float,
							help='decay epochs of exponetital lr')

	# multigpu test
	parser.add_argument('--is_test_on_multigpus', default=1, type=int,
							help='Whether test with multigpus or not.')

	parser.add_argument('--is_test_with_multistreams', default=0, type=int,
							help='Whether test with multi cuda streams or not.')

	parser.add_argument('--pretrained_dir', type=str, default=None)

	# parse
	args = parser.parse_args()
	# args, unparsed = parser.parse_known_args()

	# check arguments
	assert not args.is_identical_init
	assert args.norm_mode == 'batch'

	# number of classes
	num_classes_dict = {
						'cifar10': 10,
						'cifar100': 100,
						'imagenet': 1000,
						'svhn': 10,
					}
	args.num_classes = num_classes_dict[args.dataset]

	# check settings for certain datasets
	if args.dataset in ['cifar10', 'cifar100']:
		assert args.crop_size == 32
		args.is_label_smoothing = False

	elif args.dataset == 'imagenet':
		args.end_lr = 1e-5
		args.is_label_smoothing = True
		if not args.is_label_smoothing:
			print("Warning: The default settings on ImageNet use label_smoothing while you "
				"set it to False.")
		assert args.crop_size in [224, 240, 256, 299, 300, 320, 331,
									380, 416, 450, 456, 528, 600, 672, 800]
		assert args.epochs in [90, 100, 120, 270, 350]
	
	elif args.dataset == 'svhn':
		assert args.crop_size == 32
		args.end_lr = 1e-5

		""" lr srategy 1, following RandAugment """
		args.slow_start_epochs = -1
		args.slow_start_lr = 5e-4
		args.lr = 5e-3
		args.weight_decay = 1e-3

		""" lr strategy 2, following FastAugment
		args.epochs = 200
		args.slow_start_epochs = 5
		args.slow_start_lr = 5e-4
		args.lr = 0.01
		args.weight_decay = 5e-4
		"""
	else:
		raise NotImplementedError

	# settings about training epochs and learning rate
	if args.epochs in [90, 100, 120, 270, 350]:
		args.slow_start_epochs = 5
		args.eval_per_epoch = 1
		args.lr_milestones = [30, 60, 90]

	elif args.epochs in [160, 200]:
		# **For svhn dataset**
		args.lr_milestones = [80, 120]
	
	elif args.epochs == 300:
		args.slow_start_epochs = 20
		args.lr_milestones = [100, 225]

		# The two networks seem unstable during training in practice.
		if args.arch in ['resnext29_16x64d', 'se_resnet164']:
			args.slow_start_epochs = 30
			if args.arch == 'resnext29_16x64d':
				args.lr = args.lr / 2.0
	
	elif args.epochs == 1800:
		if args.arch == 'shake_resnet26_2x96d':
			# Xavier Gastaldi, Shake-Shake regularization, 2017.
			# The paper said lr=0.2 and no warm-up in Sec2.1.
			args.lr = 0.2
			args.slow_start_epochs = -1
		elif 'pyramidnet' in args.arch or 'densenet' in args.arch:
			# args.lr = 0.1
			args.slow_start_epochs = -1
		else:
			args.slow_start_epochs = 60
			
		args.cot_weight_warm_up_epochs = args.cot_weight_warm_up_epochs * 2

	else:
		raise NotImplementedError("The epoch number is illegal")

	if not args.is_wd_test:
		# weight decay for CIFAR
		if args.dataset in ['cifar10', 'cifar100']:
			if args.arch in ['resnet110', 'resnet164',
								'se_resnet110', 'se_resnet164',
								'wide_resnet16_8',
								'efficientnetb0', 'efficientnetb1', 'efficientnetb2',
								'shake_resnet26_2x96d',
								'pyramidnet164', 'pyramidnet272',
								'densenet190'
								]:
				args.weight_decay = 1e-4

			elif args.arch in ['efficientnetb3',
								'resnext29_8x64d', 'resnext29_16x64d',
								'wide_resnet16_12', 'wide_resnet28_10',
								'wide_resnet40_10', 'wide_resnet52_8']:
				# a larger weight decay for large model
				args.weight_decay = 5e-4
			else:
				raise NotImplementedError

		elif args.dataset == 'svhn':
			pass

		elif args.dataset == 'imagenet':
			if args.arch in ['resnet34', 'resnet50']:
				args.weight_decay = 5e-5

			elif args.arch in ['resnet101', 'resnet152', 'resnet200',
								'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_64x4d',
								'se_resnet50', 'se_resnet50_B',
								'se_resnext101_64x4d', 'se_resnext101_64x4d_B',
								'se_resnet152', 'senet154', 'senet113',
								'wide_resnet50_2', 'wide_resnet50_3', 'wide_resnet101_2',
								'resnest101', 'resnest200', 'resnest269',
								'resnexst50_32x4d', 'resnexst50_16x8d',
								'resnexst50_4x16d', 'resnexst50_8x16d',
								'resnexst50_4x32d', 'resnexst101_8x32d']:
				# For WRN, original weight decay is 5e-4
				# https://github.com/szagoruyko/wide-residual-networks
				# https://github.com/szagoruyko/wide-residual-networks/blob/master/pretrained/README.md
				args.weight_decay = 1e-4
				
				if args.arch in ['se_resnext101_64x4d_B', 'senet113', 'resnexst101_8x32d']:
					# args.is_mixup = False if args.arch != 'resnexst101_8x32d' else True
					args.is_autoaugment = False
					args.randaa = 'rand-m9-mstd0.5'
					args.lr = args.lr * (1.0 * args.batch_size / 256)
				
				if args.arch in ['resnexst50_32x4d', 'se_resnet50',
									'resnexst50_4x16d', 'resnexst50_8x16d', 'resnexst50_4x32d']:
					# args.is_mixup = False if 'resnexst' not in args.arch else True
					args.is_autoaugment = False
					args.randaa = 'rand-m6-mstd0.5'

			elif 'efficientnet' in args.arch:
				# Reference:
				# Mingxing Tan, Quoc V. Le.
				# EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
				# https://github.com/tensorflow/tpu/blob/models/official/efficientnet/main.py
				# https://github.com/rwightman/pytorch-image-models/blob/fcb6258877/docs/training_hparam_examples.md

				# args.lr_mode = 'exponential'
				# 0.016, 0.032, 0.064, 0.256
				args.lr = 0.256

				# For efficientnet, 2.4 epochs on ImageNet with total epochs 350 and batch 2048.
				# I tested decay epochs 2.4 with batch size 128, it is a too large wd for RMSprop
				# and leads to inferior performance (efficientb1, 71.14%).
				# To match the decay times, 120 epochs training uses a decay epoch 0.8 with batch 256.
				# To match the decay steps, decay steps should be equal to 1500
				# (decay_epochs 0.3, not tested).
				# This will lead to a too small lr in the late stage of training.
				# So we match the decay times.
				args.decay_epochs = 0.8
				args.weight_decay = 1e-5
				args.slow_start_epochs = 5
				args.slow_start_lr = 1e-6
				# In experiments, we found mixup will lead to inferior performance on ImageNet.
				args.is_mixup = False
				# efficientnet config: (random erasing probability, rand augment config, crop size)
				effi_config_dict = {
								'efficientnetb0': (0.2, 'rand-m4-mstd0.5', 224),
								'efficientnetb1': (0.2, 'rand-m4-mstd0.5', 240),
								'efficientnetb2': (0.3, 'rand-m5-mstd0.5', 260),
								'efficientnetb3': (0.3, 'rand-m6-mstd0.5', 300),
								'efficientnetb4': (0.4, 'rand-m6-mstd0.5', 380),
								'efficientnetb5': (0.4, 'rand-m7-mstd0.5', 456),
								'efficientnetb6': (0.5, 'rand-m8-mstd0.5', 528),
								'efficientnetb7': (0.5, 'rand-m9-mstd0.5', 600),
								'efficientnetb8': (0.5, 'rand-m9-mstd0.5', 672),
								'efficientnetl2': (0.5, 'rand-m9-mstd0.5', 800),
							}
				args.erase_p = effi_config_dict[args.arch][0]
				args.is_autoaugment = False
				args.randaa = effi_config_dict[args.arch][1]
				# change the crop size for efficientnet, this also means your user setting is useless
				if not args.is_efficientnet_user_crop:
					args.crop_size = effi_config_dict[args.arch][2]

			else:
				raise NotImplementedError

		else:
			raise NotImplementedError
	
	print("INFO:PyTorch: The crop size for {} is {}.".format(args.arch, args.crop_size))
	print("INFO:PyTorch: set the value of weight decay as: {}.".format(args.weight_decay))
	
	# if args.is_div_wd and args.split_factor > 1 and args.dataset != 'cifar100':
	if args.is_div_wd and args.split_factor > 1:
		# we assume the magnitude of weight decay is a linear function w.r.t. to the model capacity
		# args.weight_decay = args.weight_decay * max(1 - args.split_factor / args.div_wd_den, 1.0 / args.split_factor)
		# args.weight_decay = args.weight_decay / args.split_factor
		
		# non-linear function
		args.weight_decay = args.weight_decay * math.exp(1.0 / args.split_factor - 1.0)
		print("INFO:PyTorch: divide the value of weight decay."
				" The new weight_decay is : {}".format(args.weight_decay))
		print("NOTE: You are applying weight decay exponential dividing strategy here."
			"This will not always gain improvements in performance."
			"No dividing or dividing in other manners may produce better performance."
			"See Appendix <Weight decay matters> of the paper"
			"<SplitNet: Divide and Co-training> for details."
			"Now, the best way is trial-and-error.")

	# use gpu or not
	args.cuda = not args.no_cuda and torch.cuda.is_available()
	if args.cuda:
		try:
			args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
			args.gpu_ids = [i for i in range(0, len(args.gpu_ids))]
			args.num_gpus = len(args.gpu_ids)
		except ValueError:
			raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
	
	# dist_url for pytorch distributed learning
	if args.dist_url == "env://" and args.world_size == -1:
		args.world_size = int(os.environ["WORLD_SIZE"])

	# If only one model, disable some setting about co-training
	if args.split_factor == 1:
		args.is_train_sep = False
		args.is_cot_loss = False
		args.is_ensembled_loss = False
		args.is_ensembled_after_softmax = False

	if args.is_train_sep:
		args.is_diff_data_train = 0

	if args.is_apex_amp:
		print("INFO:PyTorch: Using APEX AMP training.")
		raise ValueError("is_apex_amp should not be TRUE as APEX AMP is no longer supported."
					"Use torch.cuda.amp() instead.")
	if args.is_amp:
		print("INFO:PyTorch: Using PyTorch AMP training.")

	if args.is_max_ensemble:
		print("INFO:PyTorch: Using max ensemble manner.")

	# save the hyper-parameters
	if not args.is_summary and not args.evaluate:
		save_hp_to_json(args)
	# sys.exit(0)
	return args


def save_hp_to_json(args):
	"""Save hyperparameters to a json file
	"""
	if not args.evaluate:
		filename = os.path.join(args.model_dir, 'hparams_train.json')
	else:
		filename = os.path.join(args.model_dir, 'hparams_eval.json')
	# hparams = FLAGS.flag_values_dict()
	hparams = vars(args)
	with open(filename, 'w') as f:
		json.dump(hparams, f, indent=4, sort_keys=True)


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser(description="PyTorch SplitNet Training")
	args = add_parser_params(parser)
	print(args)
