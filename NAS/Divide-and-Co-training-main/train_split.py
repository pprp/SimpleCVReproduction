# coding=utf-8
"""
Copyright (c) 2020 -      Shuai Zhao
All rights reserved.
Under Apache-2.0 LICENSE.

Reference:
	[1] https://github.com/pytorch/examples/tree/master/imagenet
	[2] https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets

Contact: zhaoshuaimcc@gmail.com
"""

import os
import sys
import time
import random
import shutil
import argparse
import warnings
import setproctitle

import torch
import torch.cuda.amp as amp
from torch import nn, distributed
from torch.backends import cudnn
from tensorboardX import SummaryWriter

import parser_params
from utils import rmsprop_tf
from utils import metric, lr_scheduler, label_smoothing, norm, prefetch, summary
from model import splitnet
from dataset import factory
from utils.thop import profile, clever_format

# global best accuracy
best_acc1 = 0


def main(args):

	# scale learning rate based on global batch size
	if args.is_linear_lr:
		args = lr_scheduler.scale_lr_and_momentum(args)

	if args.seed is not None:
		random.seed(args.seed)
		torch.manual_seed(args.seed)
		cudnn.deterministic = True
		warnings.warn('You have chosen to seed training. '
						'This will turn on the CUDNN deterministic setting, '
						'which can slow down your training considerably! '
						'You may see unexpected behavior when restarting '
						'from checkpoints.')

	if args.gpu is not None:
		warnings.warn('You have chosen a specific GPU. This will completely '
						'disable data parallelism.')
	
	# If we traing the model seperately, all the number of loops will be one.
	# It is similar as split_factor = 1
	args.loop_factor = 1 if args.is_train_sep else args.split_factor

	# use distributed training or not
	args.is_distributed = args.world_size > 1 or args.multiprocessing_distributed

	ngpus_per_node = torch.cuda.device_count()
	args.ngpus_per_node = ngpus_per_node
	print("INFO:PyTorch: The number of GPUs in this node is {}".format(ngpus_per_node))
	if args.multiprocessing_distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = ngpus_per_node * args.world_size
		# Use torch.multiprocessing.spawn to launch distributed processes: the
		# main_worker process function.
		# spawn will produce the process index for the first arg of main_worker
		torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
	else:
		# Simply call main_worker function
		print("INFO:PyTorch: Set gpu_id = 0 because you only have one visible gpu, otherwise, change the code")
		args.gpu = 0
		main_worker(args.gpu, ngpus_per_node, args)
	# clean processes
	# torch.distributed.destroy_process_group()


def main_worker(gpu, ngpus_per_node, args):
	global best_acc1
	args.gpu = gpu

	if args.is_distributed:
		print("INFO:PyTorch: Initialize process group for distributed training")
		if args.dist_url == "env://" and args.rank == -1:
			args.rank = int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# For multiprocessing distributed training, rank needs to be the
			# global rank among all the processes
			args.rank = args.rank * ngpus_per_node + gpu
		distributed.init_process_group(backend=args.dist_backend,
										init_method=args.dist_url,
										world_size=args.world_size,
										rank=args.rank)

	if args.gpu is not None:
		if not args.evaluate:
			print("INFO:PyTorch: Use GPU: {} for training, the rank of this GPU is {}".format(args.gpu, args.rank))
		else:
			print("INFO:PyTorch: Use GPU: {} for evaluating, the rank of this GPU is {}".format(args.gpu, args.rank))

	# set the name of the process
	setproctitle.setproctitle(args.proc_name + '_rank{}'.format(args.rank))
	if not args.multiprocessing_distributed or \
		(args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
		# define tensorboard summary
		val_writer = SummaryWriter(log_dir=os.path.join(args.model_dir, 'val'))
	
	# define loss function (criterion) and optimizer
	if args.is_label_smoothing:
		criterion = label_smoothing.label_smoothing_CE(reduction='mean')
	else:
		criterion = nn.CrossEntropyLoss()

	# create model
	if args.pretrained:
		model_info = "INFO:PyTorch: using pre-trained model '{}'".format(args.arch)
	else:
		model_info = "INFO:PyTorch: creating model '{}'".format(args.arch)

	print(model_info)
	model = splitnet.SplitNet(args,
								norm_layer=norm.norm(args.norm_mode),
								criterion=criterion)
	
	# print the number of parameters in the model
	print("INFO:PyTorch: The number of parameters in the model is {}".format(metric.get_the_number_of_params(model)))
	if args.is_summary:
		summary_choice = 0
		if summary_choice == 0:
			summary.summary(model,
							torch.rand((1, 3, args.crop_size, args.crop_size)),
							target=torch.ones(1, dtype=torch.long))
		else:
			flops, params = profile(model,
									inputs=(torch.rand((1, 3, args.crop_size, args.crop_size)),
									torch.ones(1, dtype=torch.long),
									'summary'))
			print(clever_format([flops, params], "%.4f"))
		return None

	if args.is_distributed:
		if args.world_size > 1 and args.is_syncbn:
			print("INFO:PyTorch: convert torch.nn.BatchNormND layer in the model to torch.nn.SyncBatchNorm layer")
			# only single gpu per process is currently supported
			model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

		# For multiprocessing distributed, DistributedDataParallel constructor
		# should always set the single device scope, otherwise,
		# DistributedDataParallel will use all available devices.
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# When using a single GPU per process and per
			# DistributedDataParallel, we need to divide the batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
			model = torch.nn.parallel.DistributedDataParallel(model,
																device_ids=[args.gpu],
																find_unused_parameters=True)
		else:
			model.cuda()
			# DistributedDataParallel will divide and allocate batch_size to all
			# available GPUs if device_ids are not set
			model = torch.nn.parallel.DistributedDataParallel(model)
	
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model = model.cuda(args.gpu)
	
	else:
		# DataParallel will divide and allocate batch_size to all available GPUs
		if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
			model.features = torch.nn.DataParallel(model.features)
			model.cuda()
		else:
			model = torch.nn.DataParallel(model).cuda()

	# optimizer
	param_groups = model.parameters() if args.is_wd_all else lr_scheduler.get_parameter_groups(model)
	if args.is_wd_all:
		print("INFO:PyTorch: Applying weight decay to all learnable parameters in the model.")

	if args.optimizer == 'SGD':
		print("INFO:PyTorch: using SGD optimizer.")
		optimizer = torch.optim.SGD(param_groups,
									args.lr,
									momentum=args.momentum,
									weight_decay=args.weight_decay,
									nesterov=True if args.is_nesterov else False
									)
	elif args.optimizer == "AdamW":
		print("INFO:PyTorch: using AdamW optimizer.")
		optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
										betas=(0.9, 0.999),
										eps=1e-4,
										weight_decay=args.weight_decay)

	elif args.optimizer == "RMSprop":
		# See efficientNet at https://github.com/tensorflow/tpu/
		print("INFO:PyTorch: using RMSprop optimizer.")
		optimizer = torch.optim.RMSprop(param_groups, lr=args.lr,
										alpha=0.9,
										weight_decay=args.weight_decay,
										momentum=0.9)

	elif args.optimizer == "RMSpropTF":
		# https://github.com/rwightman/pytorch-image-models/blob/fcb6258877/timm/optim/rmsprop_tf.py
		print("INFO:PyTorch: using RMSpropTF optimizer.")
		optimizer = rmsprop_tf.RMSpropTF(param_groups, lr=args.lr,
											alpha=0.9,
											eps=0.001,
											weight_decay=args.weight_decay,
											momentum=0.9,
											decoupled_decay=False)
	else:
		raise NotImplementedError
	
	# PyTorch AMP loss scaler
	scaler = None if not args.is_amp else amp.GradScaler()

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("INFO:PyTorch: => loading checkpoint '{}'".format(args.resume))
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = 'cuda:{}'.format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)

			args.start_epoch = checkpoint['epoch']
			best_acc1 = checkpoint['best_acc1']
			"""
			if args.gpu is not None:
				# best_acc1 may be from a checkpoint from a different GPU
				best_acc1 = best_acc1.to(args.gpu)
			"""
			model.load_state_dict(checkpoint['state_dict'])
			print("INFO:PyTorch: Loading state_dict of optimizer")
			optimizer.load_state_dict(checkpoint['optimizer'])

			if "scaler" in checkpoint:
				print("INFO:PyTorch: Loading state_dict of AMP loss scaler")
				scaler.load_state_dict(checkpoint['scaler'])

			print("INFO:PyTorch: => loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("INFO:PyTorch: => no checkpoint found at '{}'".format(args.resume))

	# accelarate the training
	torch.backends.cudnn.benchmark = True
	
	# Data loading code
	data_split_factor = args.loop_factor if args.is_diff_data_train else 1
	print("INFO:PyTorch: => The number of views of train data is '{}'".format(data_split_factor))
	train_loader, train_sampler = factory.get_data_loader(args.data,
												split_factor=data_split_factor,
												batch_size=args.batch_size,
												crop_size=args.crop_size,
												dataset=args.dataset,
												split="train",
												is_distributed=args.is_distributed,
												is_autoaugment=args.is_autoaugment,
												randaa=args.randaa,
												is_cutout=args.is_cutout,
												erase_p=args.erase_p,
												num_workers=args.workers)
	val_loader = factory.get_data_loader(args.data,
											batch_size=args.eval_batch_size,
											crop_size=args.crop_size,
											dataset=args.dataset,
											split="val",
											num_workers=args.workers)
	# learning rate scheduler
	scheduler = lr_scheduler.lr_scheduler(mode=args.lr_mode,
											init_lr=args.lr,
											num_epochs=args.epochs,
											iters_per_epoch=len(train_loader),
											lr_milestones=args.lr_milestones,
											lr_step_multiplier=args.lr_step_multiplier,
											slow_start_epochs=args.slow_start_epochs,
											slow_start_lr=args.slow_start_lr,
											end_lr=args.end_lr,
											multiplier=args.lr_multiplier,
											decay_factor=args.decay_factor,
											decay_epochs=args.decay_epochs,
											staircase=True
										)

	if args.evaluate:
		validate(val_loader, model, args)
		return None

	saved_ckpt_filenames = []

	streams = None
	# streams = [torch.cuda.Stream() for i in range(args.loop_factor)]

	for epoch in range(args.start_epoch, args.epochs + 1):
		if args.is_distributed:
			train_sampler.set_epoch(epoch)

		# train for one epoch
		train(train_loader, model, optimizer, scheduler, epoch, args, streams, scaler=scaler)

		if (epoch + 1) % args.eval_per_epoch == 0:
			# evaluate on validation set
			acc_all = validate(val_loader, model, args)

			# remember best acc@1 and save checkpoint
			is_best = acc_all[0] > best_acc1
			best_acc1 = max(acc_all[0], best_acc1)

			# save checkpoint
			if not args.multiprocessing_distributed or \
				(args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
				# summary per epoch
				val_writer.add_scalar('avg_acc1', acc_all[0], global_step=epoch)
				if args.dataset == 'imagenet':
					val_writer.add_scalar('avg_acc5', acc_all[1], global_step=epoch)
				
				for i in range(2, args.loop_factor + 2):
					val_writer.add_scalar('{}_acc1'.format(i - 1), acc_all[i], global_step=epoch)
				
				val_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)
				val_writer.add_scalar('best_acc1', best_acc1, global_step=epoch)

				# save checkpoints
				filename = "checkpoint_{0}.pth.tar".format(epoch)
				saved_ckpt_filenames.append(filename)
				# remove the oldest file if the number of saved ckpts is greater than args.max_ckpt_nums
				if len(saved_ckpt_filenames) > args.max_ckpt_nums:
					os.remove(os.path.join(args.model_dir, saved_ckpt_filenames.pop(0)))
				
				ckpt_dict = {
					'epoch': epoch + 1,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'best_acc1': best_acc1,
					'optimizer': optimizer.state_dict(),
				}

				if args.is_amp:
					ckpt_dict['scaler'] = scaler.state_dict()

				metric.save_checkpoint(ckpt_dict, is_best, args.model_dir, filename=filename)

	# clean GPU cache
	torch.cuda.empty_cache()
	sys.exit(0)


def train(train_loader, model, optimizer, scheduler, epoch, args, streams=None, scaler=None):
	"""training function"""
	batch_time = metric.AverageMeter('Time', ':6.3f')
	data_time = metric.AverageMeter('Data', ':6.3f')
	avg_ce_loss = metric.AverageMeter('ce_loss', ':.4e')
	avg_cot_loss = metric.AverageMeter('cot_loss', ':.4e')
	
	# record the top1 accuray of each small network
	top1_all = []
	for i in range(args.loop_factor):
		# ce_losses_l.append(metric.AverageMeter('{}_CE_Loss'.format(i), ':.4e'))
		top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
	avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	#if args.dataset == 'imagenet':
	#	avg_top5 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	
	# show all
	total_iters = len(train_loader)
	progress = metric.ProgressMeter(total_iters, batch_time, data_time, avg_ce_loss, avg_cot_loss,
					*top1_all, avg_top1, prefix="Epoch: [{}]".format(epoch))

	# switch to train mode
	model.train()
	end = time.time()

	# prefetch data
	prefetcher = prefetch.data_prefetcher(train_loader)
	images, target = prefetcher.next()
	i = 0
	
	"""Another way to load the data
	for i, (images, target) in enumerate(train_loader):
	
		# measure data loading time
		data_time.update(time.time() - end)

		if args.gpu is not None:
			images = images.cuda(args.gpu, non_blocking=True)
		target = target.cuda(args.gpu, non_blocking=True)
	"""
	optimizer.zero_grad()
	while images is not None:
		# measure data loading time
		data_time.update(time.time() - end)
		# adjust the lr first
		scheduler(optimizer, i, epoch)
		i += 1
	
		# compute outputs and losses
		if args.is_amp:
			# Runs the forward pass with autocasting.
			with amp.autocast():
				ensemble_output, outputs, ce_loss, cot_loss = model(images,
																target=target,
																mode='train',
																epoch=epoch,
																streams=streams)
		else:
			ensemble_output, outputs, ce_loss, cot_loss = model(images,
																target=target,
																mode='train',
																epoch=epoch,
																streams=streams)

		# measure accuracy and record loss
		batch_size_now = images.size(0)
		# notice the index i and j, avoid contradictory
		for j in range(args.loop_factor):
			acc1 = metric.accuracy(outputs[j, ...], target, topk=(1, ))
			top1_all[j].update(acc1[0].item(), batch_size_now)

		# simply average outputs of small networks
		avg_acc1 = metric.accuracy(ensemble_output, target, topk=(1, ))
		avg_top1.update(avg_acc1[0].item(), batch_size_now)
		# avg_top5.update(avg_acc1[0].item(), batch_size_now)

		avg_ce_loss.update(ce_loss.mean().item(), batch_size_now)
		avg_cot_loss.update(cot_loss.mean().item(), batch_size_now)

		# compute gradient and do SGD step
		total_loss = (ce_loss + cot_loss) / args.iters_to_accumulate
		
		if args.is_amp:
			# Scales loss.  Calls backward() on scaled loss to create scaled gradients.
			# Backward passes under autocast are not recommended.
			# Backward ops run in the same dtype autocast chose for corresponding forward ops.
			scaler.scale(total_loss).backward()
			
			if i % args.iters_to_accumulate == 0 or i == total_iters:
				# scaler.step() first unscales the gradients of the optimizer's assigned params.
				# If these gradients do not contain infs or NaNs, optimizer.step() is then called,
				# otherwise, optimizer.step() is skipped.
				scaler.step(optimizer)
				# Updates the scale for next iteration.
				scaler.update()
				optimizer.zero_grad()
		else:
			total_loss.backward()
			if i % args.iters_to_accumulate == 0 or i == total_iters:
				optimizer.step()
				optimizer.zero_grad()
		
		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if not args.multiprocessing_distributed or (args.rank % args.ngpus_per_node == 0):
			if i % (args.print_freq * args.iters_to_accumulate) == 0:
				progress.print(i)
		images, target = prefetcher.next()
	
	# clean GPU cache
	# torch.cuda.empty_cache()


def validate(val_loader, model, args, streams=None):
	"""validate function"""
	batch_time = metric.AverageMeter('Time', ':6.3f')
	avg_ce_loss = metric.AverageMeter('ce_loss', ':.4e')
	
	# record the top1 accuray of each small network
	top1_all = []
	for i in range(args.loop_factor):
		top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
	avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	avg_top5 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	progress = metric.ProgressMeter(len(val_loader), batch_time, avg_ce_loss, *top1_all,
										avg_top1, avg_top5, prefix='Test: ')

	# switch to evaluate mode
	model.eval()

	with torch.no_grad():
		end = time.time()
		for i, (images, target) in enumerate(val_loader):
			if args.gpu is not None:
				images = images.cuda(args.gpu, non_blocking=True)
			target = target.cuda(args.gpu, non_blocking=True)

			# compute outputs and losses
			if args.is_amp:
				with amp.autocast():
					ensemble_output, outputs, ce_loss = model(images,
																target=target,
																mode='val'
																)
			else:
				ensemble_output, outputs, ce_loss = model(images, target=target, mode='val')

			# measure accuracy and record loss
			batch_size_now = images.size(0)
			for j in range(args.loop_factor):
				acc1, acc5 = metric.accuracy(outputs[j, ...], target, topk=(1, 5))
				top1_all[j].update(acc1[0].item(), batch_size_now)

			# simply average outputs of small networks
			avg_acc1, avg_acc5 = metric.accuracy(ensemble_output, target, topk=(1, 5))
			avg_top1.update(avg_acc1[0].item(), batch_size_now)
			avg_top5.update(avg_acc5[0].item(), batch_size_now)

			avg_ce_loss.update(ce_loss.mean().item(), batch_size_now)

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % args.print_freq == 0:
				progress.print(i)

		acc_all = []
		acc_all.append(avg_top1.avg)
		acc_all.append(avg_top5.avg)
		acc_info = '* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc_all[0], acc_all[1])
		for j in range(args.loop_factor):
			acc_all.append(top1_all[j].avg)
			acc_info += '\t {}_acc@1 {:.3f}'.format(j, top1_all[j].avg)

		print(acc_info)

	# torch.cuda.empty_cache()
	return acc_all


if __name__ == '__main__':
	torch.cuda.seed_all()

	parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
	args = parser_params.add_parser_params(parser)

	os.makedirs(args.model_dir, exist_ok=True)
	print(args)
	main(args)
