# Copyright (c) 2020 -      Shuai Zhao
#
# All rights reserved.
#
# Contact: zhaoshuaimcc@gmail.com
#
# Ref:
# [1] https://github.com/pytorch/examples/tree/master/imagenet
# [2] https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets


import os
import sys
import time
import argparse
import setproctitle
import numpy as np

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F

import parser_params
from model import splitnet
from dataset import factory
from utils import metric, norm

_GEO_TEST = True


class data_prefetcher_2gpus():
	def __init__(self, loader, ngpus=2):
		self.loader = iter(loader)
		self.stream = torch.cuda.Stream()
		self.ngpus = ngpus
		self.preload()

	def preload(self):
		try:
			self.next_images, self.next_target = next(self.loader)
		except StopIteration:
			self.next_images_gpu0 = None
			self.next_images_gpu1 = None
			self.next_target = None
			return
		with torch.cuda.stream(self.stream):
			self.next_images_gpu0 = self.next_images.cuda(0, non_blocking=True)
			if self.ngpus > 1:
				self.next_images_gpu1 = self.next_images.cuda(1, non_blocking=True)
			else:
				self.next_images_gpu1 = None
			self.next_target = self.next_target.cuda(0, non_blocking=True)

	def next(self):
		torch.cuda.current_stream().wait_stream(self.stream)
		images_gpu0 = self.next_images_gpu0
		images_gpu1 = self.next_images_gpu1
		target = self.next_target
		self.preload()

		return images_gpu0, target, images_gpu1


def multigpu_test_2gpus(args):
	"""
	This is a simple program for validating the idea of parallel runing of multiple
	model on multiple gpus.
	"""
	model = splitnet.SplitNet(args,
								norm_layer=norm.norm(args.norm_mode),
								criterion=None)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("INFO:PyTorch: => loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			old_dict = checkpoint['state_dict']
			# orignial ckpt was save as nn.parallel.DistributedDataParallel() object
			old_dict = {k.replace("module.models", "models"): v for k, v in old_dict.items()}
			model.load_state_dict(old_dict)
			print("INFO:PyTorch: => loaded checkpoint"
					" '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("INFO:PyTorch: => no checkpoint found at '{}'".format(args.resume))

	# accelarate the training
	torch.backends.cudnn.benchmark = True

	val_loader = factory.get_data_loader(args.data,
											batch_size=args.eval_batch_size,
											crop_size=args.crop_size,
											dataset=args.dataset,
											split="val",
											num_workers=args.workers)
	# record the top1 accuray of each small network
	top1_all = []
	for i in range(args.loop_factor):
		top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
	avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	avg_top5 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	progress = metric.ProgressMeter(len(val_loader), *top1_all,
										avg_top1, avg_top5, prefix='Test: ')

	# switch to evaluate mode
	model.eval()
	# move model to the gpu
	if args.is_test_on_multigpus:
		print("INFO:PyTorch: multi GPUs test")
		cuda_models = []
		for idx in range(args.split_factor):
			cuda_models.append(model.models[idx].cuda(idx))
	else:
		print("INFO:PyTorch: single GPU test")
		model = model.cuda(0)

	with torch.no_grad():
		# record time and number of samples
		prefetcher = data_prefetcher_2gpus(val_loader, ngpus=args.split_factor)
		images_gpu0, target, images_gpu1 = prefetcher.next()
		i = 0
		n_count = 0.0
		start_time = time.time()

		while images_gpu0 is not None:
			i += 1
			# for i, (images, target) in enumerate(val_loader):
			# compute outputs and losses
			if args.is_test_on_multigpus:
				if args.is_amp:
					with amp.autocast():
						output_gpu0 = cuda_models[0](images_gpu0)
					with amp.autocast():
						output_gpu1 = cuda_models[1](images_gpu1)
				else:
					output_gpu0 = cuda_models[0](images_gpu0)
					output_gpu1 = cuda_models[1](images_gpu1)
				
				if _GEO_TEST:
					if i == 1:
						print("using geometry mean")
					output_gpu0 = F.softmax(output_gpu0, dim=-1)
					output_gpu1 = F.softmax(output_gpu1, dim=-1)
					ensemble_output = torch.sqrt(output_gpu0 * output_gpu1.cuda(0))
				else:
					outputs = torch.stack([output_gpu0, output_gpu1.cuda(0)])
					ensemble_output = torch.mean(outputs, dim=0)

			else:
				# compute outputs and losses
				if args.is_amp:
					with amp.autocast():
						ensemble_output, outputs, ce_loss = model(images_gpu0,
																	target=target,
																	mode='val'
																	)
				else:
					ensemble_output, outputs, ce_loss = model(images_gpu0, target=target, mode='val')

			# measure accuracy and record loss
			"""
			target = target.cpu()
			ensemble_output = ensemble_output.cpu().float()
			outputs = outputs.cpu().float()
			"""

			batch_size_now = images_gpu0.size(0)
			"""
			for j in range(args.loop_factor):
				acc1, acc5 = metric.accuracy(outputs[j, ...], target, topk=(1, 5))
				top1_all[j].update(acc1[0].item(), batch_size_now)
			"""
			# simply average outputs of small networks
			avg_acc1, avg_acc5 = metric.accuracy(ensemble_output, target, topk=(1, 5))
			avg_top1.update(avg_acc1[0].item(), batch_size_now)
			avg_top5.update(avg_acc5[0].item(), batch_size_now)

			images_gpu0, target, images_gpu1 = prefetcher.next()

			n_count += batch_size_now
			"""
			if i % args.print_freq == 0:
				progress.print(i)
			"""
		time_cnt = time.time() - start_time
		# print accuracy info
		acc_all = []
		acc_all.append(avg_top1.avg)
		acc_all.append(avg_top5.avg)
		acc_info = '* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc_all[0], acc_all[1])
		"""
		mean_acc = 0.0
		for j in range(args.loop_factor):
			acc_all.append(top1_all[j].avg)
			acc_info += '\t {}_acc@1 {:.3f}'.format(j, top1_all[j].avg)
			mean_acc += top1_all[j].avg
		acc_info += "\t avg_acc {:.3f}".format(mean_acc / args.split_factor)
		"""
		print(acc_info)

	print("multiple GPUs ({})".format(args.is_test_on_multigpus))
	print("The tested architecture is {} with split_factor {}".format(args.arch, args.split_factor))
	print("The number of the samples is {}".format(n_count))
	print("The total testing time is {} second".format(time_cnt))
	print("The average test time is {}ms per images".format(1000 * time_cnt / n_count))

	torch.cuda.empty_cache()
	sys.exit(0)


def multigpu_test(args):
	"""
	This is a simple program for validating the idea of parallel runing of multiple
	model on multiple gpus.
	"""
	model = splitnet.SplitNet(args, norm_layer=norm.norm(args.norm_mode), criterion=None)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("INFO:PyTorch: => loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			old_dict = checkpoint['state_dict']
			# orignial ckpt was save as nn.parallel.DistributedDataParallel() object
			old_dict = {k.replace("module.models", "models"): v for k, v in old_dict.items()}
			model.load_state_dict(old_dict)
			print("INFO:PyTorch: => loaded checkpoint"
					" '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("INFO:PyTorch: => no checkpoint found at '{}'".format(args.resume))

	# accelarate the training
	torch.backends.cudnn.benchmark = True

	val_loader = factory.get_data_loader(args.data,
											batch_size=args.eval_batch_size,
											crop_size=args.crop_size,
											dataset=args.dataset,
											split="val",
											num_workers=args.workers)
	# record the top1 accuray of each small network
	top1_all = []
	for i in range(args.loop_factor):
		top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
	avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	avg_top5 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	progress = metric.ProgressMeter(len(val_loader), *top1_all, avg_top1, avg_top5, prefix='Test: ')

	# switch to evaluate mode
	model.eval()
	n_count = 0.0

	# move model to the gpu
	cuda_models = []
	for idx in range(args.split_factor):
		cuda_models.append(model.models[idx].cuda(idx))
	start_time = time.time()

	for i, (images, target) in enumerate(val_loader):
		cuda_images = []
		cuda_outpouts = []
		collect_outputs = []
		target = target.cuda(0, non_blocking=True)
		for idx in range(args.split_factor):
			cuda_images.append(images.cuda(idx, non_blocking=True))

		if args.is_amp:
			with amp.autocast():
				for idx in range(args.split_factor):
					cuda_outpouts.append(cuda_models[idx](cuda_images[idx]))
		else:
			for idx in range(args.split_factor):
				cuda_outpouts.append(cuda_models[idx](cuda_images[idx]))

		for idx in range(args.split_factor):
			# use the first gpu as host gpu
			collect_outputs.append(cuda_outpouts[idx].cuda(0))

		if _GEO_TEST:
			if i == 1:
				print("using geometry mean")
			cmul = 1.0
			for j in range(args.split_factor):
				cmul = cmul * F.softmax(cuda_outpouts[j].cuda(0), dim=-1)
			# ensemble_output = torch.pow(cmul, 1.0 / args.split_factor)
			ensemble_output = torch.sqrt(cmul)
		else:
			outputs = torch.stack(collect_outputs, dim=0)
			ensemble_output = torch.mean(outputs, dim=0)

		batch_size_now = images.size(0)
		"""
		for j in range(args.loop_factor):
			acc1, acc5 = metric.accuracy(outputs[j, ...], target, topk=(1, 5))
			top1_all[j].update(acc1[0].item(), batch_size_now)
		"""
		# simply average outputs of small networks
		avg_acc1, avg_acc5 = metric.accuracy(ensemble_output, target, topk=(1, 5))
		avg_top1.update(avg_acc1[0].item(), batch_size_now)
		avg_top5.update(avg_acc5[0].item(), batch_size_now)

		n_count += batch_size_now
		"""
		if i % args.print_freq == 0:
			progress.print(i)
		"""
	time_cnt = time.time() - start_time
	# print accuracy info
	acc_all = []
	acc_all.append(avg_top1.avg)
	acc_all.append(avg_top5.avg)
	acc_info = '* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc_all[0], acc_all[1])
	"""
	mean_acc = 0.0
	for j in range(args.loop_factor):
		acc_all.append(top1_all[j].avg)
		acc_info += '\t {}_acc@1 {:.3f}'.format(j, top1_all[j].avg)
		mean_acc += top1_all[j].avg
	acc_info += "\t avg_acc {:.3f}".format(mean_acc / args.split_factor)
	"""
	print(acc_info)

	print("multiple GPUs ({})".format(args.is_test_on_multigpus))
	print("The tested architecture is {} with split_factor {}".format(args.arch, args.split_factor))
	print("The number of the samples is {}".format(n_count))
	print("The total testing time is {} second".format(time_cnt))
	print("The average test time is {}ms per images".format(1000 * time_cnt / n_count))

	torch.cuda.empty_cache()
	sys.exit(0)


def multistreams_test(args):
	"""
	This is a simple program for validating the idea of parallel runing of multiple
	model on single gpu via multi cuda streams.
	"""
	model = splitnet.SplitNet(args,
								norm_layer=norm.norm(args.norm_mode),
								criterion=None)

	# optionally resume from a checkpoint
	if args.resume:
		if os.path.isfile(args.resume):
			print("INFO:PyTorch: => loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume)
			old_dict = checkpoint['state_dict']
			# orignial ckpt was save as nn.parallel.DistributedDataParallel() object
			old_dict = {k.replace("module.models", "models"): v for k, v in old_dict.items()}
			
			model.load_state_dict(old_dict)
			print("INFO:PyTorch: => loaded checkpoint"
					" '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
		else:
			print("INFO:PyTorch: => no checkpoint found at '{}'".format(args.resume))

	# accelarate the training
	torch.backends.cudnn.benchmark = True

	val_loader = factory.get_data_loader(args.data,
											batch_size=args.eval_batch_size,
											crop_size=args.crop_size,
											dataset=args.dataset,
											split="val",
											num_workers=args.workers)
	# record the top1 accuray of each small network
	top1_all = []
	for i in range(args.loop_factor):
		top1_all.append(metric.AverageMeter('{}_Acc@1'.format(i), ':6.2f'))
	avg_top1 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	avg_top5 = metric.AverageMeter('Avg_Acc@1', ':6.2f')
	progress = metric.ProgressMeter(len(val_loader), *top1_all,
										avg_top1, avg_top5, prefix='Test: ')

	# switch to evaluate mode
	model.eval()
	# move model to the gpu
	cuda_models = []
	cuda_streams = []
	for idx in range(args.split_factor):
		cuda_streams.append(torch.cuda.Stream())
		cuda_models.append(model.models[idx].cuda(0))
	torch.cuda.synchronize()

	# record time and number of samples
	n_count = 0.0
	start_time = time.time()

	with torch.no_grad():
		for i, (images, target) in enumerate(val_loader):
			images = images.cuda(0, non_blocking=True)
			target = target.cuda(0, non_blocking=True)
			collect_outputs = []
	
			if args.is_amp:
				with torch.cuda.stream(cuda_streams[0]):
					with amp.autocast():
						output_0 = cuda_models[0](images)

				with torch.cuda.stream(cuda_streams[1]):
					with amp.autocast():
						output_1 = cuda_models[1](images)
			
			else:
				for idx in range(args.split_factor):
					with torch.cuda.stream(cuda_streams[idx]):
						collect_outputs.append(cuda_models[idx](images))
			torch.cuda.synchronize()

			collect_outputs.extend([output_0, output_1])
			# output is fp16
			outputs = torch.stack(collect_outputs, dim=0)
			ensemble_output = torch.mean(outputs, dim=0)

			# measure accuracy and record loss
			batch_size_now = images.size(0)
			n_count += batch_size_now
			for j in range(args.loop_factor):
				acc1, acc5 = metric.accuracy(outputs[j, ...], target, topk=(1, 5))
				top1_all[j].update(acc1[0].item(), batch_size_now)

			# simply average outputs of small networks
			avg_acc1, avg_acc5 = metric.accuracy(ensemble_output, target, topk=(1, 5))
			avg_top1.update(avg_acc1[0].item(), batch_size_now)
			avg_top5.update(avg_acc5[0].item(), batch_size_now)

			#if i >= 200:
			#	break
			
			if i % args.print_freq == 0:
				progress.print(i)

		time_cnt = time.time() - start_time

		# print accuracy info
		acc_all = []
		acc_all.append(avg_top1.avg)
		acc_all.append(avg_top5.avg)
		acc_info = '* Acc@1 {:.3f} Acc@5 {:.3f}'.format(acc_all[0], acc_all[1])
		mean_acc = 0.0
		for j in range(args.loop_factor):
			acc_all.append(top1_all[j].avg)
			acc_info += '\t {}_acc@1 {:.3f}'.format(j, top1_all[j].avg)
			mean_acc += top1_all[j].avg
		acc_info += "\t avg_acc {:.3f}".format(mean_acc / args.split_factor)
		print(acc_info)

	print("The tested architecture is {} with split_factor {}".format(args.arch, args.split_factor))
	print("The number of the samples is {}".format(n_count))
	print("The total testing time is {} second".format(time_cnt))
	print("The average test time is {}ms per images".format(1000 * time_cnt / n_count))

	torch.cuda.empty_cache()
	sys.exit(0)


def toy_test_with_streams():
	torch.cuda.set_device(0)
	s1 = torch.cuda.Stream()
	s2 = torch.cuda.Stream()
	size = 1000
	# Initialise cuda tensors here. E.g.:
	A = torch.rand(size, size).cuda()
	B = torch.rand(size, size).cuda()

	# Wait for the above tensors to initialise.
	torch.cuda.synchronize()
	start_time = time.time()
	with torch.cuda.stream(s1):
		_ = torch.mm(A, A)
	with torch.cuda.stream(s2):
		_ = torch.mm(B, B)
	# Wait for C and D to be computed.
	torch.cuda.synchronize()

	runing_time = time.time() - start_time
	print("The runing time is {}".format(runing_time))

	# Initialise cuda tensors here. E.g.:
	C = torch.rand(size, size).cuda()
	D = torch.rand(size, size).cuda()
	# Wait for the above tensors to initialise.
	torch.cuda.synchronize()
	start_time = time.time()
	with torch.cuda.stream(torch.cuda.default_stream()):
		_ = torch.mm(C, C)
		_ = torch.mm(D, D)
	torch.cuda.synchronize()
	runing_time = time.time() - start_time
	print("The runing time is {}".format(runing_time))


if __name__ == '__main__':
	torch.cuda.seed_all()

	"""
	toy_test_with_streams()
	sys.exit(0)
	"""

	parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
	args = parser_params.add_parser_params(parser)
	
	# If we traing the model seperately, all the number of loops will be one.
	# It is similar as split_factor = 1
	args.loop_factor = 1 if args.is_train_sep else args.split_factor
	# set the name of the process
	setproctitle.setproctitle(args.proc_name)
	if args.split_factor == 1:
		args.is_test_on_multigpus = 0
		args.is_test_with_multistreams = 0

	# create model
	if args.pretrained:
		model_info = "INFO:PyTorch: using pre-trained model '{}'".format(args.arch)
	else:
		model_info = "INFO:PyTorch: creating model '{}'".format(args.arch)

	print(model_info)
	os.makedirs(args.model_dir, exist_ok=True)
	print(args)

	with torch.no_grad():
		if args.is_test_with_multistreams:
			print("INFO:PyTorch: Test SplitNet with multi streams on single GPU")
			multistreams_test(args)
		elif args.split_factor <= 2:
			multigpu_test_2gpus(args)
		else:
			multigpu_test(args)
