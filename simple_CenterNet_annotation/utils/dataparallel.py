import torch
import torch.nn as nn
# from itertools import chain
from torch.nn.parallel.scatter_gather import gather
from torch.nn.parallel.replicate import replicate
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter, Gather


def scatter(inputs, target_gpus, dim=0, chunk_sizes=None):
  def scatter_map(obj):
    if isinstance(obj, torch.Tensor):
      return Scatter.apply(target_gpus, chunk_sizes, dim, obj)
    if isinstance(obj, tuple) and len(obj) > 0:
      return list(zip(*map(scatter_map, obj)))
    if isinstance(obj, list) and len(obj) > 0:
      return list(map(list, zip(*map(scatter_map, obj))))
    if isinstance(obj, dict) and len(obj) > 0:
      return list(map(type(obj), zip(*map(scatter_map, obj.items()))))
    return [obj for targets in target_gpus]

  try:
    return scatter_map(inputs)
  finally:
    scatter_map = None


def scatter_kwargs(inputs, kwargs, target_gpus, dim=0, chunk_sizes=None):
  r"""Scatter with support for kwargs dictionary"""
  inputs = scatter(inputs, target_gpus, dim, chunk_sizes) if inputs else []
  kwargs = scatter(kwargs, target_gpus, dim, chunk_sizes) if kwargs else []
  if len(inputs) < len(kwargs):
    inputs.extend([() for _ in range(len(kwargs) - len(inputs))])
  elif len(kwargs) < len(inputs):
    kwargs.extend([{} for _ in range(len(inputs) - len(kwargs))])
  inputs = tuple(inputs)
  kwargs = tuple(kwargs)
  return inputs, kwargs


class DataParallel(nn.Module):
  # TODO: update notes/cuda.rst when this class handles 8+ GPUs well

  def __init__(self, module, device_ids=None, output_device=None, dim=0, chunk_sizes=None):
    super(DataParallel, self).__init__()

    if not torch.cuda.is_available():
      self.module = module
      self.device_ids = []
      return

    if device_ids is None:
      device_ids = list(range(torch.cuda.device_count()))
    if output_device is None:
      output_device = device_ids[0]

    self.dim = dim
    self.module = module
    self.chunk_sizes = chunk_sizes
    self.device_ids = device_ids
    self.output_device = output_device
    self.src_device_obj = torch.device("cuda:{}".format(self.device_ids[0]))

    if len(self.device_ids) == 1:
      self.module.cuda(device_ids[0])

  def forward(self, *inputs, **kwargs):
    if not self.device_ids:
      return self.module(*inputs, **kwargs)

    inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
    if len(self.device_ids) == 1:
      return self.module(*inputs[0], **kwargs[0])
    replicas = self.replicate(self.module, self.device_ids[:len(inputs)])
    outputs = self.parallel_apply(replicas, inputs, kwargs)
    return self.gather(outputs, self.output_device)

  def replicate(self, module, device_ids):
    return replicate(module, device_ids)

  def scatter(self, inputs, kwargs, device_ids):
    return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim, chunk_sizes=self.chunk_sizes)

  def parallel_apply(self, replicas, inputs, kwargs):
    return parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])

  def gather(self, outputs, output_device):
    return gather(outputs, output_device, dim=self.dim)
