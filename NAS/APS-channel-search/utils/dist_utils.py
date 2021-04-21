import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.nn as nn


def init_dist(backend='nccl',
    master_ip='tcp://127.0.0.1',
              port=6669):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(port)
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist_url = master_ip + ':' + str(port)
    dist.init_process_group(backend=backend, init_method=dist_url, \
        world_size=world_size, rank=rank)
    return rank, world_size


def average_gradients(model):
    for param in model.parameters():
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)


def average_group_gradients(group_params):
    for param in group_params:
        if param.requires_grad and not (param.grad is None):
            dist.all_reduce(param.grad.data)


def sync_bn_stat(model, world_size):
    if world_size == 1:
        return
    for mod in model.modules():
        if type(mod) == nn.BatchNorm2d:
            dist.all_reduce(mod.running_mean.data)
            mod.running_mean.data.div_(world_size)
            dist.all_reduce(mod.running_var.data)
            mod.running_var.data.div_(world_size)


def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)


def reduce_vars(vars_list, world_size):
    if world_size == 1:
        return
    vars_result = []
    for var in vars_list:
        var = var / world_size
        dist.all_reduce(var)
        vars_result.append(var)
    return vars_result


def sync_adam_optimizer(opt, world_size):
    # use this when using Adam for parallel training
    pass
