##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
######################################################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
from pathlib import Path
lib_dir = (Path(__file__).parent / '..' / '..' / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from config_utils import load_config, dict2config, configure2str
from datasets     import get_datasets, get_nas_search_loaders
from procedures   import prepare_seed, prepare_logger, save_checkpoint, copy_checkpoint, get_optim_scheduler
from utils        import get_model_infos, obtain_accuracy
from log_utils    import AverageMeter, time_string, convert_secs2time
from models       import get_cell_based_tiny_net, get_search_spaces, OPS_CODING
from nas_102_api  import NASBench102API as API
import pdb
import time
import scipy
import scipy.stats
from weight_angle import get_arch_angle

def load(checkpoint_path, model):
  checkpoint  = torch.load(checkpoint_path)
  model.load_state_dict( checkpoint['search_model'] )
  operation = checkpoint['genotype']
  return operation

def main(xargs):
  assert torch.cuda.is_available(), 'CUDA is not available.'
  torch.backends.cudnn.enabled   = True
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True
  torch.set_num_threads( xargs.workers )
  prepare_seed(xargs.rand_seed)
  logger = prepare_logger(args)

  train_data, valid_data, xshape, class_num = get_datasets(xargs.dataset, xargs.data_path, -1)
  config = load_config(xargs.config_path, {'class_num': class_num, 'xshape': xshape}, logger)
  search_loader, _, valid_loader = get_nas_search_loaders(train_data, valid_data, xargs.dataset, 'configs/nas-benchmark/', \
                                        (config.batch_size, config.test_batch_size), xargs.workers)
  logger.log('||||||| {:10s} ||||||| Search-Loader-Num={:}, Valid-Loader-Num={:}, batch size={:}'.format(xargs.dataset, len(search_loader), len(valid_loader), config.batch_size))
  logger.log('||||||| {:10s} ||||||| Config={:}'.format(xargs.dataset, config))

  search_space = get_search_spaces('cell', xargs.search_space_name)
  model_config = dict2config({'name': 'SPOS', 'C': xargs.channel, 'N': xargs.num_cells,
                              'max_nodes': xargs.max_nodes, 'num_classes': class_num,
                              'space'    : search_space,
                              'affine'   : False, 'track_running_stats': bool(xargs.track_running_stats)}, None)
  logger.log('search space : {:}'.format(search_space))
  model = get_cell_based_tiny_net(model_config)
  
  flop, param  = get_model_infos(model, xshape)
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space : {:}'.format(search_space))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  angles = {}
  for arch_idx in range(0, 10000, 200):
    checkpoint_path_template = 'output/search-cell-nas-bench-102/result-{}/standalone_arch-{}/checkpoint/seed-{}_epoch-{}.pth'
    logger.log("=> loading checkpoint from {}".format(checkpoint_path_template.format(args.dataset, arch_idx, args.rand_seed, 0)))
    epochs = config.epochs - 1
    load(checkpoint_path_template.format(args.dataset, arch_idx,  args.rand_seed, 0), model)
    init_model = deepcopy(model)
    genotype = load(checkpoint_path_template.format(args.dataset, arch_idx,  args.rand_seed, epochs), model)
    logger.log("=> loading checkpoint from {}".format(checkpoint_path_template.format(args.dataset, arch_idx,  args.rand_seed, epochs)))
    cur_model = deepcopy(model)
    angle = get_arch_angle(init_model, cur_model, genotype, search_space)
    logger.log('[{:}] cal angle : angle={} | {:}, acc: {}'.format(arch_idx, angle, genotype, get_arch_real_acc(api, genotype, args)))
    angles[genotype.tostr()] = angle

  real_acc = {}
  for key in angles.keys():
    real_acc[key] = get_arch_real_acc(api, key, args)
    assert(real_acc[key] is not None)

  real_acc = sorted(real_acc.items(), key=lambda d: d[1], reverse=True)
  angles = sorted(angles.items(), key=lambda d: d[1], reverse=True)
  angle_rank = {}
  rank = 1
  for value in angles:
    angle_rank[value[0]] = rank
    rank += 1

  angle_rank_list, real_rank_list = [],[]
  rank = 1
  for value in real_acc:
    angle_rank_list.append(angle_rank[value[0]])
    real_rank_list.append(rank)
    rank += 1

  logger.log('Real_rank_list={}'.format(real_rank_list))
  logger.log('Angle_rank_list={}'.format(angle_rank_list))
  logger.log('Tau={}'.format(scipy.stats.stats.kendalltau(real_rank_list, angle_rank_list)[0]))

def get_arch_real_acc(api, genotype, args):
  info = api.query_by_arch(genotype)
  id = api.query_index_by_arch(genotype)
  if id in api.arch2infos_full:
    info = api.arch2infos_full[id]
    test_info = info.get_metrics(args.dataset, 'ori-test')
    return test_info['accuracy']
  return None

if __name__ == '__main__':
  parser = argparse.ArgumentParser("SETN")
  parser.add_argument('--data_path',          type=str,   help='Path to dataset')
  parser.add_argument('--dataset',            type=str,   choices=['cifar10', 'cifar100', 'ImageNet16-120'], help='Choose between Cifar10/100 and ImageNet-16.')
  # channels and number-of-cells
  parser.add_argument('--search_space_name',  type=str,   help='The search space name.')
  parser.add_argument('--max_nodes',          type=int,   help='The maximum number of nodes.')
  parser.add_argument('--channel',            type=int,   help='The number of channels.')
  parser.add_argument('--num_cells',          type=int,   help='The number of cells in one stage.')
  parser.add_argument('--select_num',         type=int,   help='The number of selected architectures to evaluate.')
  parser.add_argument('--track_running_stats',type=int,   choices=[0,1],help='Whether use track_running_stats or not in the BN layer.')
  parser.add_argument('--config_path',        type=str,   help='The path of the configuration.')
  # architecture leraning rate
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay',  type=float, default=1e-3, help='weight decay for arch encoding')
  # log
  parser.add_argument('--workers',            type=int,   default=2,    help='number of data loading workers (default: 2)')
  parser.add_argument('--save_dir',           type=str,   help='Folder to save checkpoints and log.')
  parser.add_argument('--arch_nas_dataset',   type=str,   help='The path to load the architecture dataset (tiny-nas-benchmark).')
  parser.add_argument('--print_freq',         type=int,   help='print frequency (default: 200)')
  parser.add_argument('--rand_seed',          type=int,   help='manual seed')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
