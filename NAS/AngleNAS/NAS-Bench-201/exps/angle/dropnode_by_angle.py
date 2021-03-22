##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
######################################################################################
import os, sys, time, glob, random, argparse
import numpy as np
from copy import deepcopy
import copy
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
from models       import get_cell_based_tiny_net, get_search_spaces, CellStructure, ReLUConvBN
from nas_102_api  import NASBench102API as API
import random
import scipy
import scipy.stats
from weight_angle import get_arch_angle

def search_func(xloader, network, operations, criterion, scheduler, w_optimizer, epoch_str, print_freq, logger):
  data_time, batch_time = AverageMeter(), AverageMeter()
  base_losses, base_top1, base_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  arch_losses, arch_top1, arch_top5 = AverageMeter(), AverageMeter(), AverageMeter()
  end = time.time()
  network.train()
  for step, (base_inputs, base_targets, arch_inputs, arch_targets) in enumerate(xloader):
    scheduler.update(None, 1.0 * step / len(xloader))
    base_targets = base_targets.cuda(non_blocking=True)
    arch_targets = arch_targets.cuda(non_blocking=True)
    # measure data loading time
    data_time.update(time.time() - end)
    
    # update the weights
    network.module.set_cal_mode( 'dropnode' )
    network.zero_grad()
    _, logits = network(base_inputs, operations)
    base_loss = criterion(logits, base_targets)
    base_loss.backward()
    w_optimizer.step()
    # record
    base_prec1, base_prec5 = obtain_accuracy(logits.data, base_targets.data, topk=(1, 5))
    base_losses.update(base_loss.item(),  base_inputs.size(0))
    base_top1.update  (base_prec1.item(), base_inputs.size(0))
    base_top5.update  (base_prec5.item(), base_inputs.size(0))

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if step % print_freq == 0 or step + 1 == len(xloader):
      Sstr = '*SEARCH* ' + time_string() + ' [{:}][{:03d}/{:03d}]'.format(epoch_str, step, len(xloader))
      Tstr = 'Time {batch_time.val:.2f} ({batch_time.avg:.2f}) Data {data_time.val:.2f} ({data_time.avg:.2f})'.format(batch_time=batch_time, data_time=data_time)
      Wstr = 'Base [Loss {loss.val:.3f} ({loss.avg:.3f})  Prec@1 {top1.val:.2f} ({top1.avg:.2f}) Prec@5 {top5.val:.2f} ({top5.avg:.2f})]'.format(loss=base_losses, top1=base_top1, top5=base_top5)
      logger.log(Sstr + ' ' + Tstr + ' ' + Wstr)
  return base_losses.avg, base_top1.avg, base_top5.avg, arch_losses.avg, arch_top1.avg, arch_top5.avg

def train_func(xargs, search_loader, network, operations, criterion, w_scheduler, w_optimizer, logger, iters, total_epoch):
  logger.log('|=> Iters={}, train: operations={}, epochs={}'.format(iters, operations, total_epoch))
  # start training
  start_time, search_time, epoch_time, start_epoch = time.time(), AverageMeter(), AverageMeter(), 0
  for epoch in range(start_epoch, total_epoch):
    w_scheduler.update(epoch, 0.0)
    need_time = 'Time Left: {:}'.format( convert_secs2time(epoch_time.val * (total_epoch-epoch), True) )
    epoch_str = '{:03d}-{:03d}'.format(epoch, total_epoch)
    logger.log('\n[Search the {:}-th epoch] {:}, LR={:}'.format(epoch_str, need_time, min(w_scheduler.get_lr())))

    search_w_loss, search_w_top1, search_w_top5, search_a_loss, search_a_top1, search_a_top5 \
                = search_func(search_loader, network, operations, criterion, w_scheduler, w_optimizer, epoch_str, xargs.print_freq, logger)
    search_time.update(time.time() - start_time)
    logger.log('[{:}] search [base] : loss={:.2f}, accuracy@1={:.2f}%, accuracy@5={:.2f}%, time-cost={:.1f} s'.format(epoch_str, search_w_loss, search_w_top1, search_w_top5, search_time.sum))    

def get_arch_real_acc(api, genotype, args):
  if args.dataset == 'cifar10':
    dataset, xset, dataset_v, xset_v = 'cifar10', 'ori-test', 'cifar10-valid', 'x-valid'
  elif args.dataset == 'cifar100':
    dataset, xset, dataset_v, xset_v = 'cifar100', 'x-test', 'cifar100', 'x-valid'
  else:
    dataset, xset, dataset_v, xset_v = 'ImageNet16-120', 'x-test', 'ImageNet16-120', 'x-valid'

  info = api.query_by_arch(genotype)
  id = api.query_index_by_arch(genotype)
  if id in api.arch2infos_full:
    info = api.arch2infos_full[id]
    test_info = info.get_metrics(dataset, xset)
    valid_info = info.get_metrics(dataset_v, xset_v)
    return test_info['accuracy']
  return None

def get_all_archs(operations):
  combs = []
  for i in range(1, 4):
    for j in range(i):
      if len(combs) == 0:
        for func in operations[(i, j)]:
          combs.append( [(func, j)] )
      else:
        new_combs = []
        for string in combs:
          for func in operations[(i, j)]:
            xstring = string + [(func, j)]
            new_combs.append( xstring )
        combs = new_combs
  operations = combs

  operations_ = []
  for ops in operations:
    temp = [[ops[0]],[ops[1],ops[2]],[ops[3],ops[4],ops[5]]]
    operations_.append(CellStructure(temp))
  return operations_

def compute_scores(extend_operators, operations, vis_dict, vis_dict_slice, search_space, api, args, base_network, network, logger):
  # Get all architecutures
  candidates = []
  all_archs = get_all_archs(operations)
  for arch in all_archs:
    candidates.append(arch)
  
  # Compute angles of all candidate architecures
  angles = {}
  for i, cand in enumerate(candidates):
      vis_dict[tuple(cand.nodes)] = {}
      info = vis_dict[tuple(cand.nodes)]
      info['angle'] = get_arch_angle(base_network, network, cand, search_space)
      angles[cand.tostr()] = info['angle']
      logger.log('i={}, angle={}'.format(i, info['angle']))

  # Caculate sum of angles for each operator
  for cand in candidates:
      info = vis_dict[tuple(cand.nodes)]
      for i in range(1, 4):
        cur_op_node = cand.nodes[i-1]
        for op_name, j in cur_op_node:
          extend_operator = (i, j, op_name)
          if extend_operator in vis_dict_slice:
              slice_info = vis_dict_slice[extend_operator]
              slice_info['angle'] += info['angle']
              slice_info['real_acc'] += get_arch_real_acc(api, cand, args)
              slice_info['count'] += 1

    # Compute scores of all candidate operators    
  for extend_operator in extend_operators:
      if vis_dict_slice[extend_operator]['count'] > 0:
          vis_dict_slice[extend_operator]['angle'] = vis_dict_slice[extend_operator]['angle'] * 1. / vis_dict_slice[extend_operator]['count']

def drop_operators(extend_operators, operations, vis_dict_slice, real_rank, iters, drop_ops_num, logger):
  # Each operator is ranked according to its score
  extend_operators.sort(key=lambda x:vis_dict_slice[x]['real_acc'], reverse=True)
  rank = 1 
  for cand in extend_operators:
    real_rank[cand] = rank
    rank += 1

  # Drop operators whose ranking fall at the tail.
  num = 0
  extend_operators.sort(key=lambda x:vis_dict_slice[x]['angle'], reverse=False)
  for idx, cand in enumerate(extend_operators):
      info = vis_dict_slice[cand]
      logger.log('Iter={}, shrinking: top {} cand={}, angle={}'.format(iters+1, idx+1, cand, info['angle']))

  num, drop_ops, drop_ranks = 0, [], []
  for idx, cand in enumerate(extend_operators):
      ii, jj, op = cand
      drop_legal = False
      # Make sure that at least a operator is reserved for each layer.
      for i in range(idx+1, len(extend_operators)):
          ii_, jj_, op_ = extend_operators[i]
          if ii == ii_ and jj == jj_:
              drop_legal = True
      if drop_legal:
          logger.log('no.{} drop_op={}'.format(num+1, cand))
          drop_ops.append(cand)
          drop_ranks.append(real_rank[cand])
          operations[(ii, jj)].remove(op)
          num += 1
      if num >= drop_ops_num:
          break
  return drop_ops, drop_ranks

# Algorithm 2
def ABS(xargs, iters, base_network, network, operations, drop_ops_num, search_space, logger, api):
  vis_dict, vis_dict_slice, real_rank = {}, {}, {}
  extend_operators = []
  # At least one operator is preserved for each edge
  # Each operator is identified by its edge and type
  for edge in operations.keys():
      if len(operations[edge]) > 1:
          for op in operations[edge]:
              cand = (edge[0], edge[1], op)
              vis_dict_slice[cand]={}
              info=vis_dict_slice[cand]
              info['angle'] = 0.
              info['count'] = 0.
              info['real_acc'] = 0.
              extend_operators.append(cand)
  logger.log('Extend_cands={}'.format(extend_operators))
  compute_scores(extend_operators, operations, vis_dict, vis_dict_slice, search_space, api, xargs, base_network, network, logger)
  drop_ops, drop_ranks = drop_operators(extend_operators, operations, vis_dict_slice, real_rank, iters, drop_ops_num, logger)
  logger.log('Iter={}, shrinking: drop_ops={}, real_ranks={}'.format(iters, drop_ops, drop_ranks))

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
  search_model = get_cell_based_tiny_net(model_config)

  w_optimizer, w_scheduler, criterion = get_optim_scheduler(search_model.get_weights(), config)
  a_optimizer = torch.optim.Adam(search_model.get_alphas(), lr=xargs.arch_learning_rate, betas=(0.5, 0.999), weight_decay=xargs.arch_weight_decay)
  logger.log('w-optimizer : {:}'.format(w_optimizer))
  logger.log('a-optimizer : {:}'.format(a_optimizer))
  logger.log('w-scheduler : {:}'.format(w_scheduler))
  logger.log('criterion   : {:}'.format(criterion))
  flop, param  = get_model_infos(search_model, xshape)
  logger.log('FLOP = {:.2f} M, Params = {:.2f} MB'.format(flop, param))
  logger.log('search-space : {:}'.format(search_space))
  if xargs.arch_nas_dataset is None:
    api = None
  else:
    api = API(xargs.arch_nas_dataset)
  logger.log('{:} create API = {:} done'.format(time_string(), api))

  last_info, model_base_path, model_best_path = logger.path('info'), logger.path('model'), logger.path('best')
  network, criterion = torch.nn.DataParallel(search_model).cuda(), criterion.cuda()
  total_epochs = xargs.epochs
  drop_ops_num = 20
  total_iters = 30 // drop_ops_num
  start_iter, epochs, operations = 0, total_epochs//total_iters, {}
  
  for i in range(1, 4):
    for j in range(i):
      node_str = (i, j)
      operations[node_str] = copy.deepcopy(search_space)
  logger.log('operations={}'.format(operations))

  # Save base weights for computing angle
  base_network = copy.deepcopy(search_model)

  for i in range(start_iter, total_iters):
      train_func(xargs, search_loader, network, operations, \
                                          criterion, w_scheduler, w_optimizer, logger, i, epochs)
      ABS(xargs, i, base_network, network.module, operations, drop_ops_num, search_space, logger, api)
  logger.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser("SPOS-DropNode")
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
  parser.add_argument('--name',               type=str,   help='name')
  parser.add_argument('--epochs',            type=int,   help='The number of epochs.')
  args = parser.parse_args()
  if args.rand_seed is None or args.rand_seed < 0: args.rand_seed = random.randint(1, 100000)
  main(args)
