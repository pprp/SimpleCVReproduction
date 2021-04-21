import argparse
import torch
from pdb import set_trace as br

__all__ = ['decode_cfg']

def transfer_archs_res20s(args, archs):
  candidates = [int(v) for v in args.candidate_width.split(',')]

  # obtain full archs (expand blockwise archs)
  if args.blockwise:
    extended_archs = []
    for layer_id, arch in enumerate(archs):
        if layer_id == 0:
            extended_archs += [arch]
        else:
            extended_archs += [arch]*2
    full_archs = extended_archs
  else:
    full_archs = archs

  # determine candidates according to archs
  nb_layers = len(full_archs)
  candidates_list = [candidates] * nb_layers
  expand_candidates = []
  for idx, cands in enumerate(candidates_list):
    if idx <= (nb_layers-1) // 3:
      expand_candidates.append([v * args.multiplier for v in cands])
    elif (nb_layers-1)//3 < idx <= (nb_layers-1)//3*2:
      expand_candidates.append(cands)
    elif idx > (nb_layers-1)//3*2:
      expand_candidates.append([v * args.multiplier * args.multiplier for v in cands])

  cfgs = []
  for arch, cands in zip(full_archs, expand_candidates):
    cfgs.append(int(cands[arch]))
  return cfgs


def transfer_archs_res18s(args, archs, num_blocks, block_layer_num):
  def extend_blockwise_archs(archs):
    # NOTE: equivalent to obtain full archs in resnet18
    extended_archs = [archs[0]] # add the first conv cand inside
    idx = 1
    block_out_idx = 0
    for num in num_blocks:
      # num + 1 because of inc and outc
      for block_id in range(num+1):
        if block_id == 0:
          extended_archs += [archs[idx]]*(block_layer_num-1)
        elif block_id == 1:
          block_out_idx = idx
          extended_archs += [archs[idx]]
        else:
          extended_archs += [archs[idx]]*(block_layer_num-1) + [archs[block_out_idx]]
        idx += 1
    extended_archs = torch.tensor(extended_archs).to(archs.device)
    return extended_archs

  candidates = [int(v) for v in args.candidate_width.split(',')]
  # obtain full archs (expand blockwise archs)
  if args.blockwise:
    full_archs = extend_blockwise_archs(archs)
  else:
    full_archs = archs

  # determine candidates according to archs
  accum_block_layer = [sum(num_blocks[:idx+1])*block_layer_num+1 for idx in range(len(num_blocks))]
  nb_layers = len(full_archs)
  expand_candidates = []
  candidates_list = [candidates] * nb_layers
  expand_times = 1.

  for layer_idx, cands in enumerate(candidates_list):
    if layer_idx in accum_block_layer:
      expand_times *= args.multiplier

    expand_cand = [int(v * expand_times) for v in cands]
    if block_layer_num == 3 and layer_idx % 3 == 0 and layer_idx > 0:
      # for Bottleneck, expansion = 4 for output of each block
      expand_cand = [v*4 for v in expand_cand]
    expand_candidates.append(expand_cand)

  cfgs = []
  for arch, cands in zip(full_archs, expand_candidates):
    cfgs.append(int(cands[arch]))
  return cfgs


def transfer_archs_mobilev2(args, archs):
  # NOTE: hard code this part
  def obtain_full_archs(archs):
    if args.blockwise:
        # make extend_archs matches blockwise=False
        extend_archs = []
        dc_used = 0
        block_id = 0
        for layer_id, arch in enumerate(archs):
            if layer_id == 0:
                extend_archs += [arch]*2
                dc_used += 1
            elif layer_id == num_search_layers - 1:
                # the last c, directly put in
                extend_archs += [arch]
            elif dc_used == num_blocks[block_id]:
                # arrive at ci==co
                block_id += 1
                dc_used = 0
                cio = arch
            else:
                # add dc
                extend_archs += [arch, cio]
                dc_used += 1
        extend_archs = torch.stack(extend_archs)
    else:
        extend_archs = archs

    # obtain the full arch
    curr_search_layers = len(extend_archs)
    full_archs = []
    for idx, arch in enumerate(extend_archs):
        if idx % 2 == 0 and idx < extended_search_layers - 2:
            full_archs.append(arch)
            full_archs.append(arch)
        else:
            full_archs.append(arch)
    full_archs = torch.stack(full_archs)
    return full_archs

  def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
      min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
      new_v += divisor
    return new_v

  assert args.multiplier == 1.0, "For mobilenet, only use multiplier == 1.0"
  num_blocks = [1,2,3,4,3,3,1]
  extended_search_layers = sum(num_blocks)*2 + 1
  num_search_layers = sum(num_blocks) + len(num_blocks) if args.blockwise else extended_search_layers
  assert extended_search_layers == 35

  block_layer_num = 3
  candidates = [int(v) for v in args.candidate_width.split(',')]
  full_archs = obtain_full_archs(archs) # blockwise == True is included inside

  # determine outc channels of mobile blocks
  accum_block_layer = [5, 11, 20, 32, 41, 50, 51] # expand candidate width
  change_expand_layer = [3, 9, 18, 30, 39, 48] # positions for depthwise conv expansion
  change_epand_idx = 0

  expansions = [3, 3, 3, 6, 6, 6]
  multi_list = [2., 1.5, 1.3333, 2., 1.5, 1.6667, 2., 4] # expand ratio, default

  nb_layers = len(full_archs)

  candidates_list = [candidates] * nb_layers
  expand_times = 1.
  expand_idx = 1
  expand_candidates = []

  for layer_idx, cands in enumerate(candidates_list):
      if layer_idx <= 1:
          # the first conv, candidate_width x 2
          expand_cand = [v * multi_list[0] for v in cands]
          expand_cand = [_make_divisible(v, 2) for v in expand_cand]
          expand_candidates.append(expand_cand)
          continue

      if layer_idx in accum_block_layer:
          expand_times *= multi_list[expand_idx]
          expand_idx += 1

      if layer_idx in change_expand_layer:
          expansion = expansions[change_epand_idx]
          change_epand_idx += 1

      expand_cand = [_make_divisible(v*expand_times, 2) for v in cands]

      if layer_idx % 3 in [0,1] and layer_idx != nb_layers - 1:
          # inc and outc of depthwise expansion layer
          expand_cand = [v * expansion for v in expand_cand]

      expand_candidates.append(expand_cand)

  assert len(expand_candidates) == nb_layers

  cfgs = []
  for arch, cands in zip(full_archs, expand_candidates):
    cfgs.append(int(cands[arch]))
  return cfgs


def decode_cfg(args, archs, num_blocks=None, block_layer_num=None):
  if args.model_type in ['resnet_20_aux', 'resnet_56_aux']:
    return transfer_archs_res20s(args, archs)
  elif args.model_type in ['resnet_18_width', 'resnet_50_width']:
    return transfer_archs_res18s(args, archs, num_blocks, block_layer_num)
  elif args.model_type == 'mobilenet_v2_width':
    return transfer_archs_mobilev2(args, archs)

