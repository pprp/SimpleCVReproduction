from nets.resnet_20s import *
from nets.resnet_20s_decode import *
from nets.resnet_20s_width import *
from nets.resnet_18s import *
from nets.resnet_18s_width import *
from nets.resnet_18s_decode import *
from nets.mobilenet_v2 import *
from nets.mobilenet_v2_width import *
from nets.mobilenet_v2_decode import *
from utils.compute_flops import *
import torch
import pdb


class ModelHelper(object):
  def __init__(self):
    self.model_dict = {'resnet_20': resnet20,\
        'resnet_32': resnet32,\
        'resnet_44': resnet44,\
        'resnet_56': resnet56,\
        'resnet_110': resnet110,\
        'resnet_1202': resnet1202,\
        'resnet_18': resnet18,\
        'resnet_34': resnet34,\
        'resnet_50': resnet50,\
        'resnet_20_width': resnet20_width, \
        'resnet_56_width': resnet56_width, \
        'resnet_18_width': resnet18_width, \
        'resnet_50_width': resnet50_width, \
        'resnet_decode': resnet_decode,\
        'resnet_12_decode': resnet12_decode, \
        'resnet_14_decode': resnet14_decode, \
        'resnet_18_decode': resnet18_decode, \
        'resnet_50_decode': resnet50_decode, \
        'mobilenet_v2': mobilenet_v2, \
        'mobilenet_v2_width': mobilenet_v2_width, \
        'mobilenet_v2_decode': mobilenet_v2_decode}

  def get_model(self, args):
    if  args.model_type not in self.model_dict.keys():
      raise ValueError('Wrong model type.')

    num_classes = self.__get_number_class(args.dataset)

    if 'decode' in args.model_type:
      if args.cfg == '':
        raise ValueError('Running decoding model. Empty cfg!')
      cfg = [int(v) for v in args.cfg.split(',')]
      model = self.model_dict[args.model_type](cfg, num_classes, args.se, args.se_reduction)
      if args.rank == 0:
        print(model)
        print("Flops and params:")
        resol = 32 if args.dataset == 'cifar10' else 224
        print_model_param_nums(model)
        print_model_param_flops(model, resol, multiply_adds=False)

    elif 'resnet' in args.model_type and \
        ('search' in args.model_type \
        or 'width' in args.model_type):
      # need to pass extra args for the one-shot model
      model = self.model_dict[args.model_type](num_classes, \
          args.candidate_width, \
          args.max_width, args=args)

    elif args.model_type.startswith('resnet'):
      model = self.model_dict[args.model_type](num_classes)

    elif args.model_type == 'mobilenet_v2':
        model = self.model_dict[args.model_type](dropout_rate=args.dropout_rate)

    elif args.model_type == 'mobilenet_v2_width':
      # dropout rate is included inside args. and configured inside.
      model = self.model_dict[args.model_type](base_width=args.candidate_width, \
          base_max_width=args.max_width, \
          args=args)

    return model

  def __get_number_class(self, dataset):
    # determine the number of classes
    if dataset == 'cifar10':
      num_classes = 10
    elif dataset == 'cifar100':
      num_classes = 100
    elif dataset == 'ilsvrc_12':
      num_classes = 1000
    return num_classes

def test():
  mh = ModelHelper()
  for k in mh.model_dict.keys():
    print(mh.get_model(k))


if __name__ == '__main__':
  test()

