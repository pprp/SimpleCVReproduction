import torch
from torchvision import datasets, transforms
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.distributed
from torch.autograd import Variable
import torch.nn.functional as F
import os
from torch.utils.data import DataLoader, Subset, sampler
import pickle
import pdb
import numpy as np
from PIL import Image


def cifar10_loader_search(args, num_workers=4):
  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                               std=[x/255.0 for x in [63.0, 62.1, 66.7]])
  if args.data_aug:
    with torch.no_grad():
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Lambda(lambda x: F.pad(
                                          Variable(x.unsqueeze(0), requires_grad=False),
                                          (4,4,4,4),mode='reflect').data.squeeze()),
      transforms.ToPILImage(),
      transforms.RandomCrop(32),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
      ])
      transform_test = transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              ])
  else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
  trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
  testset = datasets.CIFAR10(root=args.data_path, train=False, transform=transform_test)

  num_train = len(trainset)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train * args.total_portion))     # we now add a portion_total arg to just use smaller number of data
  split_upb = int(num_train * args.total_portion)

  train_loader = DataLoader(trainset, batch_size=args.batch_size,
                            sampler=sampler.SubsetRandomSampler(indices[:split]),
                            num_workers=num_workers, pin_memory=True)
  valid_loader = DataLoader(trainset, batch_size=args.batch_size,
                            sampler=sampler.SubsetRandomSampler(indices[split:split_upb]),
                            num_workers=num_workers, pin_memory=True)
  test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  return train_loader, valid_loader, test_loader

def cifar10_loader_train(args, num_workers=4):
  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                               std=[x/255.0 for x in [63.0, 62.1, 66.7]])
  if args.data_aug:
    with torch.no_grad():
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Lambda(lambda x: F.pad(Variable(x.unsqueeze(0), requires_grad=False),
                                          (4,4,4,4),mode='reflect').data.squeeze()),
      transforms.ToPILImage(),
      transforms.RandomCrop(32),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
      ])
      transform_test = transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              ])
  else:
      transform_train = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      transform_test = transforms.Compose([
          transforms.ToTensor(),
          transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])

  trainset = datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
  testset = datasets.CIFAR10(root=args.data_path, train=False, transform=transform_test)

  train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=True)
  test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False,
                           num_workers=num_workers, pin_memory=True)
  return train_loader, test_loader

def cifar100_loader(args, num_workers=4):
  normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                               std=[x/255.0 for x in [63.0, 62.1, 66.7]])
  if args.data_aug:
    with torch.no_grad():
      transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: F.pad(
                                        Variable(x.unsqueeze(0), requires_grad=False),
                                        (4,4,4,4),mode='reflect').data.squeeze()),
      transforms.ToPILImage(),
      transforms.RandomCrop(32),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
      ])
      transform_test = transforms.Compose([
                              transforms.ToTensor(),
                              normalize
                              ])
  else:
      transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
      transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
  trainset = datasets.CIFAR100(root=args.data_path, train=True, download=True, transform=transform_train)
  testset = datasets.CIFAR100(root=args.data_path, train=False, download=True, transform=transform_test)
  train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
  test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
  return train_loader, test_loader


def ilsvrc12_loader_train(args, num_workers=4):
  traindir = os.path.join(args.data_path, 'train')
  testdir = os.path.join(args.data_path, 'val')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])

  train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize,
    ]))

  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
  else:
    train_sampler = None

  train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size // args.world_size, shuffle=(train_sampler is None),
    num_workers=num_workers, pin_memory=True, sampler=train_sampler)

  test_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(testdir, transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ])),
    batch_size=args.batch_size // args.world_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)

  return [train_loader, test_loader], train_sampler

