import os
import time
import torch
import utils
import config
import torchvision
import torch.nn as nn
from thop import profile
from torchvision import datasets
from utils import data_transforms
from model import SinglePath_Network, train, validate
from torchsummary import summary


def main():
    # args & device
    args = config.get_args()
    if torch.cuda.is_available():
        print('Train on GPU!')
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # dataset
    assert args.dataset in ['cifar10', 'imagenet']
    train_transform, valid_transform = data_transforms(args)
    if args.dataset == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=True,
                                                download=True, transform=train_transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                   shuffle=True, pin_memory=True, num_workers=8)
        valset = torchvision.datasets.CIFAR10(root=os.path.join(args.data_dir, 'cifar'), train=False,
                                              download=True, transform=valid_transform)
        val_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                 shuffle=False, pin_memory=True, num_workers=8)
    elif args.dataset == 'imagenet':
        train_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'train'), train_transform)
        val_data_set = datasets.ImageNet(os.path.join(args.data_dir, 'ILSVRC2012', 'valid'), valid_transform)
        train_loader = torch.utils.data.DataLoader(train_data_set, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=8, pin_memory=True, sampler=None)
        val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=8, pin_memory=True)

    # SinglePath_OneShot
    choice = [2, 0, 2, 3, 2, 2, 3, 1, 2, 1, 0, 1, 0, 3, 1, 0, 0, 2, 3, 2]
    model = SinglePath_Network(args.dataset, args.resize, args.classes, args.layers, choice)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, args.momentum, args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - (epoch / args.epochs))

    # flops & params & structure
    flops, params = profile(model, inputs=(torch.randn(1, 3, 32, 32),) if args.dataset == 'cifar10'
                            else (torch.randn(1, 3, 224, 224),), verbose=False)
    # print(model)
    print('Random Path of the Supernet: Params: %.2fM, Flops:%.2fM' % ((params / 1e6), (flops / 1e6)))
    model = model.to(device)
    summary(model, (3, 32, 32) if args.dataset == 'cifar10' else (3, 224, 224))

    # train supernet
    start = time.time()
    for epoch in range(args.epochs):
        train(args, epoch, train_loader, device, model, criterion, optimizer, scheduler, supernet=False)
        scheduler.step()
        if (epoch + 1) % args.val_interval == 0:
            validate(args, epoch, val_loader, device, model, criterion, supernet=False)
            utils.save_checkpoint({'state_dict': model.state_dict(), }, epoch + 1, tag=args.exp_name)
    utils.time_record(start)


if __name__ == '__main__':
    main()
