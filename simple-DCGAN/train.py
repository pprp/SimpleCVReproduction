import os
import argparse
import tqdm
import sys

import torch
import torchvision as tv
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets

from config import opt
from model import NetD, NetG
from torchnet.meter import AverageValueMeter


def train():
    # change opt
    # for k_, v_ in kwargs.items():
    #     setattr(opt, k_, v_)

    device = torch.device('cuda') if torch.cuda.is_available else torch.device(
        'cpu')

    if opt.vis:
        from visualizer import Visualizer
        vis = Visualizer(opt.env)

    # rescale to -1~1
    transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(opt.data_path, transform=transform)

    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            shuffle=True,
                            num_workers=opt.num_workers,
                            drop_last=True)

    netd = NetD(opt)
    netg = NetG(opt)
    map_location = lambda storage, loc: storage
    if opt.netd_path:
        netd.load_state_dict(torch.load(opt.netd_path),
                             map_location=map_location)
    if opt.netg_path:
        netg.load_state_dict(torch.load(opt.netg_path),
                             map_location=map_location)

    if torch.cuda.is_available():
        netd.to(device)
        netg.to(device)

    # 定义优化器和损失
    optimizer_g = torch.optim.Adam(netg.parameters(),
                               opt.lr1,
                               betas=(opt.beta1, 0.999))
    optimizer_d = torch.optim.Adam(netd.parameters(),
                                   opt.lr2,
                                   betas=(opt.beta1, 0.999))

    criterion = torch.nn.BCELoss().to(device)

    # 真label为1， noises是输入噪声
    true_labels = Variable(torch.ones(opt.batch_size))
    fake_labels = Variable(torch.zeros(opt.batch_size))

    fix_noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
    noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))

    errord_meter = AverageValueMeter()
    errorg_meter = AverageValueMeter()

    if torch.cuda.is_available():
        netd.cuda()
        netg.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()

    for epoch in range(opt.max_epoch):
        print("epoch:",epoch, end='\r')
        # sys.stdout.flush()
        for ii, (img, _) in enumerate(dataloader):
            real_img = Variable(img)
            if torch.cuda.is_available():
                real_img = real_img.cuda()

            # 训练判别器, real -> 1, fake -> 0
            if (ii + 1) % opt.d_every == 0:
                # real
                optimizer_d.zero_grad()
                output = netd(real_img)
                # print(output.shape, true_labels.shape)
                error_d_real = criterion(output, true_labels)
                error_d_real.backward()
                # fake
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises).detach()  # 随机噪声生成假图
                fake_output = netd(fake_img)
                error_d_fake = criterion(fake_output, fake_labels)
                error_d_fake.backward()
                # update optimizer
                optimizer_d.step()

                error_d = error_d_fake + error_d_real

                errord_meter.add(error_d.item())

            # 训练生成器, 让生成器得到的图片能够被判别器判别为真
            if (ii + 1) % opt.g_every == 0:
                optimizer_g.zero_grad()
                noises.data.copy_(torch.randn(opt.batch_size, opt.nz, 1, 1))
                fake_img = netg(noises)
                fake_output = netd(fake_img)
                error_g = criterion(fake_output, true_labels)
                error_g.backward()
                optimizer_g.step()

                errorg_meter.add(error_g.item())

            if opt.vis and ii % opt.plot_every == opt.plot_every - 1:
                # 进行可视化
                # if os.path.exists(opt.debug_file):
                #     import ipdb
                #     ipdb.set_trace()

                fix_fake_img = netg(fix_noises)
                vis.images(fix_fake_img.detach().cpu().numpy()[:opt.batch_size] * 0.5 +
                           0.5,
                           win='fixfake')
                vis.images(real_img.data.cpu().numpy()[:opt.batch_size] * 0.5 + 0.5,
                           win='real')
                vis.plot('errord', errord_meter.value()[0])
                vis.plot('errorg', errorg_meter.value()[0])

        if (epoch + 1) % opt.save_every == 0:
            # 保存模型、图片
            tv.utils.save_image(fix_fake_img.data[:opt.batch_size],
                                '%s/%s.png' % (opt.save_path, epoch),
                                normalize=True,
                                range=(-1, 1))
            torch.save(netd.state_dict(), 'checkpoints/netd_%s.pth' % epoch)
            torch.save(netg.state_dict(), 'checkpoints/netg_%s.pth' % epoch)
            errord_meter.reset()
            errorg_meter.reset()


if __name__ == "__main__":
    train()
