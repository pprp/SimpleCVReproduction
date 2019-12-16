import torch
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from config import opt
from model import NetD, NetG
from torch.autograd import Variable


def train(**kwargs):
    # change opt
    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device('cuda') if torch.cuda.is_available else torch.device(
        'cpu')

    if opt.vis:
        from visualizer import Visualizer
        vis = Visualizer(opt.env)

    # rescale to -1~1
    transform = transforms.Compose([
        transforms.Scale(opt.image_size),
        transforms.CenterCrop(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset = datasets.ImageFolder(opt.data_path, transforms=transform)
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

    netd.to(device)
    netg.to(device)

    # 定义优化器和损失
    optimizer_g = t.optim.Adam(netg.parameters(),
                               opt.lr1,
                               betas=(opt.beta1, 0.999))
    optimizer_d = t.optim.Adam(netd.parameters(),
                               opt.lr2,
                               betas=(opt.beta1, 0.999))
    criterion = t.nn.BCELoss().to(device)

    # 真label为1， noises是输入噪声
    true_labels = Variable(torch.ones(opt.batch_size))
    fake_labels = Variable(torch.zeros(opt.batch_size))

    fix_noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))
    noises = Variable(torch.randn(opt.batch_size, opt.nz, 1, 1))

    if torch.cuda.is_available():
        netd.cuda()
        netg.cuda()
        criterion.cuda()
        true_labels,fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()
    
    for ii, (img, _) in tqdm.tqdm(enumerate(dataloader)):
        real_img = Variable(img)
        if torch.cuda.is_available:
            real_img = real_img.cuda()
        
        if (ii+1)%opt.d_every == 0:
            optimizer_d.zero_grad()
            