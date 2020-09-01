import os

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import KeyPointDatasets
from model import KeyPointModel
from utils import Visualizer, compute_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# h, w
IMG_SIZE = 360, 480

vis = Visualizer(env="keypoint")


def train(model, epoch, dataloader, optimizer, criterion, scheduler):
    model.train()
    for itr, (image, hm) in enumerate(dataloader):
        if torch.cuda.is_available():
            hm = hm.cuda()
            image = image.cuda()

        bs = image.shape[0]

        output = model(image)

        hm = hm.float()

        loss = criterion(output, hm)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()

        if itr % 2 == 0:
            print("epoch:%2d|step:%04d|loss:%.6f" %
                  (epoch, itr, loss.item()/bs))
            vis.plot_many_stack({"train_loss": loss.item()/bs})


def test(model, epoch, dataloader, criterion):
    model.eval()
    sum_loss = 0.
    n_sample = 0
    for itr, (image, hm) in enumerate(dataloader):
        if torch.cuda.is_available():
            hm = hm.cuda()
            image = image.cuda()

        output = model(image)
        hm = hm.float()

        loss = criterion(output, hm)

        sum_loss += loss.item()
        n_sample += image.shape[0]

    print("TEST: epoch:%02d-->loss:%.6f" % (epoch, sum_loss/n_sample))
    if epoch > 1:
        vis.plot_many_stack({"test_loss": sum_loss/n_sample})
    return sum_loss / n_sample


if __name__ == "__main__":

    total_epoch = 100
    bs = 2
    ########################################
    transforms_all = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((360, 480)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4372, 0.4372, 0.4373],
                             std=[0.2479, 0.2475, 0.2485])
    ])

    datasets = KeyPointDatasets(root_dir="./data", transforms=transforms_all)

    data_loader = DataLoader(datasets, shuffle=True,
                             batch_size=bs, collate_fn=datasets.collect_fn)

    model = KeyPointModel()

    if torch.cuda.is_available():
        model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = torch.nn.MSELoss()  # compute_loss
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=30,
                                                gamma=0.1)

    for epoch in range(total_epoch):
        train(model, epoch, data_loader, optimizer, criterion, scheduler)
        loss = test(model, epoch, data_loader, criterion)

        if epoch % 5 == 0:
            torch.save(model.state_dict(),
                       "weights/epoch_%d_%.3f.pt" % (epoch, loss*10000))
