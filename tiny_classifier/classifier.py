import os

import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model import TinyModel,DeepTinyModel,ShallowTinyModel
from focal_loss import FocalLoss
from torch.autograd import Variable

def train(model, dataloader, epoch, optimizer, criterion):
    model.train()
    for itr, (data, label) in enumerate(dataloader):
        output = model(data)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if itr % 10 == 0:
            print("epoch:%2d|step:%04d|loss:%.6f" % (epoch, itr, loss.item()))


def test(model, dataloader: torchvision.datasets.ImageFolder,
         criterion):
    model.eval()
    total_num = len(dataloader.dataset)
    right_num = 0
    for itr, (data, label) in enumerate(dataloader):
        outputs = model(data)
        loss = criterion(outputs, label)

        bs = data.size()[0]
        output = outputs.topk(1, dim=1)[1].view(bs, 1)

        diff = (output != label)
        diff = diff.sum(1)

        diff = (diff != 0)

        res = diff.sum(0).item()
        right_num += (bs - res)

    print("Test acc: %.3f" % (float(right_num) / total_num))
    return float(right_num) / total_num

if __name__ == "__main__":
    train_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    test_trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(1, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_datasets = ImageFolder(os.path.join("ROI_data", "train"),
                                 transform=train_trans)
    test_datasets = ImageFolder(os.path.join("ROI_data", "test"),
                                transform=test_trans)

    train_dataloader = torch.utils.data.DataLoader(train_datasets,
                                                   batch_size=32,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_datasets,
                                                  batch_size=32,
                                                  shuffle=False)

    total_epoch = 100
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    # criterion = torch.nn.CrossEntropyLoss()
    criterion = FocalLoss(2, alpha=torch.Tensor([1,1]), gamma=3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=30,
                                                gamma=0.1)

    best_acc = 0.

    for epoch in range(total_epoch):
        train(model, train_dataloader, epoch, optimizer, criterion)
        acc=test(model, test_dataloader, criterion)
        best_acc = max(best_acc, acc)
        scheduler.step()
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "checkpoints/epoch_%d_%.3f.pt" % (epoch, acc))

    print("best accuracy:%.3f" % best_acc)