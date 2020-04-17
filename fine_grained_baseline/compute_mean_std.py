import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import CubDataset

train_ds = CubDataset(transform=transforms.ToTensor())


train_dataloader = DataLoader(dataset=train_ds, batch_size=1)

if __name__ == "__main__":
    mean = 0.
    std = 0.
    n_sample = 0

    for img, id in train_dataloader:
        bs = img.size(0)
        img = img.view(bs, img.size(1), -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
        n_sample += bs 
    
    print("mean: {}".format(mean/n_sample))
    print("std: {}".format(std/n_sample))

    