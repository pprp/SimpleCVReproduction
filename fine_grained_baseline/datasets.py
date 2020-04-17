import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torchvision


class CubDataset(Dataset):
    def __init__(self, root_dir="./data", trainable=True, transform=None):
        super(CubDataset, self).__init__()
        self.img_list = []
        self.root_dir = root_dir
        self.transform = transform

        if trainable:
            self.label_path = os.path.join(root_dir, "lists", "train.txt")
        else:
            self.label_path = os.path.join(root_dir, "lists", "test.txt")

        f = open(self.label_path, "r")

        for line in f:
            line = line.strip('\n')
            line = line.rstrip()
            clss = line.split(".")
            self.img_list.append([line, int(clss[0])-1])

        f.close()

    def __getitem__(self, index):
        img_name, clss = self.img_list[index]
        img_path = os.path.join(self.root_dir, "images/images", img_name)
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor([clss])

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collect_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.stack(labels, 0)


if __name__ == "__main__":
    ds = CubDataset(transform=transforms.ToTensor())
    print(len(ds))
    torchvision.datasets.ImageFolder