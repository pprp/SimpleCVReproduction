import os
import random

import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class TripleFolder(datasets.ImageFolder):
    def __init__(self, data_root, transforms=None):
        super(TripleFolder, self).__init__(data_root, transforms)
        # labels such as 1,2,3 ..
        self.targets = np.asarray([s[1] for s in self.samples])

    def get_pos_sample(self, target, index):
        # 找到与目标一致的下角标
        pos_index = np.argwhere(self.targets == target)
        pos_index = pos_index.flatten()
        # 去掉本身
        pos_index = np.setdiff1d(pos_index, index)
        # array([5, 0, 3, 2, 9, 8, 1, 6, 7, 4]) 随机index
        rand = np.random.permutation(len(pos_index))
        same_id_path = []
        # 从以上随机选取四个图作为pos
        for i in range(4):
            idx = i % len(rand)
            tmp_index = pos_index[rand[idx]]
            same_id_path.append(self.samples[tmp_index][0])
        return same_id_path

    def get_neg_sample(self, target):
        # 找到与目标不一致的目标
        neg_index = np.argwhere(self.targets != target)
        neg_index = neg_index.flatten()
        rand = random.randint(0, len(neg_index) - 1)
        return self.samples[neg_index[rand]]

    def __getitem__(self, index):
        # target is label such as 0,1,2,3
        path, target = self.samples[index]

        pos_path = self.get_pos_sample(target, index)

        sample = self.loader(path)
        pos0 = self.loader(pos_path[0])
        pos1 = self.loader(pos_path[1])
        pos2 = self.loader(pos_path[2])
        pos3 = self.loader(pos_path[3])

        if self.transform is not None:
            sample = self.transform(sample)
            pos0 = self.transform(pos0)
            pos1 = self.transform(pos1)
            pos2 = self.transform(pos2)
            pos3 = self.transform(pos3)

        c, h, w = pos0.shape

        pos_list = torch.cat(
            (pos0.view(1, c, h, w),
            pos1.view(1, c, h, w),
            pos2.view(1, c, h, w),
            pos3.view(1, c, h, w)), 0)
        return sample, target, pos_list

if __name__ == "__main__":
    train_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    datasets = TripleFolder(
        r'D:\GitHub\SimpleCVReproduction\simple-triple-loss\Market\train', train_transforms)

    # a,b,c = TripleFolder[0]
    dataloaders = DataLoader(datasets, batch_size=2, shuffle=True, num_workers=0)
    for a, b , c in dataloaders:
        print(a.shape,b.shape,c.shape)
