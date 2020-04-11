import glob
import os

import cv2
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class KeyPointDatasets(Dataset):
    def __init__(self, root_dir="./data", transforms=None):
        super(KeyPointDatasets, self).__init__()
        self.img_path = os.path.join(root_dir, "images")
        # self.txt_path = os.path.join(root_dir, "labels")

        self.img_list = glob.glob(os.path.join(self.img_path, "*.jpg"))
        self.txt_list = [item.replace(".jpg", ".txt").replace(
            "images", "labels") for item in self.img_list]

        if transforms is not None:
            self.transforms = transforms

    def __getitem__(self, index):
        img = self.img_list[index]
        txt = self.txt_list[index]

        img = cv2.imread(img)

        if self.transforms:
            img = self.transforms(img)

        label = []

        with open(txt, "r") as f:
            for i, line in enumerate(f):
                if i == 0:
                    # 第一行
                    num_point = int(line.strip())
                else:
                    x1, y1 = [(t.strip()) for t in line.split()]
                    # range from 0 to 1
                    x1, y1 = float(x1), float(y1)

                    tmp_label = (x1, y1)
                    label.append(tmp_label)

        return img, torch.tensor(label[0])

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collect_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs, 0), torch.stack(labels, 0)


if __name__ == "__main__":
    trans = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((480, 360)),
        transforms.ToTensor(),
    ])
    kp_datasets = KeyPointDatasets(
        root_dir="./data", transforms=trans)

    # for i in range(len(kp_datasets)):
    # print(kp_datasets[i][0].shape, kp_datasets[i][1])

    data_loader = DataLoader(kp_datasets, num_workers=0, batch_size=4, shuffle=True,
                             collate_fn=kp_datasets.collect_fn
                             )

    for data, label in data_loader:
        print(data.shape, label)
