"""
2019.9.26

Ruosi Wan:
    This script is for simulating reading images from imagefolds by noris.
    Will be removed when test finished!
"""


import nori2 as nori
import numpy as np
import torch.utils.data as data
import cv2
from PIL import Image

class ImageNetdst(data.Dataset):
    def __init__(self, nori_list_dir, transform=None):
        self.name_list = []
        self.transform = transform
        with open(nori_list_dir) as fp:
            for line in fp.readlines():
                self.name_list.append(line.split())

        self.nori_f = nori.Fetcher()

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        nori_id = self.name_list[idx][0]
        label = int(self.name_list[idx][1])
        raw = self.nori_f.get(nori_id)
        img = np.fromstring(raw, np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        img = img[:,:,::-1] #BGR2RGB
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, label
