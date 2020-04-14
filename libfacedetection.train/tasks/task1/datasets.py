import os
import os.path as osp

import cv2
import numpy as np
import scipy.io

import torch
from torch.utils import data

class WIDERFace(data.Dataset):
    '''Dataset class for WIDER Face dataset'''

    def __init__(self, widerface_root, split='val', device=None):
        self.widerface_root = widerface_root
        self._split = split
        self.device = device

        self.widerface_img_paths = {
            'val':  osp.join(self.widerface_root, 'WIDER_val', 'images'),
            'test': osp.join(self.widerface_root, 'WIDER_test', 'images')
        }

        self.widerface_split_fpaths = {
            'val':  osp.join(self.widerface_root, 'wider_face_split', 'wider_face_val.mat'),
            'test': osp.join(self.widerface_root, 'wider_face_split', 'wider_face_test.mat')
        }

        self.img_list, self.num_img = self.load_list()

    def load_list(self):
        n_imgs = 0
        flist = []

        split_fpath = self.widerface_split_fpaths[self._split]
        img_path = self.widerface_img_paths[self._split]

        anno_data = scipy.io.loadmat(split_fpath)
        event_list = anno_data.get('event_list')
        file_list = anno_data.get('file_list')

        for event_idx, event in enumerate(event_list):
            event_name = event[0][ 0]
            for f_idx, f in enumerate(file_list[event_idx][0]):
                f_name = f[0][0]
                f_path = osp.join(img_path, event_name, f_name+'.jpg')
                flist.append(f_path)
                n_imgs += 1

        return flist, n_imgs

    def __getitem__(self, index):
        img = cv2.imread(self.img_list[index], cv2.IMREAD_COLOR).astype(np.float32)
        event, name = self.img_list[index].split('/')[-2:]

        if self.device is not None:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            return img_tensor, event, name

        return img, event, name

    def __len__(self):
        return self.num_img

    @property
    def size(self):
        return self.num_img

    @property
    def split(self):
        return self._split


if __name__ == '__main__':
    wf = WIDERFace('./widerface', 'val', 'cuda:1')
    print(wf.size, wf.split)
    print(wf[1430])