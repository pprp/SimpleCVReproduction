import os
import numpy as np
import torch
import torch.nn as nn
import cv2
import random
import PIL
from PIL import Image
from torch.utils.data import Sampler
import torchvision.transforms as transforms
import math
import torchvision.datasets as datasets

## data augmentation functions
class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:,::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:,::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img


class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:,::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img


class RandomResizedCrop(object):

    def __init__(self, scale=(0.08, 1.0), target_size:int=224, max_attempts:int=10):
        assert scale[0] <= scale[1]
        self.scale = scale
        assert target_size > 0
        self.target_size = target_size
        assert max_attempts >0
        self.max_attempts = max_attempts

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.uint8)
        H, W, C = img.shape
        
        well_cropped = False
        for _ in range(self.max_attempts):
            crop_area = (H*W) * random.uniform(self.scale[0], self.scale[1])
            crop_edge = round(math.sqrt(crop_area))
            dH = H - crop_edge
            dW = W - crop_edge
            crop_left = random.randint(min(dW, 0), max(dW, 0))
            crop_top = random.randint(min(dH, 0), max(dH, 0))
            if dH >= 0 and dW >= 0:
                well_cropped = True
                break
        
        crop_bottom = crop_top + crop_edge
        crop_right = crop_left + crop_edge
        if well_cropped:
            crop_image = img[crop_top:crop_bottom,:,:][:,crop_left:crop_right,:]
            
        else:
            roi_top = max(crop_top, 0)
            padding_top = roi_top - crop_top
            roi_bottom = min(crop_bottom, H)
            padding_bottom = crop_bottom - roi_bottom
            roi_left = max(crop_left, 0)
            padding_left = roi_left - crop_left
            roi_right = min(crop_right, W)
            padding_right = crop_right - roi_right

            roi_image = img[roi_top:roi_bottom,:,:][:,roi_left:roi_right,:]
            crop_image = cv2.copyMakeBorder(roi_image, padding_top, padding_bottom, padding_left, padding_right,
                borderType=cv2.BORDER_CONSTANT, value=0)
            
        random.choice([1])
        target_image = cv2.resize(crop_image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        target_image = PIL.Image.fromarray(target_image.astype('uint8'))
        return target_image


class LighteningJitter(object):

    def __init__(self, eigen_vecs, eigen_values, max_eigen_jitter=0.1):

        self.eigen_vecs = np.array(eigen_vecs, dtype=np.float32)
        self.eigen_values = np.array(eigen_values, dtype=np.float32)
        self.max_eigen_jitter = max_eigen_jitter

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.float32)
        img = np.ascontiguousarray(img/255)

        cur_eigen_jitter = np.random.normal(scale=self.max_eigen_jitter, size=self.eigen_values.shape)
        color_purb = (self.eigen_vecs @ (self.eigen_values * cur_eigen_jitter)).reshape([1, 1, -1])
        img += color_purb
        img = np.ascontiguousarray(img*255)
        img.clip(0, 255, out=img)
        img = PIL.Image.fromarray(np.uint8(img))
        return img

class RandomHorizontalFlip(object):

    def __init__(self, prob=0.5):

        self.prob=prob

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img, dtype=np.uint8)
        p = random.uniform(0., 1.)
        if p <= self.prob:
            img1 = cv2.flip(img, 1)
        else:
            img1 = img
        img1 = PIL.Image.fromarray(img1)
        return img1

class Random_Batch_Sampler(Sampler):

    def __init__(self, dataset, batch_size, total_iters, rank=None):
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")

        self.dataset_num = dataset.__len__()
        self.rank = rank
        self.batch_size = batch_size
        self.total_iters = total_iters


    def __iter__(self):
        random.seed(self.rank)
        for i in range(self.total_iters):
            batch_iter = []
            for _ in range(self.batch_size):
                batch_iter.append(random.randint(0, self.dataset_num-1))
            
            yield batch_iter

    def __len__(self):
        return self.total_iters

def get_train_dataloader(train_dir, batch_size, local_rank=None, total_iters=None, shuffle=False):
    eigvec = np.array([
        [-0.5836, -0.6948,  0.4203],
        [-0.5808, -0.0045, -0.8140],
        [-0.5675,  0.7192,  0.4009]
    ])
    eigval = np.array([0.2175, 0.0188, 0.0045])

    print('='*100)

    train_dataset = datasets.ImageFolder(train_dir,
        transforms.Compose([
            RandomResizedCrop(target_size=224, scale=(0.08, 1.0)),
            LighteningJitter(eigen_vecs=eigvec[::-1,:], eigen_values=eigval, max_eigen_jitter=0.1),
            transforms.RandomHorizontalFlip(0.5),
            ToBGRTensor()
        ])
    )

    if not shuffle: # TODO 
        datasampler = Random_Batch_Sampler(
            train_dataset, batch_size=batch_size,
            total_iters=total_iters, rank=local_rank)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=8,
            pin_memory=True, batch_sampler=datasampler)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=8,
            pin_memory=True, batch_size=batch_size, shuffle=shuffle)

    return train_loader

def get_val_dataloader(val_dir, batch_size=200):
    val_dataset = datasets.ImageFolder(val_dir, 
        transforms.Compose([
                OpencvResize(256),
                transforms.CenterCrop(224),
                ToBGRTensor(),
            ]))
    print('batch_size={}'.format(batch_size))
    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=8, pin_memory=True
        )

    return val_loader 