import numpy as np
import torch
from torchvision import transforms as T
from datasets.autoaugmentation import CIFAR10Policy
from torchvision.transforms import transforms

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
CIFAR100_STD = [0.1942, 0.1918, 0.1958]


# cutout
class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask

        return img

# mixup


def mixup_data(x, y, alpha=1.0):
    """return mixed inputs"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    bs = x.size()[0]
    
    index = torch.randperm(bs).cuda()

    mixed_x = lam * x + (1-lam) * x[index, :]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


class DatasetTransforms:
    def __init__(self, clss, cutout=0):
        if clss == 'cifar10':
            self.mean = CIFAR10_MEAN
            self.std = CIFAR10_STD
        elif clss == 'cifar100':
            self.mean = CIFAR100_MEAN
            self.std = CIFAR100_STD
        else:
            print("Not Support %s dataset." % clss)
        self.cutout = 0

    def _get_cutout(self):
        if self.cutout == 0:
            return None
        else:
            return Cutout(self.cutout)

    def _get_default_transforms(self):
        default_configure = T.Compose([
            T.RandomCrop(32, 4),
            T.RandomHorizontalFlip(),
            # T.RandomResizedCrop((32, 32)),  # for cifar10 or cifar100
            # T.RandomRotation(15)
        ])
        return default_configure

    def get_train_transforms(self, autoaug=False):
        default_conf = self._get_default_transforms()
        cutout = self._get_cutout()
        if cutout is None:
            train_transform = T.Compose([default_conf,
                                         CIFAR10Policy(),
                                         T.ToTensor(),
                                         T.Normalize(self.mean, self.std)
                                         ])
        else:
            train_transform = T.Compose([default_conf,
                                         T.ToTensor(),
                                         cutout,
                                         T.Normalize(self.mean, self.std)
                                         ])

        return train_transform

    def get_val_transform(self):
        val_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(self.mean, self.std)
        ])

        return val_transform
