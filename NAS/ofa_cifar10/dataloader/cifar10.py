import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.transforms.transforms import Resize
import torchvision

from dataprovider import DataProvider
from distributedsampler import OFADistributedSampler
import torch


class Cifar10DataProvider(DataProvider):

    def __init__(self, save_path=None, train_bs=256,
                 valid_bs=256, valid_size=None,
                 num_workers=8,
                 image_size=32,
                 num_replicas=None,
                 rank=None):
        super(Cifar10DataProvider, self).__init__()

        self.save_path = save_path
        MEAN = [0.5071, 0.4865, 0.4409]
        STD = [0.1942, 0.1918, 0.1958]

        # transforms
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
        valid_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(MEAN, STD)
        ])

        # datasets
        train_dataset = datasets.CIFAR10(
            self.save_path, train=True, download=True, transform=train_transform)
        valid_dataset = datasets.CIFAR10(
            self.save_path, train=False, download=True, transform=valid_transform)

        # samplers
        train_indices, valid_indices = self.random_sample_valid_set(
            len(train_dataset), valid_size)
        if num_replicas is not None:
            train_sampler = OFADistributedSampler(
                train_dataset, num_replicas, rank, True, np.array(train_indices))
            valid_sampler = OFADistributedSampler(
                valid_dataset, num_replicas, rank, True, np.array(valid_indices))
        else:
            train_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                train_indices)
            valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(
                valid_indices)

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, num_workers=num_workers, pin_memory=True, sampler=train_sampler, batch_size=train_bs, drop_last=True)
        self.valid_loader = torch.utils.data.DataLoader(
            valid_dataset, num_workers=num_workers, pin_memory=True, sampler=valid_sampler, batch_size=valid_bs)

    @property
    def data_shape(self):
        """Return shape of image data"""
        return 3, 32, 32

    @property
    def n_classes(self):
        """Return number of classes"""
        return 10

    @property
    def save_path(self):
        """Local path to save the data"""
        raise self.save_path

    @property
    def data_url(self):
        """Link to download the data"""
        raise ValueError('unable to download %s' % self.name())

    @staticmethod
    def name():
        """return name of dataset"""
        return "cifar10"

    @staticmethod
    def labels_to_one_hot(n_classes, labels):
        """labels shape [batchsize, classlbl]"""
        oh = np.zeros((labels.shape[0], n_classes), dtype=np.float32)
        new_labels[range(labels.shape[0]), labels] = np.ones(labels.shape)
        return new_labels

    @staticmethod
    def random_sample_valid_set(train_size, valid_size):
        g = torch.Generaotr()
        g.manual_seed(DataProvider.SEED)

        rand_indexes = torch.randperm(train_size, generator=g).tolist()

        valid_indexes = rand_indexes[:valid_size]
        train_indexes = rand_indexes[valid_size:]
        return train_indexes, valid_indexes
