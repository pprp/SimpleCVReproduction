import numpy as np
import torch

__all__ = ["DataProvider"]


class DataProvider:
    SEED = 981007

    @property
    def data_shape(self):
        """Return shape of image data"""
        raise NotImplementedError

    @property
    def n_classes(self):
        """Return number of classes"""
        raise NotImplementedError

    @property
    def save_path(self):
        """Local path to save the data"""
        raise NotImplementedError

    @property
    def data_url(self):
        """Link to download the data"""
        raise NotImplementedError

    @staticmethod
    def name():
        """return name of dataset"""
        raise NotImplementedError

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
