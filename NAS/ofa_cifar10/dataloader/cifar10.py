import torchvision.transforms as transforms
import torchvision.datasets as datasets

from dataprovider import DataProvider


class Cifar10DataProvider(DataProvider):

    def __init__(self, save_path=None, train_bs=256,
                 test_bs=256, valid_size=None, 
                 n_workers=8,
                 image_size=32):
        super(Cifar10DataProvider, self).__init__()

        self.save_path = save_path

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
