import os
from typing import Tuple

import cv2
import numpy as np

from keras.utils import Sequence
from pycocotools import mask
from pycocotools.coco import COCO

from util.array_utils import down_size_batch


class DataLoader(Sequence):
    """Loads data/labels from a coco annotated dataset.
    The images are stored in the 'images/' directory and the annotations in the 'annotation.json' file.

    # Arguments:
        data_directory: string
            absolute path to the directory
        batch_size: int
            amount of data points which are processed as one batch
        shuffle: boolean
            Shuffle data points after each epoch
        augment: boolean
            augment the data points with random horizontal/vertical flips and 90° rotations
        down_sample_factor: int
            factor used for down sampling. The image size has to be divisible by the factor without remainder
        subset_size: int
            size of the subset which is used instead of the whole dataset
        random_state: numpy.random.RandomState
            state which will be used for shuffling and augmentation
    """

    def __init__(self: 'DataLoader',
                 data_directory: str,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augment: bool = True,
                 down_sample_factor: int = 1,
                 subset_size: int = 0,
                 random_state: np.random.RandomState = None):
        self.directory = os.path.join(data_directory, 'images')
        self.coco = COCO(os.path.join(data_directory, 'annotation-small.json'))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.down_sample_factor = down_sample_factor

        # Init random state
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

        # Get ids
        self.image_ids = self.coco.getImgIds()
        if 0 < subset_size < len(self.image_ids):
            self.image_ids = self.random_state.choice(self.image_ids, subset_size)

        # Get data shape
        self.height = self.coco.loadImgs(self.image_ids[0])[0]['height']
        self.width = self.coco.loadImgs(self.image_ids[0])[0]['width']
        self.data_shape = (self.height, self.width, 3)
        self.labels_shape = (self.height, self.width, 1)

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]

        data = np.empty((self.batch_size, *self.data_shape))
        labels = np.empty((self.batch_size, *self.labels_shape))

        for i, ID in enumerate(indexes):
            x = self._get_image(ID)
            y = self._get_labels(ID)

            if self.augment:
                rotations = self.random_state.randint(4)
                flip_lr = self.random_state.randint(2)
                flip_ud = self.random_state.randint(2)

                if flip_lr:
                    x = np.fliplr(x)
                    y = np.fliplr(y)

                if flip_ud:
                    x = np.flipud(x)
                    y = np.flipud(y)

                x = np.rot90(x, k=rotations)
                y = np.rot90(y, k=rotations)

            data[i,] = x
            labels[i,] = y

        if self.down_sample_factor > 1:
            data = down_size_batch(data)
            labels = down_size_batch(labels)

        return data, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state.shuffle(self.image_ids)

    def _get_image(self, image_id):
        path = self.coco.loadImgs([image_id])[0]['file_name']
        image = cv2.imread(os.path.join(self.directory, path)) / 255
        return image

    def _get_labels(self, image_id):
        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_id))
        labels = np.zeros((self.height, self.width), dtype=float)
        for a in annotations:
            rle = mask.frPyObjects(a['segmentation'], self.height, self.width)
            m = np.squeeze(mask.decode(rle))
            labels = np.logical_or(labels, m)
        return labels.reshape((*labels.shape, 1))


class NumpyDataLoader(Sequence):
    """Loads data/labels where each data point is stored as a numpy array (.npy) on the file system.
    The data points/labels have to be stored with a given prefix and incrementing numbers for
    identification, for example with the prefix 'x_':
    x_1.npy, x_2.npy, ...

    # Arguments:
        data_directory: string
            absolute path to the directory
        batch_size: int
            amount of data points which are processed as one batch
        data_shape: tuple
            shape of the data
        labels_shape: tuple
            shape  of the labels
        data_prefix: string
            prefix for the data files
        label_prefix: string
            prefix for the label files
        shuffle: boolean
            Shuffle data points after each epoch
        augment: boolean
            augment the data points with random horizontal/vertical flips and 90° rotations
        random_state: numpy.random.RandomState
            state which will be used for shuffling and augmentation
    """

    def __init__(self: 'NumpyDataLoader',
                 data_directory: str,
                 batch_size: int = 32,
                 data_prefix: str = 'x_',
                 label_prefix: str = 'y_',
                 shuffle: bool = True,
                 augment: bool = True,
                 random_state: np.random.RandomState = None):
        self.data_directory = data_directory
        self.batch_size = batch_size
        self.data_prefix = data_prefix
        self.label_prefix = label_prefix
        self.shuffle = shuffle
        self.augment = augment

        # Get amount of data points
        data_size = len([f for f in os.listdir(data_directory) if f.startswith(data_prefix)])
        self.image_ids = np.arange(data_size)

        # Get data/label dimensionality from first sample
        # Currently only one channeled labels are supported (binary segmentation)
        # Augmentation might crash when label shape is not 2d
        self.data_shape = np.load(data_directory + data_prefix + '0.npy').shape
        self.labels_shape = (*np.load(data_directory + label_prefix + '0.npy').shape, 1)

        # Init random state
        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

    def __len__(self):
        return int(np.floor(len(self.image_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.image_ids[index * self.batch_size:(index + 1) * self.batch_size]

        data = np.empty((self.batch_size, *self.data_shape))
        labels = np.empty((self.batch_size, *self.labels_shape))

        for i, ID in enumerate(indexes):
            x = np.load(self.data_directory + 'x_' + str(ID) + '.npy') / 255
            y = np.load(self.data_directory + 'y_' + str(ID) + '.npy').reshape(self.labels_shape)

            if self.augment:
                rotations = self.random_state.randint(4)
                flip_lr = self.random_state.randint(2)
                flip_ud = self.random_state.randint(2)

                if flip_lr:
                    x = np.fliplr(x)
                    y = np.fliplr(y)

                if flip_ud:
                    x = np.flipud(x)
                    y = np.flipud(y)

                x = np.rot90(x, k=rotations)
                y = np.rot90(y, k=rotations)

            data[i,] = x
            labels[i,] = y

        return data, labels

    def on_epoch_end(self):
        if self.shuffle:
            self.random_state.shuffle(self.image_ids)


class RandomLoader(Sequence):
    """Loads a random batch of data points with a given shape.
    This can be used for analyzing, like calculating the effective receptive field with random input.
    No labels are generated.

    # Arguments
        dataset_size: int
            desired amount of data points
        batch_size: int
            amount of data points which are processed as one batch
        data_shape: Tuple
            shape of the data points (channels last)
        random_state: np.random.RandomState
            state which will be used for data point generation
    """

    def __init__(self: 'RandomLoader',
                 dataset_size: int = 100,
                 batch_size: int = 32,
                 data_shape: Tuple = (150, 150, 3),
                 random_state: np.random.RandomState = None):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.data_shape = data_shape

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()
        self._state = self.random_state.get_state()  # Used for resetting random state after epoch

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        data = self.random_state.rand(self.batch_size, *self.data_shape)
        return data, None

    def on_epoch_end(self):
        self.random_state.set_state(self._state)
