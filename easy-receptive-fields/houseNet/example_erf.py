import configparser
import os

import cv2
import numpy as np

from analysis.effective_receptive_field import get_effective_receptive_field
from networks.dnet import DNet
from util.data_loader import DataLoader

"""Example of calculating the effective receptive field and visualizing it with opencv.
"""


def prepare_receptive_field(receptive_field: np.ndarray,
                            up_scale_factor: int = 1,
                            grey_scale: bool = False):
    """Generates an numpy array which can be visualized or saved with cv2 functions.

    # Arguments:
        receptive_field: Numpy array
            array which represents the effective receptive field (channels last)
        up_scale_factor: int
            factor to use for up-scaling
        grey_scale: bool
            convert array to gray values (mean over all channels)

    # Returns:
        A Numpy array
    """

    shape = receptive_field.shape

    shape_up = (shape[0] * up_scale_factor, shape[1] * up_scale_factor)
    image = cv2.resize(receptive_field, shape_up)

    if grey_scale:
        image = image.mean(axis=2)
    return image


def visualize_receptive_field(receptive_field, up_scale_factor=1, grey_scale=False):
    """Shows the effective receptive field

    # Arguments:
        receptive_field: Numpy array
            array which represents the effective receptive field (channels last)
        up_scale_factor: int
            factor to use for up-scaling
        grey_scale: bool
            convert array to gray values (mean over all color channels)
    """

    image = prepare_receptive_field(receptive_field, up_scale_factor, grey_scale)
    cv2.imshow('Effective Receptive Field', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.pardir, 'config.ini'))
    path_to_dataset = config['DIRECTORIES']['val_data']
    data_loader = DataLoader(data_directory=path_to_dataset,
                             batch_size=1,
                             augment=False,
                             down_sample_factor=2)

    net = DNet()
    rf = get_effective_receptive_field(net, data_loader=data_loader)
    visualize_receptive_field(rf, up_scale_factor=4)
