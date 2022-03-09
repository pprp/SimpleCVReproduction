import os

import cv2

import numpy as np


def down_size_image(image: np.ndarray):
    """Halves image by deleting every other row/column.

    # Arguments:
        image: np.ndarray
            array with shape (height, width, channels)

    # Returns:
        numpy array
    """
    image = np.delete(image, list(range(1, image.shape[0], 2)), axis=0)
    image = np.delete(image, list(range(1, image.shape[1], 2)), axis=1)
    return image


def down_size_batch(batch: np.ndarray):
    """Halves batch of images by deleting every other row/column.

        # Arguments:
            image: np.ndarray
                array with shape (batch_size, height, width, channels)

        # Returns:
            numpy array
    """
    batch = np.delete(batch, list(range(1, batch.shape[1], 2)), axis=1)
    batch = np.delete(batch, list(range(1, batch.shape[2], 2)), axis=2)
    return batch


def image_as_array(images_path, image_annotation):
    """Loads an images as a numpy array from disk.

    # Arguments:
        images_path: str
            path to images dataset
        image_annotation: obj
            coco annotated object which contains the file_name

    # Returns:
        image as down sized numpy array
    """
    path = os.path.join(images_path, image_annotation['file_name'])
    img = cv2.imread(path)
    return down_size_image(img)
