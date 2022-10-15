import configparser
import cv2
import os
import numpy as np

import keras
from pycocotools import mask
from pycocotools.coco import COCO

from networks.uNet3 import UNet3
from util.array_utils import down_size_image, image_as_array


class Predictor:
    """Helper for getting predictions for random images from a keras model.

    # Arguments:
        data_path: str
            absolute path to data directory
        model: keras.Model
            trained model
        random_state : np.random.RandomState
            random_state which will be used for drawing images from the validation data
    """

    def __init__(self: 'Predictor',
                 data_path: str,
                 model: keras.Model,
                 random_state: np.random.RandomState):
        self.images_path = os.path.join(data_path, 'images')
        self.annotations_path = os.path.join(data_path, 'annotation-small.json')
        self.coco = COCO(self.annotations_path)

        self.model = model

        if isinstance(random_state, np.random.RandomState):
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState()

    def get_random_prediction(self, batch_size: int = 4):
        """Gets predictions for <batch_size> random images.

        # Arguments:
            batch_size: int
                amount of images to draw

        # Returns
            Tuple of images (3-channeled), truth and prediction (both 1-channeled)
        """
        image_annotations = self._load_random_batch(batch_size)

        images, truth, predictions = [], [], []
        for image_annotation in image_annotations:
            image = image_as_array(self.images_path, image_annotation)
            label = self._label_as_array(image_annotation)
            prediction = self._prediction_as_array(image)

            images.append(image)
            truth.append(label)
            predictions.append(prediction)

        return images, truth, predictions

    def _load_random_batch(self, batch_size):
        image_ids = self.coco.getImgIds()
        random_ids = self.random_state.choice(image_ids, batch_size).tolist()
        images = self.coco.loadImgs(random_ids)
        return images

    def _label_as_array(self, image_annotation):
        annotations = self.coco.loadAnns(self.coco.getAnnIds(image_annotation['id']))
        return self._mask_from_annotation(annotations, (image_annotation['height'], image_annotation['width']))

    def _mask_from_annotation(self, annotation, shape):
        label = np.zeros(shape)
        for a in annotation:
            rle = mask.frPyObjects(a['segmentation'], *shape)
            m = np.squeeze(mask.decode(rle))
            label = np.logical_or(label, m)
        return down_size_image(label)

    def _prediction_as_array(self, image):
        image = image.reshape((1, *image.shape)) / 255
        prediction = self.model.predict(image)
        return np.squeeze(prediction)


def visualize(images, ground_truth, predictions):
    ground_truth = [cv2.cvtColor(gt.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) for gt in ground_truth]
    predictions = [cv2.cvtColor(p.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR) for p in predictions]

    collection = np.ones((800, 600, 3), dtype=np.uint8) * 50
    for i in range(len(images)):
        pos = i * 200 + 25
        collection[pos: pos + 150, 25:175, :] = images[i]
        collection[pos: pos + 150, 225:375, :] = ground_truth[i]
        collection[pos: pos + 150, 425:575, :] = predictions[i]
    cv2.imshow('Example Prediction', collection)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example prediction
if __name__ == '__main__':
    # Get config
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.pardir, 'config.ini'))

    # Build all paths
    path_to_model = os.path.join(config['DIRECTORIES']['models'], 'uNet.50-0.8445.hdf5')
    path_to_data = config['DIRECTORIES']['val_data']

    # Load model
    my_model = UNet3()
    my_model.load_weights(path_to_model)

    # Predict and visualize
    evaluator = Predictor(path_to_data, my_model, np.random.RandomState(2013))
    data = evaluator.get_random_prediction()
    visualize(*data)
