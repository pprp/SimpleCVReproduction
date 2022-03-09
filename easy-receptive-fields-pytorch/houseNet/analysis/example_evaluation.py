import configparser
import json
import os

from keras import Model
from keras.layers import UpSampling2D

from analysis.evaluation import Evaluator
from networks.uNet3 import UNet3


def get_model(model, path):
    _model = model()
    _model.load_weights(path)
    up = UpSampling2D()(_model.output)
    return Model(_model.input, up)


if __name__ == '__main__':
    # Get config
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.pardir, 'config.ini'))

    # Build all paths
    path_to_model = os.path.join(config['DIRECTORIES']['models'], 'uNet.50-0.8445.hdf5')
    path_to_predictions = os.path.join(config['DIRECTORIES']['predictions'], 'uNet_prediction.json')
    path_to_dataset = config['DIRECTORIES']['val_data']

    # Initialize model and evaluator
    my_model = get_model(UNet3, path_to_model)
    evaluator = Evaluator(path_to_dataset, my_model)

    # Can be skipped if you have already saved the predictions
    # evaluator.save_predictions_as_json(path_to_predictions)

    # Evaluate
    prediction_annotations = json.loads(open(path_to_predictions).read())
    evaluator.evaluate(prediction_annotations)
