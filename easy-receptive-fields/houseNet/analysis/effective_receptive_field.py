import keras.backend as K
import tensorflow as tf
import numpy as np
from keras import Model
from keras.utils import Sequence

from util.data_loader import RandomLoader


def randomize_weights(model):
    session = K.get_session()
    for layer in model.layers:
        for v in layer.__dict__:
            v_arg = getattr(layer, v)
            if hasattr(v_arg, 'initializer'):
                initializer = getattr(v_arg, 'initializer')
                initializer.run(session=session)


def get_effective_receptive_field(model: Model,
                                  randomize_model: bool = True,
                                  data_loader: Sequence = None):
    """Calculates the effective receptive field for a given neural network architecture.
    The values are normalized after calculation for better visualization.

    # Arguments:
        model: Keras model
            a keras model
        randomize_model: bool
            Randomizes the model weights for each run (set this to False for trained models)
        data_loader: Sequence
            sequence that holds the input data for gradient calculation. (default: random)

    # Returns:
        A Numpy array which represents the receptive field
    """

    input_shape = model.input_shape[1:]
    center_x = int(input_shape[0] / 2)
    center_y = int(input_shape[1] / 2)

    output_shape = (1, *model.output_shape[1:])
    initial = np.zeros(output_shape, dtype=np.float32)
    initial[:, center_x, center_y, :] = 1

    if data_loader is None:
        data_loader = RandomLoader(dataset_size=100, batch_size=1, data_shape=input_shape)

    receptive_field = np.zeros(input_shape)
    gradients = tf.gradients(ys=model.output, xs=model.input, grad_ys=initial)
    session = K.get_session()

    print('Calculating gradient ...')
    for img, label in data_loader:
        if randomize_model:
            randomize_weights(model)
        grad = session.run(gradients, feed_dict={model.input.name: img})
        receptive_field += grad[0][0]

    # Normalize
    receptive_field = receptive_field / len(data_loader)
    receptive_field[receptive_field < 0] = 0
    receptive_field /= np.ptp(receptive_field)
    return receptive_field
