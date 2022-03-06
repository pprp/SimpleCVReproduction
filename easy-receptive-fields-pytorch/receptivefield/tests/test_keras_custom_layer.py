import unittest
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, Layer
from tensorflow.keras.models import Model

from receptivefield.image import get_default_image
from receptivefield.keras import KerasReceptiveField
from receptivefield.types import ReceptiveFieldDescription, Size

layers = tf.keras.layers
models = tf.keras.models


class MyLayer(Layer):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
        self.conv = Conv2D(1, 5, name="inner_conv")

    def call(self, inputs):
        return self.conv(inputs)

    def compute_output_shape(self, input_shape):
        return [*input_shape[:1], 1]


def model_build_func(input_shape=[224, 224, 3]):
    im = Input(input_shape, name="image")
    conv_im = MyLayer(name="feature_map")(im)
    model = Model(im, conv_im)
    return model


def get_test_image(shape=(64, 64, 3), tile_factor=0):
    image = get_default_image(shape=shape, tile_factor=tile_factor)
    return image


class TestCustomLayer(unittest.TestCase):
    def test_custom_layer(self):

        rf_params = KerasReceptiveField(model_build_func, init_weights=True).compute(
            input_shape=(224, 224, 3),
            input_layer="image",
            output_layers=["feature_map"],
        )
        expected_rf = ReceptiveFieldDescription(
            offset=(2.5, 2.5), stride=(1.0, 1.0), size=Size(w=5, h=5)
        )
        self.assertEqual(rf_params[0].rf, expected_rf)
