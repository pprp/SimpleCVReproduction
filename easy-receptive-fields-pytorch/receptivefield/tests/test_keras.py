import pytest
from numpy.testing import assert_allclose
import tensorflow as tf

# from keras.layers import Conv2D, Input, AvgPool2D
# from keras.models import Model
# import keras

from receptivefield.image import get_default_image
from receptivefield.keras import KerasReceptiveField
from receptivefield.types import ImageShape


layers = tf.keras.layers
models = tf.keras.models


def get_build_func(padding="same", activation="linear"):
    def model_build_func(input_shape):
        inp = layers.Input(shape=input_shape, name="input_image")
        x = layers.Conv2D(32, (5, 5), padding=padding, activation=activation)(inp)
        x = layers.Conv2D(
            32, (3, 3), padding=padding, activation=activation, name="conv0"
        )(x)
        x = layers.AvgPool2D()(x)
        x = layers.Conv2D(64, (3, 3), activation=activation, padding=padding)(x)
        x = layers.Conv2D(
            64, (3, 3), activation=activation, padding=padding, name="conv1"
        )(x)
        model = models.Model(inp, x)
        return model

    return model_build_func


def get_test_image(shape=(64, 64), tile_factor=0):
    image = get_default_image(shape=shape, tile_factor=tile_factor)
    return image


def test_same():
    image = get_test_image(tile_factor=0)
    rf = KerasReceptiveField(get_build_func(padding="same"), init_weights=True)
    rf_params0 = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["conv1"],
    )

    print(rf_params0)

    image = get_test_image(tile_factor=1)
    rf = KerasReceptiveField(get_build_func(padding="same"), init_weights=True)
    rf_params1 = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["conv1"],
    )

    print(rf_params1)
    assert_allclose(rf_params0[0].rf, rf_params1[0].rf)


def test_valid():
    image = get_test_image(tile_factor=0)
    rf = KerasReceptiveField(get_build_func(padding="valid"), init_weights=True)
    rf_params0 = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["conv1"],
    )

    print(rf_params0)

    image = get_test_image(tile_factor=1)
    rf = KerasReceptiveField(get_build_func(padding="valid"), init_weights=True)
    rf_params1 = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["conv1"],
    )

    print(rf_params1)
    assert_allclose(rf_params0[0].rf, rf_params1[0].rf)


def test_expected_values():
    image = get_test_image(tile_factor=0)
    rf = KerasReceptiveField(get_build_func(padding="valid"), init_weights=True)
    rf_params0 = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["conv1"],
    )[0]

    assert_allclose(rf_params0.rf.stride, (2, 2))
    assert_allclose(rf_params0.rf.size, (((2 + 1) * 2 + 2) * 2, ((2 + 1) * 2 + 2) * 2))


def test_expected_values_two_fm():
    image = get_test_image(tile_factor=0)
    rf = KerasReceptiveField(get_build_func(padding="valid"), init_weights=True)
    rf_params = rf.compute(
        input_shape=ImageShape(*image.shape),
        input_layer="input_image",
        output_layers=["conv0", "conv1"],
    )

    assert_allclose(rf_params[0].rf.stride, (1, 1))
    assert_allclose(rf_params[0].rf.size, ((2 + 1) * 2 + 1, (2 + 1) * 2 + 1))

    assert_allclose(rf_params[1].rf.stride, (2, 2))
    assert_allclose(
        rf_params[1].rf.size, (((2 + 1) * 2 + 2) * 2, ((2 + 1) * 2 + 2) * 2)
    )


if __name__ == "__main__":
    pytest.main([__file__])
