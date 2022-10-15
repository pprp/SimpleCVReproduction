import pytest
import torch.nn as nn
from numpy.testing import assert_allclose
from receptivefield.pytorch import PytorchReceptiveField
from receptivefield.image import get_default_image
from receptivefield.types import ImageShape


class Linear(nn.Module):
    """An identity activation function"""

    def forward(self, x):
        return x


class SimpleVGG(nn.Module):
    def __init__(self, disable_activations: bool = False):
        super(SimpleVGG, self).__init__()
        self.blocks = self._build_blocks(disable_activations)
        self.feature_maps = None

    def forward(self, x):
        self.feature_maps = []
        for block in self.blocks:
            for layer in block:
                x = layer(x)
            self.feature_maps.append(x)
        return x

    def _build_blocks(self, disable_activations: bool):
        activation = lambda: Linear() if disable_activations else nn.ReLU()

        block1 = [
            nn.Conv2d(3, 64, kernel_size=3),
            activation(),
            nn.Conv2d(64, 64, kernel_size=3),
            activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ]
        block2 = [
            nn.Conv2d(64, 128, kernel_size=3),
            activation(),
            nn.Conv2d(128, 128, kernel_size=3),
            activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ]
        block3 = [
            nn.Conv2d(128, 256, kernel_size=3),
            activation(),
            nn.Conv2d(256, 256, kernel_size=3),
            activation(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        ]
        return [nn.Sequential(*block1), nn.Sequential(*block2), nn.Sequential(*block3)]


def model_fn() -> nn.Module:
    model = SimpleVGG(disable_activations=True)
    model.eval()
    return model


def get_test_image(shape=(64, 64), tile_factor=0):
    image = get_default_image(shape=shape, tile_factor=tile_factor)
    return image


def test_example_network():
    input_shape = [96, 96, 3]
    rf = PytorchReceptiveField(model_fn)
    rf_params = rf.compute(input_shape=ImageShape(*input_shape))

    assert_allclose(rf_params[0].rf.size, (6, 6))
    assert_allclose(rf_params[0].rf.stride, (2, 2))

    rs = 6 + (2 + 2 + 1) * 2
    assert_allclose(rf_params[1].rf.size, (rs, rs))
    assert_allclose(rf_params[1].rf.stride, (4, 4))

    rs = 6 + (2 + 2 + 1) * 2 + (2 + 2 + 1) * 4
    assert_allclose(rf_params[2].rf.size, (rs, rs))
    assert_allclose(rf_params[2].rf.stride, (8, 8))


if __name__ == "__main__":
    pytest.main([__file__])
