from typing import Tuple, List

import numpy as np

from receptivefields.logging import get_logger
from receptivefields.types import GridShape, ReceptiveFieldRect

_logger = get_logger()


def _compute_fans(shape: GridShape) -> Tuple[int, int]:
    """Computes the number of input and output units for a weight shape.

    # Arguments
        shape: Integer shape tuple of type GridShape (N, W, H, C)

    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        receptive_field_size = np.prod(shape[:-2])
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def scaled_constant(
    scale: float, shape: GridShape, mode: str = "fan_avg", dtype: type = np.float32
) -> np.ndarray:
    """
    Returns np array of given @shape filed with constant value.
    :param scale: constant value multiplier, default 1
    :param shape: shape of resulting tensor
    :param mode: type of constant value estimation.
    :param dtype:
    :return: numpy tensor
    """
    fan_in, fan_out = _compute_fans(shape)
    if mode == "fan_in":
        scale /= max(1.0, fan_in)
    elif mode == "fan_out":
        scale /= max(1.0, fan_out)
    elif mode == "fan_avg":
        scale /= max(1.0, float(fan_in + fan_out) / 2)

    limit = scale
    return limit * np.ones(shape, dtype=dtype)


def estimate_rf_from_gradient(receptive_field_grad: np.ndarray) -> ReceptiveFieldRect:
    """
    Given input gradient tensors of shape [N, W, H, C] it returns the
    estimated size of gradient `blob` in W-H directions i.e. this
    function computes the size of gradient in W-H axis for each feature map.

    :param receptive_field_grad: a numpy tensor with gradient values
        obtained for certain feature map
    :return: a corresponding ReceptiveFieldRect
    """

    receptive_field_grad = np.array(receptive_field_grad).mean(0).mean(-1)
    binary_map: np.ndarray = (receptive_field_grad[:, :] > 0)

    x_cs: np.ndarray = binary_map.sum(-1) >= 1
    y_cs: np.ndarray = binary_map.sum(0) >= 1

    x = np.arange(len(x_cs))
    y = np.arange(len(y_cs))

    width = x_cs.sum()
    height = y_cs.sum()

    x = np.sum(x * x_cs) / width
    y = np.sum(y * y_cs) / height

    return ReceptiveFieldRect(x, y, width, height)


def estimate_rf_from_gradients(
    receptive_field_grads: List[np.ndarray],
) -> List[ReceptiveFieldRect]:
    """
    Given input gradient tensors of shape [N, W, H, C] it returns the
    estimated size of gradient `blob` in W-H directions i.e. this
    function computes the size of gradient in W-H axis for each feature map.

    :param receptive_field_grads: a list of numpy tensor with gradient values
        obtained for different feature maps
    :return: a list of corresponding ReceptiveFieldRect
    """

    return [
        estimate_rf_from_gradient(receptive_field_grad)
        for receptive_field_grad in receptive_field_grads
    ]
