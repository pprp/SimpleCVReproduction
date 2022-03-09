import itertools
from typing import Any, Optional

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

from receptivefields.common import estimate_rf_from_gradient
from receptivefields.image import get_default_image
from receptivefields.types import (
    ImageShape,
    GridPoint,
    GridShape,
    ReceptiveFieldDescription,
    ReceptiveFieldRect,
    to_rf_rect,
)


def _plot_rect(
    ax,
    rect: ReceptiveFieldRect,
    color: Any,
    alpha: float = 0.9,
    linewidth: float = 5,
    size: float = 90,
) -> None:
    """
    Plot rectangle and center point.

    :param ax: matplotlib axis
    :param rect: definition of rectangle
    :param color:
    :param alpha:
    :param linewidth:
    :param size: point size
    """
    ax.add_patch(
        patches.Rectangle(
            (rect.y - rect.h / 2, rect.x - rect.w / 2),
            rect.h,
            rect.w,
            alpha=alpha,
            fill=False,
            facecolor="white",
            edgecolor=color,
            linewidth=linewidth,
        )
    )
    plt.scatter([rect.y], [rect.x], s=size, c=color)


def plot_gradient_field(
    receptive_field_grad: np.ndarray,
    image: np.ndarray = None,
    axis: Optional[Any] = None,
    **plot_params
) -> None:
    """
    Plot gradient map from gradient tensor.

    :param receptive_field_grad: numpy tensor of shape [N, W, H, C]
    :param image: optional image of shape [W, H, 3]
    :param axis: a matplotlib axis object as returned by the e.g. plt.subplot
        function. If not None then axis is used for visualizations otherwise
        default figure is created.
    :param plot_params: additional plot params: figsize=(5, 5)
    """
    receptive_field = estimate_rf_from_gradient(receptive_field_grad)

    receptive_field_grad = np.array(receptive_field_grad).mean(0).mean(-1)
    receptive_field_grad /= receptive_field_grad.max()
    receptive_field_grad += (np.abs(receptive_field_grad) > 0) * 0.2

    if image is not None:
        receptive_field_grad = np.expand_dims(receptive_field_grad, -1)
        receptive_field_grad = 255 / 2 * (receptive_field_grad + 1) + image * 0.5
        receptive_field_grad = receptive_field_grad.astype("uint8")

    if axis is None:
        figsize = plot_params.get("figsize", (5, 5))
        plt.figure(figsize=figsize)
        axis = plt.subplot(111)

    plt.title("Normalized gradient map")
    im = plt.imshow(receptive_field_grad, cmap="Greys_r")
    plt.xlabel("x")
    plt.ylabel("y")

    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    axis.add_patch(
        patches.Rectangle(
            (
                receptive_field.y - receptive_field.h / 2,
                receptive_field.x - receptive_field.w / 2,
            ),  # (x,y)
            receptive_field.h,
            receptive_field.w,
            fill=False,
            alpha=0.9,
            linewidth=4,
            edgecolor=(0.2, 0.2, 0.2),
        )
    )
    axis.set_aspect("equal")
    plt.tight_layout()


def plot_receptive_grid(
    input_shape: GridShape,
    output_shape: GridShape,
    rf_params: ReceptiveFieldDescription,
    custom_image: Optional[np.ndarray] = None,
    plot_naive_rf: bool = False,
    axis: Optional[Any] = None,
    **plot_params
) -> None:
    """
    Visualize receptive field grid.

    :param input_shape: an input image shape as an instance of GridShape
    :param output_shape: an output feature map shape
    :param rf_params: an instance of ReceptiveFieldDescription computed for
        this feature map.
    :param custom_image: optional image [height, width, 3] to be plotted as
        a background.
    :param plot_naive_rf: plot naive version of the receptive field. Naive
        version of RF does not take strides, and offsets into considerations,
        it is a simple linear mapping from N points in feature map to pixels
        in the image.
    :param axis: a matplotlib axis object as returned by the e.g. plt.subplot
        function. If not None then axis is used for visualizations otherwise
        default figure is created.
    :param plot_params: additional plot params: figsize=(5, 5)
    """
    if custom_image is None:
        img = get_default_image(shape=ImageShape(input_shape.h, input_shape.w))
    else:
        img = custom_image

    figsize = plot_params.get("figsize", (10, 10))

    # plot image
    if axis is None:
        plt.figure(figsize=figsize)
        axis = plt.subplot(111)

    axis.imshow(img)
    # plot naive receptive field grid
    if plot_naive_rf:
        dw = input_shape.w / output_shape.w
        dh = input_shape.h / output_shape.h
        for i, j in itertools.product(range(output_shape.w), range(output_shape.h)):
            x0, x1 = i * dw, (i + 1) * dw
            y0, y1 = j * dh, (j + 1) * dh

            axis.add_patch(
                patches.Rectangle(
                    (y0, x0),
                    dh,
                    dw,
                    alpha=0.9,
                    fill=False,
                    edgecolor="gray",
                    linewidth=0.3,
                )
            )

    rf_offset = rf_params.offset
    rf_size = rf_params.size
    rf_stride = rf_params.stride

    # map from output grid space to input image
    def map_point(i: int, j: int):
        return np.array(rf_offset) + np.array([i, j]) * np.array(rf_stride)

    # plot RF grid based on rf params
    points = [
        map_point(i, j)
        for i, j in itertools.product(range(output_shape.w), range(output_shape.h))
    ]

    points = np.array(points)
    axis.scatter(points[:, 1], points[:, 0], marker="o", c=(0.2, 0.9, 0.1, 0.9), s=10)

    # plot receptive field from corner point
    _plot_rect(
        axis,
        rect=to_rf_rect(rf_offset, rf_size),
        color=(0.9, 0.3, 0.2),
        linewidth=5,
        size=90,
    )
    center_point = map_point(output_shape.w // 2, output_shape.h // 2)
    _plot_rect(
        axis,
        rect=to_rf_rect(GridPoint(center_point[0], center_point[1]), rf_size),
        color=(0.1, 0.3, 0.9),
        linewidth=5,
        size=90,
    )
    last_point = map_point(output_shape.w - 1, output_shape.h - 1)
    _plot_rect(
        axis,
        rect=to_rf_rect(GridPoint(last_point[0], last_point[1]), rf_size),
        color=(0.1, 0.9, 0.3),
        linewidth=5,
        size=90,
    )
    axis.set_aspect("equal")
