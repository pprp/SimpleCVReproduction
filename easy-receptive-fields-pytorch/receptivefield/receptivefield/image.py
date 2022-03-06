import os
from typing import Union, List

import numpy
from PIL import Image

from receptivefield.types import ImageShape

dir_path = os.path.dirname(os.path.realpath(__file__))


def get_default_images() -> List[str]:
    """
    List available image names, included in receptive field library.
    Those names can be used in get_default_image() function.

    :return: list of available names
    """
    images = os.listdir(os.path.join(dir_path, "resources"))
    return [im.replace(".jpg", "") for im in images if "jpg" in im]


def _get_default_image_path(name: str) -> str:

    if name not in get_default_images():
        raise Exception(
            f"Image name is '{name}' but should be one of "
            f"the following: {get_default_images()}"
        )

    image_path = os.path.join(dir_path, f"resources/{name}.jpg")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image {name} does not exist :(")

    return image_path


def tile_numpy_image(
    image: numpy.ndarray, tile_factor: int = 0, shade: bool = True
) -> numpy.ndarray:
    """
    Tile array of shape [W, H, C], where C=3.

    :param image: image array
    :param tile_factor: number of tiles, if tile_factor=0, this
        function returns a copy of the input image
    :param shade: make non-central tiles gray scaled.
    :return: numpy array image of shape [W*n, H*n, C],
        where n=2 * tile_factor + 1
    """
    timage = tile_pil_image(Image.fromarray(numpy.uint8(image)), tile_factor, shade)
    return numpy.array(timage)


def tile_pil_image(
    image: Image.Image, tile_factor: int = 0, shade: bool = True
) -> Image.Image:
    """
    Tile PIL Image.

    :param image: PIL Image object
    :param tile_factor: number of tiles, if tile_factor=0, this
        function returns a copy of the input image
    :param shade: make non-central tiles gray scaled.
    :return: Image of shape [W*n, H*n, C], where n=2 * tile_factor + 1
    """
    shape = ImageShape(*image.size)
    new_img = image.copy()
    if tile_factor > 0:
        tf = 2 * tile_factor + 1
        new_img = Image.new("RGB", (shape.w * tf, shape.h * tf))
        new_shape = ImageShape(*new_img.size)
        gray_img = image.convert("LA").convert("RGB")

        for n, i in enumerate(range(0, new_shape.w, shape.w)):
            for m, j in enumerate(range(0, new_shape.h, shape.h)):
                distance = (abs(m - tile_factor) + abs(n - tile_factor)) / tf
                alpha = distance > 0 if shade else 0
                # place image at position (i, j)
                new_img.paste(Image.blend(image, gray_img, alpha), (i, j))
    return new_img


def get_default_image(
    shape: ImageShape,
    tile_factor: int = 0,
    shade: bool = True,
    as_image: bool = False,
    name: str = "lena",
) -> Union[numpy.ndarray, Image.Image]:
    """
    Loads default image from resources and reshape it to size
    shape.

    :param shape: [width, height]
    :param tile_factor: tile image, if 0 the resulting image shape is
        [width, height], otherwise the output size is defined by number of
        tiles. tile_factor is a non-negative integer number.
    :param shade: if True and tile_factor > 0 it makes tiles gray scale
    :param as_image: if True, function returns PIL Image object, else
        numpy array.
    :param name: name of the default image to be loaded. Call
        get_default_images() to list available images names (default - lena).

    :return: numpy array of shape [width, height, 3] if as_image=False,
        otherwise PIL Image object
    """
    shape = ImageShape(*shape)
    tile_factor = int(tile_factor)

    img = Image.open(_get_default_image_path(name=name), mode="r")
    img = img.resize((shape.w, shape.h), Image.ANTIALIAS)
    img = tile_pil_image(img, tile_factor=tile_factor, shade=shade)

    if as_image:
        return img
    return numpy.array(img)
