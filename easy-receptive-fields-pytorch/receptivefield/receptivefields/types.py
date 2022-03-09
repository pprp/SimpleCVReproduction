from typing import NamedTuple, Any


class Size(NamedTuple):
    w: int
    h: int


class ImageShape(NamedTuple):
    w: int
    h: int
    c: int = 3


class GridPoint(NamedTuple):
    x: int
    y: int


class GridShape(NamedTuple):
    n: int
    w: int
    h: int
    c: int

    def replace(self, **kwargs: Any) -> "GridShape":
        return self._replace(**kwargs)


class ReceptiveFieldRect(NamedTuple):
    x: int
    y: int
    w: int
    h: int


class ReceptiveFieldDescription(NamedTuple):
    offset: GridPoint
    stride: GridPoint
    size: Size


class FeatureMapDescription(NamedTuple):
    """
    size: a feature map size
    rf: a ReceptiveFieldDescription
    """

    size: Size
    rf: ReceptiveFieldDescription


def to_rf_rect(point: GridPoint, size: Size) -> ReceptiveFieldRect:
    point = GridPoint(*point)
    size = Size(*size)
    return ReceptiveFieldRect(x=point.x, y=point.y, w=size.w, h=size.h)
