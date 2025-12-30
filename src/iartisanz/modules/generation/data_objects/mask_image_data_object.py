import attr


@attr.s
class MaskImageDataObject:
    node_id: int = attr.ib(default=None)
    background: str = attr.ib(default=None)
    image: str = attr.ib(default=None)
    thumb: str = attr.ib(default=None)
