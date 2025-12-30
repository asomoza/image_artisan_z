import attr


@attr.s(auto_attribs=True, slots=True)
class SourceImageDataObject:
    enabled: bool = attr.ib(default=False)
    source_image_path: str = attr.ib(default=None)
    source_thumb_path: str = attr.ib(default=None)
    source_image_mask_path: str = attr.ib(default=None)
    source_image_final_mask_path: str = attr.ib(default=None)
    source_image_mask_thumb_path: str = attr.ib(default=None)
    source_image_layers: list = attr.ib(factory=list)
    mask_image_path: str = attr.ib(default=None)
