import attr


@attr.s(slots=True)
class LoraDataObject:
    name: str = attr.ib()
    filename: str = attr.ib()
    version: str = attr.ib()
    path: str = attr.ib()
    lora_node_name: str = attr.ib()
    enabled: bool = attr.ib(default=True)
    transformer_weight: float = attr.ib(default=1.00)
    node_id: int = attr.ib(default=None)
    lora_id = attr.ib(default=None)
    locked: bool = attr.ib(default=True)
    is_slider: bool = attr.ib(default=False)
    type: str = attr.ib(default="")
    id: int = attr.ib(default=0)
