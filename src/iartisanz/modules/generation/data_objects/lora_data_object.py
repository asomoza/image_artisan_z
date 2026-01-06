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
    granular_transformer_weights_enabled: bool = attr.ib(default=False)
    granular_transformer_weights: dict = attr.ib(
        default=attr.Factory(
            lambda: {
                "layers.0": 1.0,
                "layers.1": 1.0,
                "layers.2": 1.0,
                "layers.3": 1.0,
                "layers.4": 1.0,
                "layers.5": 1.0,
                "layers.6": 1.0,
                "layers.7": 1.0,
                "layers.8": 1.0,
                "layers.9": 1.0,
                "layers.10": 1.0,
                "layers.11": 1.0,
                "layers.12": 1.0,
                "layers.13": 1.0,
                "layers.14": 1.0,
                "layers.15": 1.0,
                "layers.16": 1.0,
                "layers.17": 1.0,
                "layers.18": 1.0,
                "layers.19": 1.0,
                "layers.20": 1.0,
                "layers.21": 1.0,
                "layers.22": 1.0,
                "layers.23": 1.0,
                "layers.24": 1.0,
                "layers.25": 1.0,
                "layers.26": 1.0,
                "layers.27": 1.0,
                "layers.28": 1.0,
                "layers.29": 1.0,
            }
        )
    )
