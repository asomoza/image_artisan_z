import os

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_z_image_lora_to_diffusers
from diffusers.models.model_loading_utils import load_state_dict

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class LoraNode(Node):
    PRIORITY = 1
    REQUIRED_INPUTS = ["transformer"]
    OUTPUTS = ["lora"]

    def __init__(
        self,
        path: str = None,
        adapter_name: str = None,
        lora_name: str = None,
        version: str = None,
        transformer_weight: float = None,
        is_slider: bool = False,
        database_id: int = 0,
    ):
        super().__init__()
        self.path = path
        self.adapter_name = adapter_name
        self.lora_name = lora_name
        self.version = version
        self.transformer_weight = transformer_weight
        self.is_slider = is_slider
        self.database_id = database_id
        self.lora_enabled = True

    def update_lora(self, lora_enabled: bool, transformer_weight: float, is_slider: bool):
        self.lora_enabled = lora_enabled
        self.transformer_weight = transformer_weight
        self.is_slider = is_slider
        self.set_updated()

    def update_lora_weight(self, transformer_weight: float):
        self.transformer_weight = transformer_weight
        self.set_updated()

    def update_lora_enabled(self, lora_enabled: bool):
        self.lora_enabled = lora_enabled
        self.set_updated()

    def __call__(self):
        if self.adapter_name not in getattr(self.transformer, "peft_config", {}):
            if not self.path:
                raise IArtisanZNodeError("LoRA path is empty.", self.name)

            if not os.path.isfile(self.path):
                raise IArtisanZNodeError(f"LoRA file not found: {self.path}", self.name)

            if not self.adapter_name:
                raise IArtisanZNodeError("adapter_name is empty.", self.name)

            state_dict = load_state_dict(self.path)

            is_dora_scale_present = any("dora_scale" in k for k in state_dict)
            if is_dora_scale_present:
                state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

            has_alphas_in_sd = any(k.endswith(".alpha") for k in state_dict)
            has_diffusion_model = any(k.startswith("diffusion_model.") for k in state_dict)
            has_default = any("default." in k for k in state_dict)

            if has_alphas_in_sd or has_diffusion_model or has_default:
                state_dict = _convert_non_diffusers_z_image_lora_to_diffusers(state_dict)

            is_correct_format = all("lora" in key for key in state_dict.keys())
            if not is_correct_format:
                raise IArtisanZNodeError("Invalid LoRA checkpoint.", self.name)

            self.transformer.load_lora_adapter(
                state_dict,
                network_alphas=None,
                adapter_name=self.adapter_name,
                metadata=None,
                _pipeline=None,
                low_cpu_mem_usage=False,
                hotswap=False,
            )

        scale = {"transformer": float(self.transformer_weight) if self.transformer_weight is not None else 1.0}

        if not self.lora_enabled:
            scale = {"transformer": 0.0}

        self.values["lora"] = (self.adapter_name, scale)
        return self.values

    def before_delete(self):
        try:
            transformer = self.transformer
        except Exception:
            return

        try:
            transformer.delete_adapters([self.adapter_name])
        except Exception:
            pass
