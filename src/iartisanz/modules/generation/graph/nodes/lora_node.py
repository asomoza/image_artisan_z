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
    ):
        super().__init__()
        self.path = path
        self.adapter_name = adapter_name
        self.lora_name = lora_name
        self.version = version
        self.transformer_weight = transformer_weight
        self.is_slider = is_slider

    def update_lora(self, enabled: bool, transformer_weight: float, is_slider: bool):
        self.enabled = enabled
        self.transformer_weight = transformer_weight
        self.is_slider = is_slider
        self.set_updated()

    def __call__(self):
        if not os.path.isfile(self.path):
            raise IArtisanZNodeError(f"LoRA file not found: {self.path}", self.name)

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
            raise ValueError("Invalid LoRA checkpoint.")

        self.transformer.load_lora_adapter(
            state_dict,
            network_alphas=None,
            adapter_name=self.adapter_name,
            metadata=None,
            _pipeline=None,
            low_cpu_mem_usage=False,
            hotswap=False,
        )

        if not self.enabled:
            self.transformer_weight = 0.0

        self.values["lora"] = (self.adapter_name, self.transformer_weight)

    def before_delete(self):
        try:
            self.transformer.delete_adapters([self.adapter_name])
        except IArtisanZNodeError:
            pass
