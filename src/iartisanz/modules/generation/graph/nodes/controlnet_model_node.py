import os

import torch
from diffusers import ZImageControlNetModel
from safetensors.torch import load_file

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


def _get_controlnet_config(path: str) -> dict:
    """Get the appropriate config for a ControlNet model based on its path.

    The lite models use fewer control layers than the regular models.
    """
    is_lite = "lite" in path.lower()

    base_config = {
        "control_in_dim": 33,
        "control_refiner_layers_places": [0, 1],
        "add_control_noise_refiner": "control_noise_refiner",
        "all_patch_size": [2],
        "all_f_patch_size": [1],
        "dim": 3840,
        "n_refiner_layers": 2,
        "n_heads": 30,
        "n_kv_heads": 30,
        "norm_eps": 1e-05,
        "qk_norm": True,
    }

    if is_lite:
        # Lite models use fewer control injection points
        base_config["control_layers_places"] = [0, 10, 20]
    else:
        # Regular models use more control injection points
        base_config["control_layers_places"] = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28]

    return base_config


class ControlNetModelNode(Node):
    """Loads an optional Z-Image ControlNet.

    This node is not part of the default graph. When present and enabled, it
    registers a `controlnet` component with the global ModelManager and outputs
    a `ModelHandle("controlnet")` for downstream nodes.
    """

    PRIORITY = 2
    REQUIRED_INPUTS = ["transformer"]
    OUTPUTS = ["controlnet"]

    def __init__(self, path: str | None = None):
        super().__init__()
        self.path = f"{path}.safetensors"

    def update_value(self, path: str | None):
        self.path = f"{path}.safetensors"
        self.set_updated()

    def __call__(self):
        if not self.path:
            raise IArtisanZNodeError("No ControlNet path provided", self.__class__.__name__)

        if not os.path.exists(self.path):
            raise IArtisanZNodeError(f"ControlNet file not found: {self.path}", self.__class__.__name__)

        mm = get_model_manager()

        # If already loaded from this exact path, reuse it.
        if mm.has("controlnet"):
            try:
                existing = mm.get_raw("controlnet")
                if getattr(existing, "_iartisanz_source_path", None) == self.path:
                    self.values["controlnet"] = ModelHandle("controlnet")
                    return self.values
            except Exception:
                pass

        transformer = mm.resolve(self.transformer)

        try:
            # Get the appropriate config for this model
            config = _get_controlnet_config(self.path)

            # Create the ControlNet model with the config (offline, no network calls)
            controlnet = ZImageControlNetModel(**config)

            # Load the weights from the safetensors file
            state_dict = load_file(self.path)

            # Load the state dict into the model
            # Use strict=False to allow missing/unexpected keys (e.g., transformer-specific weights)
            load_result = controlnet.load_state_dict(state_dict, strict=False)

            # Convert to the appropriate dtype
            if self.dtype is not None:
                controlnet = controlnet.to(dtype=self.dtype)

            # Copy shared components from the transformer
            controlnet = ZImageControlNetModel.from_transformer(controlnet, transformer)
        except Exception as e:
            raise IArtisanZNodeError(f"Failed to load ControlNet: {e}", self.__class__.__name__) from e

        # Track provenance for reuse across runs.
        try:
            setattr(controlnet, "_iartisanz_source_path", self.path)
        except Exception:
            pass

        mm.register_active_model(model_id=mm.active_model_id(), controlnet=controlnet)
        self.values["controlnet"] = ModelHandle("controlnet")
        return self.values
