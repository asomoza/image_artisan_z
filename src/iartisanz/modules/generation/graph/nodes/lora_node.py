import logging
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image

from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_z_image_lora_to_diffusers
from diffusers.models.model_loading_utils import load_state_dict

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


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
        granular_transformer_weights_enabled: bool = False,
        transformer_granular_weights: dict[str, float] = None,
        is_slider: bool = False,
        database_id: int = 0,
        spatial_mask_enabled: bool = False,
        spatial_mask_path: str = "",
    ):
        super().__init__()
        self.path = path
        self.adapter_name = adapter_name
        self.lora_name = lora_name
        self.version = version
        self.transformer_weight = transformer_weight
        self.granular_transformer_weights_enabled = granular_transformer_weights_enabled
        self.transformer_granular_weights = transformer_granular_weights
        self.is_slider = is_slider
        self.database_id = database_id
        self.lora_enabled = True
        # Spatial masking
        self.spatial_mask_enabled = spatial_mask_enabled
        self.spatial_mask_path = spatial_mask_path
        self._cached_mask: Optional[torch.Tensor] = None
        self._mask_load_failed: bool = False

    def update_lora(
        self,
        lora_enabled: bool,
        transformer_weight: float,
        granular_transformer_weights_enabled: bool,
        granular_transformer_weights: dict[str, float],
        is_slider: bool,
        spatial_mask_enabled: bool = None,
        spatial_mask_path: str = None,
    ):
        self.lora_enabled = lora_enabled
        self.transformer_weight = transformer_weight
        self.granular_transformer_weights_enabled = granular_transformer_weights_enabled
        self.transformer_granular_weights = granular_transformer_weights
        self.is_slider = is_slider
        # Update spatial mask if provided
        if spatial_mask_enabled is not None:
            self.spatial_mask_enabled = spatial_mask_enabled
        if spatial_mask_path is not None:
            if spatial_mask_path != self.spatial_mask_path:
                # Path changed, clear cache
                self._cached_mask = None
                self._mask_load_failed = False
            self.spatial_mask_path = spatial_mask_path
        self.set_updated()

    def update_lora_weights(self, transformer_weight: float, granular_transformer_weights: dict[str, float]):
        self.transformer_weight = transformer_weight
        self.transformer_granular_weights = granular_transformer_weights
        self.set_updated()

    def update_lora_transformer_granular_enabled(self, enabled: bool):
        self.granular_transformer_weights_enabled = enabled
        self.set_updated()

    def update_slider_enabled(self, is_slider: bool):
        self.is_slider = is_slider
        self.set_updated()

    def update_lora_enabled(self, lora_enabled: bool):
        self.lora_enabled = lora_enabled
        self.set_updated()

    def update_spatial_mask(self, enabled: bool, path: str = None):
        """Update spatial mask settings.

        Args:
            enabled: Whether spatial masking is enabled
            path: Path to mask file (optional, keeps existing if None)
        """
        self.spatial_mask_enabled = enabled
        if path is not None:
            if path != self.spatial_mask_path:
                # Path changed, clear cache
                self._cached_mask = None
                self._mask_load_failed = False
            self.spatial_mask_path = path
        self.set_updated()

    def _load_spatial_mask(self) -> Optional[torch.Tensor]:
        """Load spatial mask from file.

        Returns:
            Mask tensor [1, 1, H, W] with values 0.0-1.0, or None if loading fails
        """
        if not self.spatial_mask_enabled:
            return None

        if not self.spatial_mask_path or not os.path.exists(self.spatial_mask_path):
            if not self._mask_load_failed:  # Log once
                logger.warning(
                    f"[LoraNode] Spatial mask file not found: {self.spatial_mask_path}"
                )
                self._mask_load_failed = True
            return None

        try:
            # Load mask image
            mask_img = Image.open(self.spatial_mask_path)

            # Convert to grayscale if needed
            if mask_img.mode != "L":
                # Check for alpha channel (RGBA) - use black-over-alpha convention
                if mask_img.mode == "RGBA":
                    # Extract alpha channel (painted=255, unpainted=0)
                    mask_array = np.array(mask_img)[:, :, 3]
                else:
                    # Convert RGB to grayscale
                    mask_img = mask_img.convert("L")
                    mask_array = np.array(mask_img)
            else:
                mask_array = np.array(mask_img)

            # Normalize to [0, 1]
            mask_array = mask_array.astype(np.float32) / 255.0

            # Convert to tensor: [H, W] -> [1, 1, H, W]
            mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0)

            logger.info(
                f"[LoraNode] Loaded spatial mask from {self.spatial_mask_path} "
                f"(shape={mask_tensor.shape})"
            )

            return mask_tensor

        except Exception as e:
            logger.error(f"[LoraNode] Failed to load spatial mask: {e}")
            self._mask_load_failed = True
            return None

    def __call__(self):
        mm = get_model_manager()
        transformer = mm.resolve(self.transformer)

        # If this adapter name was previously loaded from a different file, force a reload.
        # This prevents stale adapters when users swap LoRAs that share adapter names.
        existing_src = mm.get_lora_source(self.adapter_name) if self.adapter_name else None
        if (
            self.adapter_name
            and existing_src is not None
            and self.path
            and existing_src != self.path
            and self.adapter_name in getattr(transformer, "peft_config", {})
        ):
            try:
                transformer.delete_adapters([self.adapter_name])
            except Exception:
                pass
            mm.clear_lora_source(self.adapter_name)

        if self.adapter_name not in getattr(transformer, "peft_config", {}):
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

            transformer.load_lora_adapter(
                state_dict,
                network_alphas=None,
                adapter_name=self.adapter_name,
                metadata=None,
                _pipeline=None,
                low_cpu_mem_usage=False,
                hotswap=False,
            )

            mm.set_lora_source(self.adapter_name, self.path)

        scale = {
            "transformer": self.transformer_granular_weights
            if self.granular_transformer_weights_enabled
            else self.transformer_weight
        }

        if not self.lora_enabled:
            scale = {"transformer": 0.0}

        # Load spatial mask if enabled
        spatial_mask = None
        if self.spatial_mask_enabled and self._cached_mask is None:
            self._cached_mask = self._load_spatial_mask()
        if self.spatial_mask_enabled:
            spatial_mask = self._cached_mask

        # Output 3-tuple: (adapter_name, scale_dict, spatial_mask)
        self.values["lora"] = (self.adapter_name, scale, spatial_mask)
        return self.values

    def before_delete(self):
        mm = get_model_manager()
        try:
            transformer = mm.resolve(self.transformer)
        except Exception:
            return

        try:
            transformer.delete_adapters([self.adapter_name])
        except Exception:
            pass

        if self.adapter_name:
            mm.clear_lora_source(self.adapter_name)
