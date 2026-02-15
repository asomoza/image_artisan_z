import logging
import os
from typing import Optional

import numpy as np
import torch
from diffusers.loaders.lora_conversion_utils import _convert_non_diffusers_z_image_lora_to_diffusers
from diffusers.models.model_loading_utils import load_state_dict
from PIL import Image

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


def _normalize_flux2_lora_keys(state_dict: dict) -> dict:
    """Normalize kohya-format lora_down/lora_up keys to lora_A/lora_B.

    Handles alpha scaling: when .alpha keys are present, bakes scale into weights
    (scale = alpha / rank). When absent, default alpha = rank so scale = 1.0.
    """
    if not any("lora_down" in k for k in state_dict):
        return state_dict

    normalized = {}
    # Group keys by layer prefix (everything before .lora_down/.lora_up/.alpha)
    layer_prefixes = set()
    for k in state_dict:
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if k.endswith(suffix):
                layer_prefixes.add(k[: -len(suffix)])
                break

    for lp in layer_prefixes:
        down_key = f"{lp}.lora_down.weight"
        up_key = f"{lp}.lora_up.weight"
        alpha_key = f"{lp}.alpha"

        if down_key not in state_dict or up_key not in state_dict:
            continue

        down = state_dict[down_key]
        up = state_dict[up_key]
        rank = down.shape[0]

        alpha = state_dict.get(alpha_key)
        if alpha is not None:
            scale = alpha.item() / rank
        else:
            scale = 1.0

        # Bake alpha scaling into lora_A (matches diffusers convention)
        normalized[f"{lp}.lora_A.weight"] = down * scale if scale != 1.0 else down
        normalized[f"{lp}.lora_B.weight"] = up

    return normalized


def _convert_flux2_lora_to_diffusers(state_dict: dict) -> dict:
    """Convert non-diffusers Flux2 LoRA to diffusers PEFT format.

    Unlike diffusers' built-in converter which hardcodes 48 single blocks (full Flux2),
    this dynamically detects block counts from the keys — works for Klein 9B (24 single),
    Klein 4B, or any other variant. Also handles kohya-format lora_down/lora_up naming.
    """
    converted = {}

    # Strip diffusion_model. prefix
    prefix = "diffusion_model."
    sd = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}

    # Detect block indices from keys
    single_indices = set()
    double_indices = set()
    for k in sd:
        if k.startswith("single_blocks."):
            single_indices.add(int(k.split(".")[1]))
        elif k.startswith("double_blocks."):
            double_indices.add(int(k.split(".")[1]))

    lora_keys = ("lora_A", "lora_B")

    for sl in sorted(single_indices):
        src = f"single_blocks.{sl}"
        dst = f"single_transformer_blocks.{sl}.attn"
        for lk in lora_keys:
            converted[f"{dst}.to_qkv_mlp_proj.{lk}.weight"] = sd.pop(f"{src}.linear1.{lk}.weight")
            converted[f"{dst}.to_out.{lk}.weight"] = sd.pop(f"{src}.linear2.{lk}.weight")

    for dl in sorted(double_indices):
        tb = f"transformer_blocks.{dl}"
        for lk in lora_keys:
            for attn_type in ("img_attn", "txt_attn"):
                fused = sd.pop(f"double_blocks.{dl}.{attn_type}.qkv.{lk}.weight")
                if lk == "lora_A":
                    proj_keys = (
                        ["to_q", "to_k", "to_v"]
                        if attn_type == "img_attn"
                        else ["add_q_proj", "add_k_proj", "add_v_proj"]
                    )
                    for pk in proj_keys:
                        converted[f"{tb}.attn.{pk}.{lk}.weight"] = torch.cat([fused])
                else:
                    q, k_val, v = torch.chunk(fused, 3, dim=0)
                    if attn_type == "img_attn":
                        converted[f"{tb}.attn.to_q.{lk}.weight"] = q
                        converted[f"{tb}.attn.to_k.{lk}.weight"] = k_val
                        converted[f"{tb}.attn.to_v.{lk}.weight"] = v
                    else:
                        converted[f"{tb}.attn.add_q_proj.{lk}.weight"] = q
                        converted[f"{tb}.attn.add_k_proj.{lk}.weight"] = k_val
                        converted[f"{tb}.attn.add_v_proj.{lk}.weight"] = v

        proj_mappings = [
            ("img_attn.proj", "attn.to_out.0"),
            ("txt_attn.proj", "attn.to_add_out"),
        ]
        for org, diff in proj_mappings:
            for lk in lora_keys:
                converted[f"{tb}.{diff}.{lk}.weight"] = sd.pop(f"double_blocks.{dl}.{org}.{lk}.weight")

        mlp_mappings = [
            ("img_mlp.0", "ff.linear_in"),
            ("img_mlp.2", "ff.linear_out"),
            ("txt_mlp.0", "ff_context.linear_in"),
            ("txt_mlp.2", "ff_context.linear_out"),
        ]
        for org, diff in mlp_mappings:
            for lk in lora_keys:
                converted[f"{tb}.{diff}.{lk}.weight"] = sd.pop(f"double_blocks.{dl}.{org}.{lk}.weight")

    # Root-level layers (modulation, embedders, final layer) — simple 1:1 renames
    root_mappings = {
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "txt_in": "context_embedder",
        "img_in": "x_embedder",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "final_layer.linear": "proj_out",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    }
    for org, diff in root_mappings.items():
        for lk in lora_keys:
            src_key = f"{org}.{lk}.weight"
            if src_key in sd:
                converted[f"{diff}.{lk}.weight"] = sd.pop(src_key)

    if sd:
        raise ValueError(f"Unexpected keys remaining after Flux2 LoRA conversion: {list(sd.keys())}")

    return {f"transformer.{k}": v for k, v in converted.items()}


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
        trigger_words: str = "",
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
        # FreeFuse trigger words
        self.trigger_words = trigger_words

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

    def update_trigger_words(self, trigger_words: str):
        self.trigger_words = trigger_words
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
                logger.warning(f"[LoraNode] Spatial mask file not found: {self.spatial_mask_path}")
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

            logger.info(f"[LoraNode] Loaded spatial mask from {self.spatial_mask_path} (shape={mask_tensor.shape})")

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

            # Strip dora_scale and normalize lora_down/lora_up → lora_A/lora_B
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}
            state_dict = _normalize_flux2_lora_keys(state_dict)

            # Detect LoRA format: Flux2 original vs Z-Image vs already-diffusers
            # "double_blocks." only appears in original format (diffusers uses "transformer_blocks.")
            # "single_blocks." must exclude "single_transformer_blocks." (diffusers format)
            is_flux2_original = any(
                "double_blocks." in k
                or ("single_blocks." in k and "single_transformer_blocks." not in k)
                for k in state_dict
            )

            if is_flux2_original:
                state_dict = _convert_flux2_lora_to_diffusers(state_dict)
            else:
                has_alphas_in_sd = any(k.endswith(".alpha") for k in state_dict)
                has_diffusion_model = any(k.startswith("diffusion_model.") for k in state_dict)
                has_default = any("default." in k for k in state_dict)

                if has_alphas_in_sd or has_diffusion_model or has_default:
                    state_dict = _convert_non_diffusers_z_image_lora_to_diffusers(state_dict)

            is_correct_format = all("lora" in key for key in state_dict.keys())
            if not is_correct_format:
                raise IArtisanZNodeError("Invalid LoRA checkpoint.", self.name)

            try:
                transformer.load_lora_adapter(
                    state_dict,
                    network_alphas=None,
                    adapter_name=self.adapter_name,
                    metadata=None,
                    _pipeline=None,
                    low_cpu_mem_usage=False,
                    hotswap=False,
                )
            except Exception as e:
                raise IArtisanZNodeError(f"Failed to load LoRA adapter: {e}", self.name)

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

        # Output 4-tuple: (adapter_name, scale_dict, spatial_mask, trigger_words)
        self.values["lora"] = (self.adapter_name, scale, spatial_mask, self.trigger_words)
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
