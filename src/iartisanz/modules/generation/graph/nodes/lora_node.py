import logging
import os
from typing import Optional

import numpy as np
import torch
from diffusers.models.model_loading_utils import load_state_dict
from PIL import Image

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.lora_conversion import (
    _convert_flux2_lora,
    _convert_zimage_lora,
    _is_lokr_format,
    _map_flux2_lokr_layer_to_targets,
    _map_zimage_lokr_layer_to_targets,
)
from iartisanz.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


def _parse_lokr_entries(state_dict: dict, transformer) -> list[dict]:
    """Parse LoKr state dict into entries for direct weight merging.

    Each entry contains w1/w2 tensors and target parameter mappings.
    Alpha scaling is baked into w1 (unless sentinel value detected).
    Dispatches to model-specific layer mapping based on transformer type.
    """
    from diffusers import Flux2Transformer2DModel

    if isinstance(transformer, Flux2Transformer2DModel):
        map_fn = _map_flux2_lokr_layer_to_targets
    else:
        map_fn = _map_zimage_lokr_layer_to_targets

    # Strip diffusion_model. prefix
    dm_prefix = "diffusion_model."
    sd = {k[len(dm_prefix):] if k.startswith(dm_prefix) else k: v for k, v in state_dict.items()}

    # Identify LoKr layer prefixes
    lokr_w_suffixes = (
        ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b",
        ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
    )
    lokr_prefixes = set()
    for k in sd:
        for sfx in lokr_w_suffixes:
            if k.endswith(sfx):
                lokr_prefixes.add(k[:-len(sfx)])
                break

    entries = []
    for lp in sorted(lokr_prefixes):
        w1 = sd.get(f"{lp}.lokr_w1")
        w2 = sd.get(f"{lp}.lokr_w2")
        w1_a = sd.get(f"{lp}.lokr_w1_a")
        w1_b = sd.get(f"{lp}.lokr_w1_b")
        w2_a = sd.get(f"{lp}.lokr_w2_a")
        w2_b = sd.get(f"{lp}.lokr_w2_b")
        t2 = sd.get(f"{lp}.lokr_t2")
        alpha = sd.get(f"{lp}.alpha")

        # Reconstruct from low-rank factors if needed
        if w1 is None and w1_a is not None and w1_b is not None:
            w1 = w1_a @ w1_b
        if w2 is None and w2_a is not None and w2_b is not None:
            w2 = w2_a @ w2_b

        if w1 is None or w2 is None:
            logger.warning("[LoKr] Incomplete layer %s, skipping", lp)
            continue

        if t2 is not None:
            w2 = t2 @ w2

        # Bake alpha scaling into w1 (float32, tiny tensor).
        # Skip sentinel values (e.g. ~1e10 in bfloat16) that produce unreasonable scale.
        w1_scaled = w1.float()
        if alpha is not None:
            lokr_dim = min(w1.shape)
            scale = alpha.float().item() / lokr_dim
            if 1e-3 <= abs(scale) <= 1e3:
                w1_scaled = w1_scaled * scale

        targets = map_fn(lp)
        if not targets:
            logger.warning("[LoKr] No target mapping for layer %s, skipping", lp)
            continue

        entries.append({"w1": w1_scaled, "w2": w2, "targets": targets})

    logger.info("[LoKr] Parsed %d layers for weight merging", len(entries))
    return entries


def _get_module_weight(model, module_path: str):
    """Get a module's weight tensor, handling PEFT wrapping transparently."""
    mod = model
    for part in module_path.split("."):
        mod = getattr(mod, part, None)
        if mod is None:
            return None
    # PEFT wraps Linear → LoraLayer with base_layer attribute
    if hasattr(mod, "base_layer"):
        return mod.base_layer.weight
    return mod.weight


def _apply_lokr_merge(
    model, entries: list[dict], new_scale: float, original_weights: dict[str, torch.Tensor]
):
    """Apply LoKr weight merge to model parameters using save/restore strategy.

    Instead of accumulating deltas (which loses precision due to floating-point
    rounding), this saves the original weights on first call and restores them
    before applying the new scale. This guarantees exact restoration when
    new_scale=0.

    Args:
        model: The transformer model.
        entries: LoKr entries from _parse_lokr_entries.
        new_scale: The desired LoKr scale (0.0 = fully unmerged).
        original_weights: Dict mapping module_path to saved original weight
            tensors. Populated on first call, used on subsequent calls.
    """
    for entry in entries:
        delta = torch.kron(entry["w1"], entry["w2"].float())

        for module_path, split_idx in entry["targets"]:
            weight = _get_module_weight(model, module_path)
            if weight is None:
                logger.warning("[LoKr] Parameter not found: %s", module_path)
                continue

            # Save original weight on first encounter (CPU to avoid VRAM bloat)
            if module_path not in original_weights:
                original_weights[module_path] = weight.data.detach().cpu().clone()

            if split_idx is not None:
                chunk = delta.chunk(3, dim=0)[split_idx]
            else:
                chunk = delta

            # Restore original weight, then apply at new scale
            weight.data.copy_(original_weights[module_path])
            if abs(new_scale) > 1e-8:
                weight.data.add_(chunk.to(device=weight.device, dtype=weight.dtype) * new_scale)


def _convert_lora_to_diffusers(state_dict: dict, transformer) -> dict:
    """Convert LoRA state dict to diffusers PEFT format based on transformer type."""
    from diffusers import Flux2Transformer2DModel

    if isinstance(transformer, Flux2Transformer2DModel):
        return _convert_flux2_lora(state_dict)
    return _convert_zimage_lora(state_dict)


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
        # LoKr weight merge state (set on first load if LoKr format detected)
        self._is_lokr = False
        self._lokr_entries: list[dict] | None = None
        self._lokr_scale = 0.0
        self._lokr_transformer_id: int | None = None
        self._lokr_original_weights: dict[str, torch.Tensor] = {}

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

        # LoKr weight merge path (already detected from prior call)
        if self._is_lokr:
            self._update_lokr_merge(transformer)
            self.values["lora"] = (None, None, self._get_spatial_mask(), self.trigger_words)
            return self.values

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

            # Strip dora_scale keys
            state_dict = {k: v for k, v in state_dict.items() if "dora_scale" not in k}

            # Detect LoKr format → use direct weight merge (lossless, no PEFT adapter)
            if _is_lokr_format(state_dict):
                self._lokr_entries = _parse_lokr_entries(state_dict, transformer)
                self._is_lokr = True
                self._lokr_scale = 0.0
                self._lokr_transformer_id = None
                self._update_lokr_merge(transformer)
                self.values["lora"] = (None, None, self._get_spatial_mask(), self.trigger_words)
                return self.values

            # Convert non-diffusers LoRA keys based on the target model.
            state_dict = _convert_lora_to_diffusers(state_dict, transformer)

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

    def _update_lokr_merge(self, transformer):
        """Update LoKr weight merge to match current scale/enabled state."""
        current_id = id(transformer)

        # If transformer object changed (model switch), previous merge is gone
        if self._lokr_transformer_id != current_id:
            self._lokr_scale = 0.0
            self._lokr_transformer_id = current_id
            self._lokr_original_weights = {}

        target_scale = self.transformer_weight if self.lora_enabled else 0.0

        if abs(target_scale - self._lokr_scale) > 1e-8:
            old_scale = self._lokr_scale
            _apply_lokr_merge(
                transformer, self._lokr_entries, target_scale, self._lokr_original_weights
            )
            self._lokr_scale = target_scale
            logger.info("[LoKr] Weight merge updated: scale %.4f → %.4f",
                        old_scale, target_scale)

    def _get_spatial_mask(self) -> Optional[torch.Tensor]:
        """Load and return spatial mask if enabled, or None."""
        if self.spatial_mask_enabled and self._cached_mask is None:
            self._cached_mask = self._load_spatial_mask()
        return self._cached_mask if self.spatial_mask_enabled else None

    def before_delete(self):
        mm = get_model_manager()
        try:
            transformer = mm.resolve(self.transformer)
        except Exception:
            if self._is_lokr and self._lokr_scale != 0.0:
                logger.error(
                    "[LoKr] Cannot resolve transformer in before_delete — "
                    "LoKr weights at scale %.4f will NOT be restored!",
                    self._lokr_scale,
                )
            return

        if self._is_lokr:
            # Restore original weights (scale → 0)
            if (
                self._lokr_scale != 0.0
                and self._lokr_entries
                and id(transformer) == self._lokr_transformer_id
            ):
                try:
                    _apply_lokr_merge(
                        transformer, self._lokr_entries, 0.0, self._lokr_original_weights
                    )
                    logger.info("[LoKr] Restored original weights (scale was %.4f)", self._lokr_scale)
                except Exception:
                    logger.warning("[LoKr] Failed to restore weights on delete")
            self._lokr_entries = None
            self._lokr_scale = 0.0
            self._lokr_original_weights = {}
            self._is_lokr = False
            return

        try:
            transformer.delete_adapters([self.adapter_name])
        except Exception:
            pass

        if self.adapter_name:
            mm.clear_lora_source(self.adapter_name)
