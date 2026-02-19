import logging
import os
import re
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

    Mixed-format files (some layers already using lora_A/lora_B, others using
    lora_down/lora_up) are handled by preserving existing lora_A/lora_B keys.
    """
    if not any("lora_down" in k for k in state_dict):
        return state_dict

    normalized = {}

    # Preserve keys that already use lora_A/lora_B naming (mixed-format files)
    for k, v in state_dict.items():
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            normalized[k] = v

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


def _is_lokr_format(state_dict: dict) -> bool:
    """Check if state dict contains LoKr-format weights (lokr_w1/lokr_w2)."""
    return any("lokr_w" in k for k in state_dict)


def _map_zimage_lokr_layer_to_targets(layer_prefix: str) -> list[tuple[str, int | None]]:
    """Map a Z-Image LoKr layer prefix to diffusers model parameter targets.

    Z-Image LoKr keys (after stripping diffusion_model. prefix) already match
    the diffusers parameter names, so this is a 1:1 identity mapping.
    """
    if re.match(r"layers\.\d+\.", layer_prefix):
        return [(layer_prefix, None)]
    if re.match(r"(context_refiner|noise_refiner)\.\d+\.", layer_prefix):
        return [(layer_prefix, None)]
    return []


def _map_flux2_lokr_layer_to_targets(layer_prefix: str) -> list[tuple[str, int | None]]:
    """Map a Flux2 LoKr layer prefix to diffusers model parameter targets.

    Returns list of (module_path, split_idx) tuples.
    split_idx is None for 1:1 mappings, or 0/1/2 for QKV chunk index.
    """
    # Single blocks: 1:1 mapping
    m = re.match(r"single_blocks\.(\d+)\.linear1$", layer_prefix)
    if m:
        return [(f"single_transformer_blocks.{m.group(1)}.attn.to_qkv_mlp_proj", None)]

    m = re.match(r"single_blocks\.(\d+)\.linear2$", layer_prefix)
    if m:
        return [(f"single_transformer_blocks.{m.group(1)}.attn.to_out", None)]

    # Double blocks: fused QKV → split into Q, K, V
    m = re.match(r"double_blocks\.(\d+)\.img_attn\.qkv$", layer_prefix)
    if m:
        tb = f"transformer_blocks.{m.group(1)}"
        return [(f"{tb}.attn.to_q", 0), (f"{tb}.attn.to_k", 1), (f"{tb}.attn.to_v", 2)]

    m = re.match(r"double_blocks\.(\d+)\.txt_attn\.qkv$", layer_prefix)
    if m:
        tb = f"transformer_blocks.{m.group(1)}"
        return [(f"{tb}.attn.add_q_proj", 0), (f"{tb}.attn.add_k_proj", 1), (f"{tb}.attn.add_v_proj", 2)]

    # Double blocks: projection and MLP (1:1)
    simple_patterns = [
        (r"double_blocks\.(\d+)\.img_attn\.proj$", "transformer_blocks.{}.attn.to_out.0"),
        (r"double_blocks\.(\d+)\.txt_attn\.proj$", "transformer_blocks.{}.attn.to_add_out"),
        (r"double_blocks\.(\d+)\.img_mlp\.0$", "transformer_blocks.{}.ff.linear_in"),
        (r"double_blocks\.(\d+)\.img_mlp\.2$", "transformer_blocks.{}.ff.linear_out"),
        (r"double_blocks\.(\d+)\.txt_mlp\.0$", "transformer_blocks.{}.ff_context.linear_in"),
        (r"double_blocks\.(\d+)\.txt_mlp\.2$", "transformer_blocks.{}.ff_context.linear_out"),
    ]
    for pattern, template in simple_patterns:
        m = re.match(pattern, layer_prefix)
        if m:
            return [(template.format(m.group(1)), None)]

    # Root-level mappings
    root_map = {
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
    if layer_prefix in root_map:
        return [(root_map[layer_prefix], None)]

    return []


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


def _apply_lokr_delta(model, entries: list[dict], scale_delta: float):
    """Apply LoKr weight deltas to model parameters.

    Computes kron(w1, w2) per layer and adds scale_delta * delta to weights.
    QKV-fused layers are split into separate Q/K/V parameter updates.
    """
    for entry in entries:
        delta = torch.kron(entry["w1"], entry["w2"].float())

        for module_path, split_idx in entry["targets"]:
            weight = _get_module_weight(model, module_path)
            if weight is None:
                logger.warning("[LoKr] Parameter not found: %s", module_path)
                continue

            if split_idx is not None:
                chunk = delta.chunk(3, dim=0)[split_idx]
            else:
                chunk = delta

            weight.data.add_(chunk.to(device=weight.device, dtype=weight.dtype) * scale_delta)


def _strip_lora_unet_prefix(sd: dict) -> dict:
    """Convert kohya lora_unet_ prefixed keys to raw Flux2 block format.

    Examples:
        lora_unet_single_blocks_8_linear1.lora_A.weight
            → single_blocks.8.linear1.lora_A.weight
        lora_unet_double_blocks_5_img_attn_qkv.lora_A.weight
            → double_blocks.5.img_attn.qkv.lora_A.weight
    """
    lora_unet = "lora_unet_"
    result = {}
    for k, v in sd.items():
        if not k.startswith(lora_unet):
            result[k] = v
            continue
        remainder = k[len(lora_unet):]  # e.g. "single_blocks_8_linear1.lora_A.weight"
        dot_pos = remainder.find(".")
        layer_path = remainder[:dot_pos] if dot_pos >= 0 else remainder
        suffix = remainder[dot_pos:] if dot_pos >= 0 else ""
        m = re.match(r"^(single_blocks|double_blocks)_(\d+)_(.+)$", layer_path)
        if m:
            block_type, idx, rest = m.groups()
            # For double blocks, compound names like img_attn/txt_attn/img_mlp/txt_mlp
            # keep their underscore; only the separator after them becomes a dot.
            # e.g. img_attn_qkv → img_attn.qkv, txt_mlp_0 → txt_mlp.0
            # For single blocks, rest is just linear1/linear2 — no substitution needed.
            converted_rest = re.sub(r"^(img|txt)_(attn|mlp)_", r"\1_\2.", rest)
            result[f"{block_type}.{idx}.{converted_rest}{suffix}"] = v
        else:
            result[k] = v  # Unknown format, keep as-is
    return result


def _convert_flux2_lora_to_diffusers(state_dict: dict) -> dict:
    """Convert non-diffusers Flux2 LoRA to diffusers PEFT format.

    Unlike diffusers' built-in converter which hardcodes 48 single blocks (full Flux2),
    this dynamically detects block counts from the keys — works for Klein 9B (24 single),
    Klein 4B, or any other variant. Also handles kohya-format lora_down/lora_up naming
    and mixed-format files (e.g. double blocks in diffusion_model. format, single blocks
    in lora_unet_ kohya format).
    """
    converted = {}

    # Strip diffusion_model. prefix, then normalise any remaining lora_unet_ prefixes
    prefix = "diffusion_model."
    sd = {k[len(prefix) :] if k.startswith(prefix) else k: v for k, v in state_dict.items()}
    sd = _strip_lora_unet_prefix(sd)

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


def _is_diffusers_format(state_dict: dict) -> bool:
    """Check if state dict is already in diffusers PEFT format.

    Diffusers PEFT keys start with ``transformer.`` and use ``lora_A``/``lora_B``
    naming.  Non-diffusers formats use other prefixes (``diffusion_model.``,
    ``lora_unet``), kohya naming (``lora_down``/``lora_up``), or ``.alpha`` keys.
    A file can have a ``transformer.`` prefix but still use kohya lora_down/lora_up
    naming — that is NOT diffusers format and needs conversion.
    """
    return (
        all(k.startswith("transformer.") for k in state_dict)
        and not any("lora_down" in k for k in state_dict)
    )


def _convert_zimage_lora(state_dict: dict) -> dict:
    """Convert a non-diffusers Z-Image LoRA to diffusers PEFT format.

    Handles kohya format (lora_unet__ prefixed keys with lora_down/lora_up)
    and other non-diffusers formats (alpha keys, diffusion_model. prefix, etc.).
    Already-diffusers-format LoRAs are returned as-is.
    """
    if _is_diffusers_format(state_dict):
        return state_dict
    return _convert_non_diffusers_z_image_lora_to_diffusers(state_dict)


def _convert_flux2_lora(state_dict: dict) -> dict:
    """Convert a non-diffusers Flux2 LoRA to diffusers PEFT format.

    Handles kohya format (lora_down/lora_up naming), original format
    (double_blocks/single_blocks), and already-diffusers-format LoRAs.
    """
    if _is_diffusers_format(state_dict):
        return state_dict

    # Normalize kohya lora_down/lora_up → lora_A/lora_B
    state_dict = _normalize_flux2_lora_keys(state_dict)

    # Original format uses double_blocks/single_blocks (not transformer_blocks/single_transformer_blocks)
    is_original_format = any(
        "double_blocks." in k or ("single_blocks." in k and "single_transformer_blocks." not in k)
        for k in state_dict
    )
    if is_original_format:
        state_dict = _convert_flux2_lora_to_diffusers(state_dict)

    return state_dict


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

        target_scale = self.transformer_weight if self.lora_enabled else 0.0
        scale_delta = target_scale - self._lokr_scale

        if abs(scale_delta) > 1e-8:
            _apply_lokr_delta(transformer, self._lokr_entries, scale_delta)
            self._lokr_scale = target_scale
            logger.info("[LoKr] Weight merge updated: scale %.4f → %.4f",
                        target_scale - scale_delta, target_scale)

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
            return

        if self._is_lokr:
            # Unmerge LoKr weight delta
            if (
                self._lokr_scale != 0.0
                and self._lokr_entries
                and id(transformer) == self._lokr_transformer_id
            ):
                try:
                    _apply_lokr_delta(transformer, self._lokr_entries, -self._lokr_scale)
                    logger.info("[LoKr] Unmerged weights (scale was %.4f)", self._lokr_scale)
                except Exception:
                    logger.warning("[LoKr] Failed to unmerge weights on delete")
            self._lokr_entries = None
            self._lokr_scale = 0.0
            self._is_lokr = False
            return

        try:
            transformer.delete_adapters([self.adapter_name])
        except Exception:
            pass

        if self.adapter_name:
            mm.clear_lora_source(self.adapter_name)
