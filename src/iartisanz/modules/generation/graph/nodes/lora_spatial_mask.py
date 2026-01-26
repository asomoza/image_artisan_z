"""Spatial masking for LoRA adapters.

This module provides utilities for applying spatial masks to LoRA layers by
patching their forward methods. The approach is similar to ControlNet spatial
masking but operates on LoRA adapter outputs within the transformer.

The mask restricts where LoRA effects apply:
- Mask value 1.0: LoRA applies fully
- Mask value 0.0: LoRA is blocked (base model only)
- Intermediate values: Partial LoRA application
"""

import logging
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


logger = logging.getLogger(__name__)


# Global registry to track patched layers and their original forward methods
_PATCHED_LAYERS: Dict[int, Tuple[nn.Module, callable]] = {}


def patch_lora_layer_with_spatial_mask(
    layer: nn.Module,
    spatial_mask: torch.Tensor,
    spatial_dims: Tuple[int, int],
    latent_spatial_dims: Optional[Tuple[int, int]] = None,
) -> None:
    """Patch a LoRA layer to apply spatial mask during forward pass.

    This modifies the layer's forward method to multiply LoRA outputs by a
    spatial mask, effectively restricting where the LoRA adapter applies.

    Args:
        layer: The LoRA layer to patch (e.g., lora_A, lora_B, or combined)
        spatial_mask: Mask tensor [1, 1, H, W] or [B, 1, H, W]
                     Values should be 0.0 (block) or 1.0 (apply)
        spatial_dims: (height, width) - hint for spatial dimensions (may be
                     overridden based on actual hidden_states at runtime)
        latent_spatial_dims: Optional (height, width) of the actual latent space.
                            When provided, this is used directly instead of
                            inferring from sequence length. Essential for
                            non-square aspect ratios.

    Note:
        - Mask is resized dynamically to match actual sequence dimensions
        - For joint attention (image + text tokens), only image tokens are masked
        - Original forward method is stored for later restoration

    Raises:
        ValueError: If spatial_mask has invalid dimensions
    """
    # Validate mask shape
    if spatial_mask.ndim not in [3, 4]:
        raise ValueError(f"Spatial mask must be 3D or 4D, got {spatial_mask.ndim}D (shape={spatial_mask.shape})")

    # Ensure mask is 4D: [B, 1, H, W]
    if spatial_mask.ndim == 3:
        spatial_mask = spatial_mask.unsqueeze(0)

    # Store original forward method
    layer_id = id(layer)
    if layer_id not in _PATCHED_LAYERS:
        original_forward = layer.forward
        _PATCHED_LAYERS[layer_id] = (layer, original_forward)
    else:
        # Already patched, get original from registry
        _, original_forward = _PATCHED_LAYERS[layer_id]

    # Store original 4D mask (don't pre-flatten - we'll resize on-the-fly)
    original_mask = spatial_mask.clone()
    # Store latent spatial dims for non-square aspect ratios
    stored_latent_dims = latent_spatial_dims

    def masked_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        """Modified forward that applies spatial mask to LoRA output.

        Args:
            hidden_states: Input tensor [B, N, D] where N may be H*W (spatial)
                          or H*W + text_tokens (joint attention),
                          or [B, D] for non-spatial layers (e.g., adaLN_modulation)

        Returns:
            Masked LoRA output (same shape as input)
        """
        # Apply original LoRA computation
        lora_output = original_forward(hidden_states)

        # Only apply spatial masking to 3D tensors (spatial hidden states)
        # 2D tensors (like adaLN conditioning) pass through unchanged since
        # spatial masking doesn't make sense for non-spatial inputs
        if hidden_states.ndim != 3:
            return lora_output

        B, N, D = hidden_states.shape

        # Use provided latent dimensions if available (essential for non-square aspect ratios)
        # Otherwise fall back to inference from sequence length
        if stored_latent_dims is not None:
            H, W = stored_latent_dims
            num_image_tokens = H * W

            # Check if N matches H*W (pure spatial) or H*W + text_tokens (joint attention)
            # Also check for packed representation (H/2 * W/2)
            packed_tokens = (H // 2) * (W // 2)

            if N < packed_tokens:
                # Even packed doesn't fit - fall back to inference
                H, W, num_image_tokens = _infer_spatial_from_sequence_length(N)
            elif N < num_image_tokens:
                # Sequence suggests packed representation (2x2 patches)
                H, W = H // 2, W // 2
                num_image_tokens = H * W
        else:
            # Infer spatial dimensions from sequence length
            # For joint attention, N might be H*W + text_tokens
            H, W, num_image_tokens = _infer_spatial_from_sequence_length(N)

        if H is None:
            # Can't determine spatial dims, skip masking for this layer
            logger.warning(f"[LoRA Mask] Could not determine spatial dims for N={N}, skipping mask")
            return lora_output

        # Resize mask to inferred spatial dimensions
        resized_mask = F.interpolate(original_mask.float(), size=(H, W), mode="bilinear", align_corners=False)

        # Flatten mask: [B_m, 1, H, W] -> [B_m, H*W, 1]
        B_m = resized_mask.shape[0]
        flat_mask = resized_mask.view(B_m, 1, H * W).transpose(1, 2)

        # Handle batch size mismatch
        if B != B_m:
            flat_mask = flat_mask.expand(B, -1, -1)

        # Move mask to correct device/dtype
        flat_mask = flat_mask.to(device=lora_output.device, dtype=lora_output.dtype)

        # If joint attention (N > num_image_tokens), only mask image tokens
        if N > num_image_tokens:
            # Create full mask with ones for text tokens (don't mask them)
            full_mask = torch.ones(B, N, 1, device=lora_output.device, dtype=lora_output.dtype)
            full_mask[:, :num_image_tokens, :] = flat_mask
            return lora_output * full_mask
        else:
            return lora_output * flat_mask

    # Replace forward method
    layer.forward = masked_forward

    logger.debug(f"Patched LoRA layer {layer.__class__.__name__} with spatial mask (hint spatial_dims={spatial_dims})")


def _infer_spatial_from_sequence_length(N: int) -> Tuple[Optional[int], Optional[int], int]:
    """Infer spatial dimensions (H, W) from sequence length.

    Handles both pure spatial sequences (N = H*W) and joint attention
    sequences (N = H*W + text_tokens).

    Args:
        N: Sequence length

    Returns:
        (H, W, num_image_tokens) tuple, or (None, None, 0) if can't infer
    """
    # Check if N is a perfect square (pure spatial)
    side = int(N**0.5)
    if side * side == N:
        return side, side, N

    # N is not a perfect square - might be joint attention
    # Try common spatial sizes (latent space dimensions)
    # For 1024x1024 image with 8x VAE compression: 128x128 = 16384
    # For 512x512 image: 64x64 = 4096
    # For 768x768 image: 96x96 = 9216
    common_spatial_sizes = [
        (128, 128),  # 1024x1024 images
        (64, 64),  # 512x512 images
        (96, 96),  # 768x768 images
        (48, 48),  # 384x384 images
        (32, 32),  # 256x256 images
        (80, 80),  # 640x640 images
        (112, 112),  # 896x896 images
        (144, 144),  # 1152x1152 images
    ]

    for H, W in common_spatial_sizes:
        num_image_tokens = H * W
        if N >= num_image_tokens:
            # Check if remainder is reasonable for text tokens (< 1024)
            text_tokens = N - num_image_tokens
            if text_tokens < 1024:
                return H, W, num_image_tokens

    # Last resort: find largest square that fits
    side = int(N**0.5)
    if side > 0:
        num_image_tokens = side * side
        text_tokens = N - num_image_tokens
        if text_tokens < 1024:  # Reasonable text token count
            return side, side, num_image_tokens

    # Can't determine spatial dimensions
    return None, None, 0


def unpatch_lora_layer(layer: nn.Module) -> None:
    """Restore original forward method to a patched LoRA layer.

    Args:
        layer: The LoRA layer to unpatch

    Raises:
        ValueError: If layer was not patched
    """
    layer_id = id(layer)

    if layer_id not in _PATCHED_LAYERS:
        raise ValueError(f"Layer {layer.__class__.__name__} was not patched, cannot unpatch")

    # Restore original forward method
    _, original_forward = _PATCHED_LAYERS[layer_id]
    layer.forward = original_forward

    # Remove from registry
    del _PATCHED_LAYERS[layer_id]

    logger.debug(f"Unpatched LoRA layer {layer.__class__.__name__}")


def unpatch_all_lora_layers() -> int:
    """Restore all patched LoRA layers to their original state.

    This is useful for cleanup after generation to ensure no stale patches remain.

    Returns:
        Number of layers that were unpatched
    """
    # Get all layer IDs to unpatch (create list to avoid dict size change during iteration)
    layer_ids = list(_PATCHED_LAYERS.keys())

    for layer_id in layer_ids:
        layer, original_forward = _PATCHED_LAYERS[layer_id]
        layer.forward = original_forward

    count = len(_PATCHED_LAYERS)
    _PATCHED_LAYERS.clear()

    if count > 0:
        logger.debug(f"Unpatched {count} LoRA layers")

    return count


def get_patched_layer_count() -> int:
    """Get the number of currently patched layers.

    Returns:
        Number of layers in the patch registry
    """
    return len(_PATCHED_LAYERS)


def patch_lora_adapter_with_spatial_mask(
    transformer: nn.Module,
    adapter_name: str,
    spatial_mask: torch.Tensor,
    latent_spatial_dims: Optional[Tuple[int, int]] = None,
) -> int:
    """Patch all LoRA layers for a specific adapter with spatial mask.

    This is the high-level function used by DenoiseNode. It finds all LoRA
    layers belonging to the specified adapter and patches each one with the
    appropriate mask for that layer's spatial resolution.

    Args:
        transformer: The transformer model with loaded LoRA adapters
        adapter_name: Name of the LoRA adapter to patch
        spatial_mask: Mask tensor in image space [1, 1, H_img, W_img]
        latent_spatial_dims: Optional (height, width) of the latent space.
                            When provided, this is used directly to determine
                            spatial dimensions instead of inferring from
                            sequence length. Essential for non-square aspect
                            ratios like 1376x960 -> 172x120 latents.

    Returns:
        Number of layers that were patched

    Note:
        Each transformer layer may have different spatial resolutions (similar
        to ControlNet blocks). The mask is resized per-layer automatically.
    """
    # Get all LoRA layers for this adapter
    lora_layers = _get_lora_layers_for_adapter(transformer, adapter_name)

    if not lora_layers:
        logger.warning(f"No LoRA layers found for adapter '{adapter_name}'. Skipping spatial masking.")
        return 0

    # Infer default spatial dimensions from mask
    if spatial_mask.ndim == 3:
        default_spatial_dims = (spatial_mask.shape[-2], spatial_mask.shape[-1])
    else:
        default_spatial_dims = (spatial_mask.shape[-2], spatial_mask.shape[-1])

    # Patch each layer with appropriately sized mask
    patched_count = 0
    for layer_name, layer in lora_layers.items():
        # Infer spatial dimensions for this layer
        spatial_dims = _infer_spatial_dims_for_layer(layer, default_spatial_dims)

        # Patch this layer
        try:
            patch_lora_layer_with_spatial_mask(
                layer, spatial_mask, spatial_dims, latent_spatial_dims=latent_spatial_dims
            )
            patched_count += 1
        except Exception as e:
            logger.error(f"Failed to patch layer {layer_name}: {e}")

    logger.info(f"Patched {patched_count} LoRA layers for adapter '{adapter_name}' with spatial mask")

    return patched_count


def _get_lora_layers_for_adapter(transformer: nn.Module, adapter_name: str) -> Dict[str, nn.Module]:
    """Get all LoRA layers belonging to a specific adapter.

    This function navigates the transformer structure to find all LoRA layers
    associated with the given adapter name. The structure depends on how PEFT
    organizes adapters.

    Args:
        transformer: The transformer model with LoRA adapters
        adapter_name: Name of the adapter

    Returns:
        Dict mapping layer names to layer modules

    Note:
        This implementation handles both:
        - Test fixtures with get_lora_layers() method
        - Real PEFT models with named_modules() iteration
    """
    lora_layers = {}

    # Test fixture provides this method
    if hasattr(transformer, "get_lora_layers"):
        lora_layers = transformer.get_lora_layers(adapter_name)
    else:
        # Real implementation: search transformer for LoRA layers
        # PEFT typically stores adapters as:
        # - base_model.model.layers.N.self_attn.q_proj.lora_A.adapter_name
        # - base_model.model.layers.N.self_attn.q_proj.lora_B.adapter_name
        # We look for modules containing 'lora' in the name

        for name, module in transformer.named_modules():
            # Look for PEFT's LoRA module pattern
            name_lower = name.lower()
            if "lora" in name_lower:
                # Check if this is for the right adapter
                if adapter_name.lower() in name_lower or "default" in name_lower:
                    # Only include actual LoRA layers (with weights)
                    if hasattr(module, "weight") or hasattr(module, "lora_A"):
                        lora_layers[name] = module

    return lora_layers


def _infer_spatial_dims_for_layer(layer: nn.Module, default_dims: Tuple[int, int] = (64, 64)) -> Tuple[int, int]:
    """Infer spatial dimensions (H, W) for a transformer layer.

    Different layers in the transformer may operate at different spatial
    resolutions. This function infers the appropriate (H, W) for mask resizing.

    Args:
        layer: The LoRA layer
        default_dims: Default (height, width) if inference fails

    Returns:
        (height, width) tuple for this layer's spatial resolution

    Note:
        For Z-Image/Flux models, all attention layers typically operate at
        the same spatial resolution (latent space dimensions).
        Default is 64x64 which corresponds to 1024x1024 images.
    """
    # Check if layer has spatial_dims attribute (custom annotation)
    if hasattr(layer, "spatial_dims"):
        return layer.spatial_dims

    # Check if layer has a resolution hint
    if hasattr(layer, "resolution"):
        res = layer.resolution
        return (res, res)

    # Default to provided dimensions
    # For Z-Image at 1024x1024 -> latents are 128x128
    # For SDXL at 1024x1024 -> latents are 128x128
    return default_dims
