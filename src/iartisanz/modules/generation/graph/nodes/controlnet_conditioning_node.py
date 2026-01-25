from __future__ import annotations

from typing import Any

import numpy as np
import torch
from diffusers.image_processor import VaeImageProcessor
from PIL import Image

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


def _retrieve_latents(encoder_output: Any, *, sample_mode: str = "sample") -> torch.Tensor:
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample()
    if hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    raise AttributeError("Could not access latents of provided encoder_output")


def _extract_alpha_mask(image: Any) -> Image.Image | None:
    """Extract alpha channel as mask (transparent → white/255, opaque → black/0).

    Args:
        image: PIL Image, numpy array, or torch tensor

    Returns:
        PIL Image (mode='L') with inverted alpha, or None if no alpha channel
    """
    # PIL Image with alpha
    if hasattr(image, "mode") and image.mode == "RGBA":
        from PIL import ImageOps

        alpha = image.split()[-1]  # Get alpha channel
        mask = ImageOps.invert(alpha)  # Invert: 0 (transparent) → 255 (mask)
        return mask

    # Torch tensor with alpha (BCHW format with 4 channels)
    if isinstance(image, torch.Tensor) and image.ndim == 4 and image.shape[1] == 4:
        alpha = image[:, 3:4, :, :]  # Extract alpha channel
        mask = 1.0 - alpha  # Invert: 0 (transparent) → 1.0 (mask)
        # Convert to PIL for consistency
        mask_np = (mask.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(mask_np, mode="L")

    # Numpy array with alpha
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:  # HWC format
            alpha = image[:, :, 3]
            mask = 255 - alpha  # Invert
            return Image.fromarray(mask.astype(np.uint8), mode="L")
        elif image.ndim == 4 and image.shape[1] == 4:  # BCHW format
            alpha = image[0, 3, :, :]
            mask = 255 - alpha  # Invert
            return Image.fromarray(mask.astype(np.uint8), mode="L")

    return None


def _composite_rgba_to_rgb(image: Any, background_color=(127, 127, 127)) -> Any:
    """Composite RGBA image onto solid background to remove alpha channel.

    Args:
        image: PIL Image, numpy array, or torch tensor
        background_color: RGB tuple (0-255) for background

    Returns:
        Same type as input, with alpha removed
    """
    # PIL Image with alpha
    if hasattr(image, "mode") and image.mode == "RGBA":
        bg = Image.new("RGB", image.size, background_color)
        bg.paste(image, mask=image.split()[-1])
        return bg

    # Torch tensor with alpha (BCHW format)
    if isinstance(image, torch.Tensor) and image.ndim == 4 and image.shape[1] == 4:
        alpha = image[:, 3:4, :, :]
        rgb = image[:, :3, :, :]
        bg_tensor = torch.tensor(background_color, device=image.device, dtype=image.dtype).view(1, 3, 1, 1) / 255.0
        bg = bg_tensor.expand_as(rgb)
        composited = rgb * alpha + bg * (1.0 - alpha)
        return composited

    # Numpy array with alpha
    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[2] == 4:  # HWC format
            alpha = image[:, :, 3:4] / 255.0
            rgb = image[:, :, :3]
            bg = np.full_like(rgb, background_color)
            composited = (rgb * alpha + bg * (1.0 - alpha)).astype(np.uint8)
            return composited

    return image


def _union_masks(
    mask1: Image.Image | torch.Tensor | None, mask2: Image.Image | torch.Tensor | None
) -> Image.Image | torch.Tensor | None:
    """Union two masks: white where either mask is white.

    Args:
        mask1: First mask (PIL Image, torch tensor, or None)
        mask2: Second mask (PIL Image, torch tensor, or None)

    Returns:
        Unioned mask (same type as mask1), or None if both are None
    """
    if mask1 is None:
        return mask2
    if mask2 is None:
        return mask1

    # Convert both to numpy for union
    if isinstance(mask1, torch.Tensor):
        # Torch tensor: squeeze to 2D if needed
        m1 = mask1.squeeze().cpu().numpy()
        if m1.dtype == np.float32 or m1.dtype == np.float64:
            m1 = (m1 * 255).astype(np.uint8)
        is_tensor_result = True
        original_shape = mask1.shape
    elif isinstance(mask1, Image.Image):
        m1 = np.array(mask1)
        is_tensor_result = False
    else:
        m1 = np.array(mask1)
        is_tensor_result = False

    if isinstance(mask2, Image.Image):
        m2 = np.array(mask2)
        # Ensure same size
        if m2.shape != m1.shape:
            mask2_pil = mask2 if isinstance(mask2, Image.Image) else Image.fromarray(m2, mode="L")
            target_size = (m1.shape[1], m1.shape[0])  # PIL uses (width, height)
            mask2_pil = mask2_pil.resize(target_size, Image.Resampling.NEAREST)
            m2 = np.array(mask2_pil)
    elif isinstance(mask2, torch.Tensor):
        m2 = mask2.squeeze().cpu().numpy()
        if m2.dtype == np.float32 or m2.dtype == np.float64:
            m2 = (m2 * 255).astype(np.uint8)
        # Ensure same size
        if m2.shape != m1.shape:
            m2_pil = Image.fromarray(m2, mode="L")
            target_size = (m1.shape[1], m1.shape[0])
            m2_pil = m2_pil.resize(target_size, Image.Resampling.NEAREST)
            m2 = np.array(m2_pil)
    else:
        m2 = np.array(mask2)

    # Union
    union = np.maximum(m1, m2)

    # Return in original format
    if is_tensor_result:
        union_float = union.astype(np.float32) / 255.0
        result = torch.from_numpy(union_float)
        # Restore original shape
        while result.ndim < len(original_shape):
            result = result.unsqueeze(0)
        return result
    else:
        return Image.fromarray(union.astype(np.uint8), mode="L")


class ControlNetConditioningNode(Node):
    """Prepares a control image latent tensor for Z-Image ControlNet with differential diffusion awareness.

    Output is a CPU tensor shaped (B, C, 1, H, W) (the extra dim is 'F').
    DenoiseNode is responsible for duplicating for CFG and for any channel
    padding required by specific ControlNet versions.

    Supports multiple modes with differential diffusion awareness:

    WITHOUT differential diffusion:
    - Standard ControlNet: control_image only
    - Spatial ControlNet: control_image + mask_image (spatial restriction)
    - Inpainting: control_image + mask_image + init_image (33-channel context)
    - Inpainting-only: mask_image + init_image (no control)

    WITH differential diffusion (source_mask present):
    - ControlNet + Diff Diff: control_image + source → 16ch, spatial=source_mask
    - Full Pipeline: control_image + source + cn_mask → 16ch, spatial=cn_mask
    - Diff Diff + Spatial: source + cn_mask → 16ch (source as base), spatial=cn_mask

    Key: When differential diffusion is active, controlnet_mask becomes a SPATIAL
    RESTRICTION (where ControlNet applies) rather than an inpainting boundary.
    This avoids conflicting mask semantics.

    Supports automatic alpha channel extraction: images with transparency automatically
    generate masks (transparent regions → masked). Alpha masks are unioned
    with any explicit mask provided.

    init_image is typically sourced from SourceImageDialog (source_image),
    not from ControlNet-specific UI.

    This node is optional and not included in the default graph.
    """

    PRIORITY = 1
    REQUIRED_INPUTS = ["vae", "width", "height", "vae_scale_factor"]
    OPTIONAL_INPUTS = [
        "control_image",
        "mask_image",  # ControlNet mask
        "init_image",  # Source image
        "differential_diffusion_active",  # Boolean flag
    ]
    OUTPUTS = [
        "control_image_latents",
        "control_image_size",
        "spatial_mask",  # For spatial restriction in DenoiseNode
        "control_mode",  # For debugging/logging
    ]

    SERIALIZE_EXCLUDE = {"control_image_latents", "spatial_mask"}

    def __init__(self, alpha_composite_color=(127, 127, 127)):
        """Initialize the node.

        Args:
            alpha_composite_color: RGB tuple for background when compositing RGBA→RGB.
                                  Default is gray (127, 127, 127).
        """
        super().__init__()
        self.alpha_composite_color = alpha_composite_color

    def _process_mask_to_pixel(self, mask_image, width, height, vae):
        """Convert mask to pixel-space tensor [B, 1, H, W] in [0, 1].

        Args:
            mask_image: Mask as PIL Image, numpy array, or torch tensor
            width: Target width
            height: Target height
            vae: VAE model for determining device

        Returns:
            Torch tensor [B, 1, H, W] with binary mask values
        """
        if isinstance(mask_image, torch.Tensor):
            mask_pixel = mask_image
            if mask_pixel.ndim == 2:
                mask_pixel = mask_pixel.unsqueeze(0).unsqueeze(0)
            elif mask_pixel.ndim == 3:
                mask_pixel = mask_pixel.unsqueeze(0)

            # Normalize to 1-channel BCHW float mask in [0, 1]
            if mask_pixel.shape[1] != 1:
                mask_pixel = mask_pixel.mean(dim=1, keepdim=True)

            if mask_pixel.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                mask_pixel = mask_pixel.float() / 255.0
            else:
                mask_pixel = mask_pixel.float()

            mask_pixel = (mask_pixel > 0.5).float()
        else:
            # Use Diffusers processor when given PIL/np-like inputs.
            pil_mask = mask_image
            if not isinstance(pil_mask, Image.Image):
                if isinstance(pil_mask, np.ndarray):
                    np_mask = pil_mask
                else:
                    np_mask = np.array(pil_mask)

                # Squeeze out singleton dimensions for PIL compatibility
                np_mask = np.squeeze(np_mask)

                # Ensure 2D array for grayscale
                if np_mask.ndim > 2:
                    # If still has extra dims, take first channel
                    np_mask = np_mask[..., 0] if np_mask.shape[-1] <= 4 else np_mask[0]

                if np_mask.dtype != np.uint8:
                    # Check if mask is in [0, 1] range (float) and scale to [0, 255]
                    # Otherwise assume it's already in [0, 255] range
                    if np_mask.max() <= 1.0:
                        np_mask = (np_mask * 255).clip(0, 255).astype(np.uint8)
                    else:
                        np_mask = np_mask.clip(0, 255).astype(np.uint8)
                pil_mask = Image.fromarray(np_mask, mode="L")

            try:
                mask_processor = VaeImageProcessor(
                    vae_scale_factor=int(self.vae_scale_factor) * 2,
                    do_binarize=True,
                    do_convert_grayscale=True,
                )
            except TypeError:
                # Older/newer diffusers signatures vary slightly.
                mask_processor = VaeImageProcessor(vae_scale_factor=int(self.vae_scale_factor) * 2)

            mask_pixel = mask_processor.preprocess(pil_mask, height=height, width=width)
            if mask_pixel.shape[1] != 1:
                mask_pixel = mask_pixel.mean(dim=1, keepdim=True)
            mask_pixel = (mask_pixel > 0.5).float()

        return mask_pixel

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        vae = mm.resolve(self.vae)

        width = int(self.width)
        height = int(self.height)

        def _to_pil_rgb(image_like: object) -> Image.Image:
            if isinstance(image_like, Image.Image):
                pil_img = image_like
            else:
                if isinstance(image_like, np.ndarray):
                    np_img = image_like
                else:
                    np_img = np.array(image_like)

                if np_img.ndim == 2:
                    np_img = np.stack([np_img, np_img, np_img], axis=-1)

                if np_img.dtype != np.uint8:
                    np_img = np.clip(np_img, 0, 255).astype(np.uint8)

                pil_img = Image.fromarray(np_img)

            if pil_img.mode != "RGB":
                pil_img = pil_img.convert("RGB")
            return pil_img

        def _preprocess_image_tensor(image_like: object) -> torch.Tensor:
            if isinstance(image_like, torch.Tensor):
                image_tensor = image_like
                # Accept CHW/HW inputs in tests; normalize to BCHW.
                if image_tensor.ndim == 2:
                    image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
                elif image_tensor.ndim == 3:
                    image_tensor = image_tensor.unsqueeze(0)
                return image_tensor

            pil = _to_pil_rgb(image_like)
            processor = VaeImageProcessor(vae_scale_factor=int(self.vae_scale_factor) * 2)
            return processor.preprocess(pil, height=height, width=width)

        def _encode_image_latents(image_like: object) -> torch.Tensor:
            image_tensor = _preprocess_image_tensor(image_like)

            # Encode on the VAE device (ModelManager scope decides CPU/GPU).
            try:
                vae_device = next(vae.parameters()).device
            except Exception:
                vae_device = torch.device("cpu")
            image_tensor = image_tensor.to(device=vae_device, dtype=vae.dtype)

            try:
                latents = _retrieve_latents(vae.encode(image_tensor), sample_mode="argmax")
            except Exception as e:
                raise IArtisanZNodeError(f"Failed encoding control image: {e}", self.__class__.__name__) from e

            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
            return latents

        # Get inputs
        control_image = getattr(self, "control_image", None)
        mask_image = getattr(self, "mask_image", None)
        init_image = getattr(self, "init_image", None)
        diff_diff_active = getattr(self, "differential_diffusion_active", False)

        # Extract alpha masks from images (transparent → mask)
        control_alpha_mask = _extract_alpha_mask(control_image) if control_image is not None else None
        init_alpha_mask = _extract_alpha_mask(init_image) if init_image is not None else None

        # Union all masks: explicit mask + control alpha + init alpha
        final_mask = mask_image
        if control_alpha_mask is not None:
            final_mask = _union_masks(final_mask, control_alpha_mask)
        if init_alpha_mask is not None:
            final_mask = _union_masks(final_mask, init_alpha_mask)
        mask_image = final_mask

        # Composite RGBA images to RGB (remove alpha channels)
        if control_image is not None and _extract_alpha_mask(control_image) is not None:
            control_image = _composite_rgba_to_rgb(control_image, self.alpha_composite_color)
        if init_image is not None and _extract_alpha_mask(init_image) is not None:
            init_image = _composite_rgba_to_rgb(init_image, self.alpha_composite_color)

        # Determine mode flags
        has_control = control_image is not None
        has_mask = mask_image is not None
        has_init = init_image is not None

        # === DECISION TREE: DIFFERENTIAL DIFFUSION MODE ===
        if diff_diff_active:
            # Don't use 33-channel inpainting context
            # Use controlnet_mask as spatial restriction only

            if has_control:
                # Scenario 5 or 6: ControlNet + Diff Diff
                base_image = control_image
                control_latents = _encode_image_latents(base_image)
                control_latents_5d = control_latents.unsqueeze(2)  # 16 channels
                spatial_mask = mask_image if has_mask else None  # cn_mask → spatial
                control_mode = "controlnet_diff_diff"

            elif has_mask and has_init:
                # Scenario 10: Diff Diff + Spatial (no control_image)
                # Use source as control base for self-guidance
                base_image = init_image
                control_latents = _encode_image_latents(base_image)
                control_latents_5d = control_latents.unsqueeze(2)  # 16 channels
                spatial_mask = mask_image  # cn_mask → spatial restriction
                control_mode = "diff_diff_spatial"

            else:
                # Scenario 9: Diff Diff only (no ControlNet needed)
                raise IArtisanZNodeError(
                    "ControlNet not needed for differential diffusion without spatial mask",
                    self.__class__.__name__
                )

        # === DECISION TREE: STANDARD OR INPAINTING MODE ===
        else:
            if has_mask and has_init:
                # Scenario 4 or 8: Inpainting mode (33-channel context)
                if has_control:
                    base_image = control_image
                    control_mode = "controlnet_inpainting"
                else:
                    base_image = init_image  # Inpainting-only
                    control_mode = "inpainting_only"

                # Encode base image
                control_latents = _encode_image_latents(base_image)
                control_latents_5d = control_latents.unsqueeze(2)

                # Process mask to pixel space
                mask_pixel = self._process_mask_to_pixel(mask_image, width, height, vae)

                # Apply pre-masking to init_image (official implementation)
                init_image_tensor = _preprocess_image_tensor(init_image)
                try:
                    vae_device = next(vae.parameters()).device
                except Exception:
                    vae_device = torch.device("cpu")

                mask_pixel = mask_pixel.to(device=vae_device, dtype=vae.dtype)

                # Ensure mask matches init_image dimensions
                if mask_pixel.shape[-2:] != init_image_tensor.shape[-2:]:
                    mask_pixel = torch.nn.functional.interpolate(
                        mask_pixel,
                        size=(init_image_tensor.shape[-2], init_image_tensor.shape[-1]),
                        mode="nearest",
                    )

                # Apply inverted mask: preserve only non-masked regions
                init_image_tensor = init_image_tensor.to(device=vae_device, dtype=vae.dtype)
                init_image_masked = init_image_tensor * (1.0 - mask_pixel)

                # Encode the masked init_image
                try:
                    init_latents = _retrieve_latents(vae.encode(init_image_masked), sample_mode="argmax")
                except Exception as e:
                    raise IArtisanZNodeError(f"Failed encoding masked init image: {e}", self.__class__.__name__) from e

                init_latents = (init_latents - vae.config.shift_factor) * vae.config.scaling_factor
                init_latents_5d = init_latents.unsqueeze(2)

                # Downsample mask to latent resolution
                mask_tensor = torch.nn.functional.interpolate(
                    mask_pixel,
                    size=(int(control_latents.shape[-2]), int(control_latents.shape[-1])),
                    mode="nearest",
                )

                # Broadcast mask batch if needed
                if mask_tensor.shape[0] != control_latents.shape[0] and mask_tensor.shape[0] == 1:
                    mask_tensor = mask_tensor.expand(control_latents.shape[0], -1, -1, -1)

                mask_5d = mask_tensor.unsqueeze(2)

                # Build 33-channel context: [control_latents, mask, init_latents]
                control_latents_5d = torch.cat([control_latents_5d, mask_5d, init_latents_5d], dim=1)

                # Also apply spatially (optional - for stronger boundary enforcement)
                spatial_mask = mask_image

            elif has_mask and not has_init:
                # Scenario 2: Spatial ControlNet only (no source_image)
                if not has_control:
                    raise IArtisanZNodeError(
                        "Spatial mask requires control_image",
                        self.__class__.__name__
                    )
                base_image = control_image
                control_latents = _encode_image_latents(base_image)
                control_latents_5d = control_latents.unsqueeze(2)  # 16 channels
                spatial_mask = mask_image  # Spatial restriction
                control_mode = "spatial_controlnet"

            elif has_control:
                # Scenario 1 or 3: Standard ControlNet
                base_image = control_image
                control_latents = _encode_image_latents(base_image)
                control_latents_5d = control_latents.unsqueeze(2)  # 16 channels
                spatial_mask = None
                control_mode = "standard_controlnet"

            else:
                raise IArtisanZNodeError(
                    "Invalid ControlNet configuration",
                    self.__class__.__name__
                )

        # Return CPU tensor for downstream nodes (matches PromptEncoderNode/DenoiseNode conventions).
        self.values["control_image_latents"] = control_latents_5d.detach().to("cpu")

        # Output spatial mask for DenoiseNode (may be None)
        import logging
        logger = logging.getLogger(__name__)

        if spatial_mask is not None:
            # Convert to tensor if needed
            if not isinstance(spatial_mask, torch.Tensor):
                spatial_mask_tensor = self._process_mask_to_pixel(spatial_mask, width, height, vae)
            else:
                spatial_mask_tensor = spatial_mask
            self.values["spatial_mask"] = spatial_mask_tensor.detach().to("cpu")
        else:
            self.values["spatial_mask"] = None

        # Output control mode for debugging/logging
        self.values["control_mode"] = control_mode

        # Best-effort size reporting.
        try:
            control_tensor_for_size = _preprocess_image_tensor(base_image)
            self.values["control_image_size"] = (
                int(control_tensor_for_size.shape[-1]),
                int(control_tensor_for_size.shape[-2]),
            )
        except Exception:
            self.values["control_image_size"] = (width, height)
        return self.values
