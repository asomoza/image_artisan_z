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


class ControlNetConditioningNode(Node):
    """Prepares a control image latent tensor for Z-Image ControlNet.

    Output is a CPU tensor shaped (B, C, 1, H, W) (the extra dim is 'F').
    DenoiseNode is responsible for duplicating for CFG and for any channel
    padding required by specific ControlNet versions.

    This node is optional and not included in the default graph.
    """

    PRIORITY = 1
    REQUIRED_INPUTS = ["vae", "control_image", "width", "height", "vae_scale_factor"]
    OPTIONAL_INPUTS = ["mask_image", "init_image"]
    OUTPUTS = ["control_image_latents", "control_image_size"]

    SERIALIZE_EXCLUDE = {"control_image_latents"}

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

        control_image = self.control_image
        control_latents = _encode_image_latents(control_image)

        control_latents_5d = control_latents.unsqueeze(2)

        # Optional inpaint conditioning: if a mask is connected, concatenate
        # [control_latents, mask_condition, init_image_latents] along channels.
        mask_image = getattr(self, "mask_image", None)
        init_image = getattr(self, "init_image", None)

        if mask_image is not None:
            # Default init image to control image if not provided.
            if init_image is None:
                init_image = control_image

            init_latents = _encode_image_latents(init_image)
            init_latents_5d = init_latents.unsqueeze(2)

            # Build mask condition in latent resolution. Prefer nearest interpolation.
            if isinstance(mask_image, torch.Tensor):
                mask_tensor = mask_image
                if mask_tensor.ndim == 2:
                    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0)
                elif mask_tensor.ndim == 3:
                    mask_tensor = mask_tensor.unsqueeze(0)

                # Normalize to 1-channel BCHW float mask in [0, 1]
                if mask_tensor.shape[1] != 1:
                    mask_tensor = mask_tensor.mean(dim=1, keepdim=True)

                if mask_tensor.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
                    mask_tensor = mask_tensor.float() / 255.0
                else:
                    mask_tensor = mask_tensor.float()

                mask_tensor = (mask_tensor > 0.5).float()
            else:
                # Use Diffusers processor when given PIL/np-like inputs.
                pil_mask = mask_image
                if not isinstance(pil_mask, Image.Image):
                    if isinstance(pil_mask, np.ndarray):
                        np_mask = pil_mask
                    else:
                        np_mask = np.array(pil_mask)

                    if np_mask.dtype != np.uint8:
                        np_mask = np.clip(np_mask, 0, 255).astype(np.uint8)
                    pil_mask = Image.fromarray(np_mask)

                try:
                    mask_processor = VaeImageProcessor(
                        vae_scale_factor=int(self.vae_scale_factor) * 2,
                        do_binarize=True,
                        do_convert_grayscale=True,
                    )
                except TypeError:
                    # Older/newer diffusers signatures vary slightly.
                    mask_processor = VaeImageProcessor(vae_scale_factor=int(self.vae_scale_factor) * 2)

                mask_tensor = mask_processor.preprocess(pil_mask, height=height, width=width)
                if mask_tensor.shape[1] != 1:
                    mask_tensor = mask_tensor.mean(dim=1, keepdim=True)
                mask_tensor = (mask_tensor > 0.5).float()

            try:
                vae_device = next(vae.parameters()).device
            except Exception:
                vae_device = torch.device("cpu")

            mask_tensor = mask_tensor.to(device=vae_device, dtype=vae.dtype)
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor,
                size=(int(control_latents.shape[-2]), int(control_latents.shape[-1])),
                mode="nearest",
            )

            # Broadcast mask batch if needed.
            if mask_tensor.shape[0] != control_latents.shape[0] and mask_tensor.shape[0] == 1:
                mask_tensor = mask_tensor.expand(control_latents.shape[0], -1, -1, -1)

            mask_5d = mask_tensor.unsqueeze(2)

            control_latents_5d = torch.cat([control_latents_5d, mask_5d, init_latents_5d], dim=1)

        # Return CPU tensor for downstream nodes (matches PromptEncoderNode/DenoiseNode conventions).
        self.values["control_image_latents"] = control_latents_5d.detach().to("cpu")

        # Best-effort size reporting.
        try:
            control_tensor_for_size = _preprocess_image_tensor(control_image)
            self.values["control_image_size"] = (
                int(control_tensor_for_size.shape[-1]),
                int(control_tensor_for_size.shape[-2]),
            )
        except Exception:
            self.values["control_image_size"] = (width, height)
        return self.values
