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
    OUTPUTS = ["control_image_latents", "control_image_size"]

    SERIALIZE_EXCLUDE = {"control_image_latents"}

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        vae = mm.resolve(self.vae)

        width = int(self.width)
        height = int(self.height)

        control_image = self.control_image
        if isinstance(control_image, torch.Tensor):
            image_tensor = control_image
        else:
            if isinstance(control_image, np.ndarray):
                np_img = control_image
            else:
                np_img = np.array(control_image)

            if np_img.ndim == 2:
                np_img = np.stack([np_img, np_img, np_img], axis=-1)

            if np_img.dtype != np.uint8:
                np_img = np.clip(np_img, 0, 255).astype(np.uint8)

            pil = Image.fromarray(np_img)
            processor = VaeImageProcessor(vae_scale_factor=int(self.vae_scale_factor) * 2)
            image_tensor = processor.preprocess(pil, height=height, width=width)

        # Encode on the VAE device (ModelManager scope decides CPU/GPU).
        try:
            vae_device = next(vae.parameters()).device
        except Exception:
            vae_device = torch.device("cpu")
        image_tensor = image_tensor.to(device=vae_device, dtype=vae.dtype)

        try:
            control_latents = _retrieve_latents(vae.encode(image_tensor), sample_mode="argmax")
        except Exception as e:
            raise IArtisanZNodeError(f"Failed encoding control image: {e}", self.__class__.__name__) from e

        control_latents = (control_latents - vae.config.shift_factor) * vae.config.scaling_factor
        control_latents = control_latents.unsqueeze(2)

        # Return CPU tensor for downstream nodes (matches PromptEncoderNode/DenoiseNode conventions).
        self.values["control_image_latents"] = control_latents.detach().to("cpu")
        self.values["control_image_size"] = (int(image_tensor.shape[-1]), int(image_tensor.shape[-2]))
        return self.values
