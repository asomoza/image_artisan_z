import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


class Flux2InpaintBaseLatentsNode(Node):
    """Encode the first edit image at output resolution for differential diffusion blending.

    Produces packed latents in the same format as ``Flux2LatentsNode`` (no BN
    normalization) so they can be directly blended with the noise latents during
    the denoise loop.
    """

    REQUIRED_INPUTS = ["vae", "vae_scale_factor", "image", "width", "height"]
    OUTPUTS = ["base_latents"]

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        vae = mm.resolve(self.vae)
        vae_scale_factor = self.vae_scale_factor

        # image is numpy HWC uint8 from ImageLoadNode
        img_tensor = torch.from_numpy(self.image).float() / 255.0
        img_tensor = img_tensor * 2.0 - 1.0
        # (H, W, C) -> (1, C, H, W)
        img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

        # Resize to output resolution aligned to vae_scale_factor * 2
        align = vae_scale_factor * 2
        target_h = 2 * (int(self.height) // align)
        target_w = 2 * (int(self.width) // align)
        target_h = max(target_h, 2)
        target_w = max(target_w, 2)
        # Convert from patchified dims back to pixel-space for VAE input
        pixel_h = target_h * vae_scale_factor
        pixel_w = target_w * vae_scale_factor

        img_tensor = torch.nn.functional.interpolate(
            img_tensor, size=(pixel_h, pixel_w), mode="bilinear", align_corners=False
        )

        # VAE encode
        img_tensor = img_tensor.to(vae.device, vae.dtype)
        latents = vae.encode(img_tensor).latent_dist.mode()

        # Patchify: (B, C, H, W) -> (B, C*4, H/2, W/2)
        latents = self._patchify_latents(latents)

        # Pack: (B, C, H, W) -> (B, H*W, C) — NO BN normalization
        latents = self._pack_latents(latents)

        self.values["base_latents"] = latents.cpu()
        return self.values

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents
