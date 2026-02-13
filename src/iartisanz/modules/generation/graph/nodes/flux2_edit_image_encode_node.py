import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


class Flux2EditImageEncodeNode(Node):
    """Encodes edit images for Flux.2 Klein edit/inpaint models.

    Accepts up to 4 images (from connected ImageLoadNodes), VAE-encodes each,
    patchifies, BN-normalizes, packs to sequence format, and generates position
    IDs with time-offsets so the transformer can distinguish edit images from
    noise latents.

    Output ``image_latents`` and ``image_latent_ids`` are concatenated with the
    noise latents/IDs in :class:`Flux2DenoiseNode` before each transformer step.
    """

    REQUIRED_INPUTS = ["vae", "vae_scale_factor"]
    OPTIONAL_INPUTS = ["image_0", "image_1", "image_2", "image_3"]
    OUTPUTS = ["image_latents", "image_latent_ids"]

    @torch.inference_mode()
    def __call__(self):
        # Collect connected images (skip None optional inputs)
        images = []
        for i in range(4):
            img = getattr(self, f"image_{i}", None)
            if img is not None:
                images.append(img)

        if not images:
            self.values["image_latents"] = None
            self.values["image_latent_ids"] = None
            return self.values

        mm = get_model_manager()
        vae = mm.resolve(self.vae)
        vae_scale_factor = self.vae_scale_factor

        all_latents = []
        all_ids = []

        scale = 10

        for seq_idx, image in enumerate(images):
            # image is numpy HWC uint8 from ImageLoadNode
            img_tensor = torch.from_numpy(image).float() / 255.0
            img_tensor = img_tensor * 2.0 - 1.0
            # (H, W, C) -> (1, C, H, W)
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)

            # Align H, W to vae_scale_factor * 2 (=16 typically) for patchification
            align = vae_scale_factor * 2
            h, w = img_tensor.shape[2], img_tensor.shape[3]
            new_h = (h // align) * align
            new_w = (w // align) * align
            new_h = max(new_h, align)
            new_w = max(new_w, align)
            if new_h != h or new_w != w:
                img_tensor = torch.nn.functional.interpolate(
                    img_tensor, size=(new_h, new_w), mode="bilinear", align_corners=False
                )

            # VAE encode
            img_tensor = img_tensor.to(vae.device, vae.dtype)
            latents = vae.encode(img_tensor).latent_dist.mode()

            # Patchify: (B, C, H, W) -> (B, C*4, H/2, W/2)
            latents = self._patchify_latents(latents)

            # BN normalize: (x - mean) / sqrt(var + eps)
            bn = vae.bn
            bn_mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            bn_var = bn.running_var.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents = (latents - bn_mean) / torch.sqrt(bn_var + vae.config.batch_norm_eps)

            # Generate position IDs with time-offset: T = scale + scale * seq_idx
            t_offset = scale + scale * seq_idx
            latent_ids = self._prepare_image_ids(latents, t_offset)

            # Pack: (B, C, H, W) -> (B, H*W, C)
            latents = self._pack_latents(latents)

            all_latents.append(latents)
            all_ids.append(latent_ids)

        # Concatenate all edit images along the sequence dimension
        image_latents = torch.cat(all_latents, dim=1)
        image_latent_ids = torch.cat(all_ids, dim=1)

        self.values["image_latents"] = image_latents.cpu()
        self.values["image_latent_ids"] = image_latent_ids.cpu()
        return self.values

    @staticmethod
    def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Patchify spatial latents: (B, C, H, W) -> (B, C*4, H/2, W/2)."""
        batch_size, channels, height, width = latents.shape
        latents = latents.reshape(batch_size, channels, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(batch_size, channels * 4, height // 2, width // 2)
        return latents

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack spatial latents to sequence format: (B, C, H, W) -> (B, H*W, C)."""
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents

    @staticmethod
    def _prepare_image_ids(latents: torch.Tensor, t_offset: int) -> torch.Tensor:
        """Generate 4D position IDs (T, H, W, L) for image tokens with a time offset.

        Args:
            latents: Patchified latent tensor of shape (B, C, H, W).
            t_offset: Time coordinate for this edit image (10, 20, 30, 40).

        Returns:
            Position IDs of shape (B, H*W, 4).
        """
        batch_size, _, height, width = latents.shape

        t = torch.tensor([t_offset])
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids
