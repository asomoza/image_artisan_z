import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class Flux2LatentsNode(Node):
    """Creates latents for Flux.2 text-to-image or img2img generation.

    When an optional source image is provided, encodes it via VAE and produces
    patchified latents in packed sequence format plus matching noise for img2img.
    Otherwise produces pure random noise for text-to-image.
    """

    REQUIRED_INPUTS = ["vae_scale_factor", "width", "height", "num_channels_latents", "seed"]
    OPTIONAL_INPUTS = ["image", "vae"]
    OUTPUTS = ["latents", "latent_ids", "noise", "source_image", "has_source_image"]
    SERIALIZE_EXCLUDE = {"latents"}

    def __init__(self, latents=None):
        super().__init__()
        self.latents = latents

    def __call__(self):
        mm = get_model_manager()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # Latent spatial dimensions (divisible by vae_scale_factor * 2 for patchify)
        height = 2 * (int(self.height) // (self.vae_scale_factor * 2))
        width = 2 * (int(self.width) // (self.vae_scale_factor * 2))

        image = getattr(self, "image", None)
        noise = None

        if image is not None:
            try:
                vae_input = getattr(self, "vae", None)
                if vae_input is None:
                    vae_input = ModelHandle("vae")
                vae = mm.resolve(vae_input)

                if isinstance(image, np.ndarray):
                    np_image = image
                else:
                    np_image = np.array(image)

                if np_image.ndim == 2:
                    np_image = np.stack([np_image, np_image, np_image], axis=-1)

                if np_image.dtype == np.uint8:
                    np_image = np_image.astype(np.float32) / 255.0
                else:
                    np_image = np_image.astype(np.float32)

                # (H, W, C) -> (1, C, H, W), normalize to [-1, 1]
                img_tensor = torch.from_numpy(np_image).permute(2, 0, 1).unsqueeze(0)
                img_tensor = img_tensor * 2.0 - 1.0

                # Resize to output resolution aligned to vae_scale_factor * 2
                align = self.vae_scale_factor * 2
                pixel_h = height * self.vae_scale_factor
                pixel_w = width * self.vae_scale_factor

                img_tensor = torch.nn.functional.interpolate(
                    img_tensor, size=(pixel_h, pixel_w), mode="bilinear", align_corners=False
                )

                # VAE encode
                img_tensor = img_tensor.to(vae.device, vae.dtype)
                latents = vae.encode(img_tensor).latent_dist.sample(generator)

                # Patchify: (B, C, H, W) -> (B, C*4, H/2, W/2)
                latents = self._patchify_latents(latents)

                # Generate position IDs from patchified shape
                latent_ids = self._prepare_latent_ids(latents)
                latent_ids = latent_ids.to(self.device)

                # Pack: (B, C, H, W) -> (B, H*W, C) — NO BN normalization
                latents = self._pack_latents(latents)

                # Generate noise in the same packed shape
                patchified_shape = (1, self.num_channels_latents * 4, height // 2, width // 2)
                noise = randn_tensor(patchified_shape, generator=generator, device=self.device, dtype=self.dtype)
                noise = self._pack_latents(noise)
            except Exception as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)
        else:
            # Pure random noise for text-to-image
            patchified_shape = (1, self.num_channels_latents * 4, height // 2, width // 2)
            latents = randn_tensor(patchified_shape, generator=generator, device=self.device, dtype=self.dtype)

            latent_ids = self._prepare_latent_ids(latents)
            latent_ids = latent_ids.to(self.device)

            latents = self._pack_latents(latents)

        self.values["latents"] = latents
        self.values["latent_ids"] = latent_ids
        self.values["noise"] = noise
        self.values["source_image"] = image if image is not None else None
        self.values["has_source_image"] = image is not None

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
    def _prepare_latent_ids(latents: torch.Tensor) -> torch.Tensor:
        """Generate 4D position coordinates (T, H, W, L) for latent tokens.

        Args:
            latents: Patchified latent tensor of shape (B, C, H, W).

        Returns:
            Position IDs of shape (B, H*W, 4).
        """
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)

        latent_ids = torch.cartesian_prod(t, h, w, l)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        """Pack spatial latents to sequence format.

        (B, C, H, W) -> (B, H*W, C)
        """
        batch_size, num_channels, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
        return latents
