import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanz.modules.generation.graph.nodes.node import Node


class Flux2LatentsNode(Node):
    """Creates random latents for Flux.2 Klein text-to-image generation.

    Produces patchified noise in packed sequence format plus position IDs.
    """

    REQUIRED_INPUTS = ["vae_scale_factor", "width", "height", "num_channels_latents", "seed"]
    OUTPUTS = ["latents", "latent_ids"]
    SERIALIZE_EXCLUDE = {"latents"}

    def __init__(self, latents=None):
        super().__init__()
        self.latents = latents

    def __call__(self):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        # Latent spatial dimensions (divisible by vae_scale_factor * 2 for patchify)
        height = 2 * (int(self.height) // (self.vae_scale_factor * 2))
        width = 2 * (int(self.width) // (self.vae_scale_factor * 2))

        # Patchified shape: channels * 4, half spatial dims
        shape = (1, self.num_channels_latents * 4, height // 2, width // 2)
        latents = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

        # Generate position IDs from patchified spatial dims
        latent_ids = self._prepare_latent_ids(latents)
        latent_ids = latent_ids.to(self.device)

        # Pack: (B, C, H, W) -> (B, H*W, C)
        latents = self._pack_latents(latents)

        self.values["latents"] = latents
        self.values["latent_ids"] = latent_ids
        return self.values

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
