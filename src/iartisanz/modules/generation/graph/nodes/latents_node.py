import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanz.modules.generation.graph.nodes.node import Node


class LatentsNode(Node):
    REQUIRED_INPUTS = [
        "vae_scale_factor",
        "width",
        "height",
        "num_channels_latents",
        "seed",
    ]
    OUTPUTS = ["latents"]
    SERIALIZE_EXCLUDE = {"latents"}

    def __init__(self, latents=None):
        super().__init__()
        self.latents = latents

    def __call__(self):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        height = 2 * (int(self.height) // (self.vae_scale_factor * 2))
        width = 2 * (int(self.width) // (self.vae_scale_factor * 2))

        shape = (1, self.num_channels_latents, height, width)

        self.values["latents"] = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

        return self.values
