import numpy as np
import torch
from PIL import Image

from iartisanz.modules.generation.graph.nodes.node import Node


class LatentsDecoderNode(Node):
    REQUIRED_INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    def __init__(self):
        super().__init__()

    @torch.inference_mode()
    def __call__(self):
        latents = self.latents.to(self.device, self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        decoded = self.vae.decode(latents, return_dict=False)[0]
        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(1, 2, 0).float().numpy()

        self.values["image"] = Image.fromarray(np.uint8(image * 255))

        return self.values
