import numpy as np
import torch

from iartisanz.modules.generation.graph.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


class LatentsDecoderNode(Node):
    REQUIRED_INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        vae = mm.resolve(self.vae)

        latents = self.latents.to(self.device, vae.dtype)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        decoded = vae.decode(latents, return_dict=False)[0]
        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.detach().cpu().permute(1, 2, 0).float().numpy()
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)

        self.values["image"] = image
        return self.values
