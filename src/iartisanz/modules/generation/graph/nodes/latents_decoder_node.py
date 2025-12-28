import numpy as np
import torch

from iartisanz.modules.generation.graph.nodes.node import Node


class LatentsDecoderNode(Node):
    REQUIRED_INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    @torch.inference_mode()
    def __call__(self):
        latents = self.latents.to(self.device, self.vae.dtype)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

        decoded = self.vae.decode(latents, return_dict=False)[0]
        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.detach().cpu().permute(1, 2, 0).float().numpy()
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)

        self.values["image"] = image
        return self.values
