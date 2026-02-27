import numpy as np
import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class ZImageLatentsDecoderNode(Node):
    REQUIRED_INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    @torch.no_grad()
    def __call__(self):
        mm = get_model_manager()

        with mm.use_components("vae", device=self.device):
            vae = mm.resolve(self.vae)

            latents = self.latents.to(self.device, vae.dtype)
            latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

            try:
                decoded = vae.decode(latents, return_dict=False)[0]
            except Exception as e:
                # Reactive mitigation: if decode OOMs, free VRAM and retry once.
                if not mm.is_cuda_oom(e):
                    raise

                mm.free_vram_for_forward_pass(preserve=("vae",))
                try:
                    decoded = vae.decode(latents, return_dict=False)[0]
                except Exception as retry_exc:
                    if mm.is_cuda_oom(retry_exc):
                        raise IArtisanZNodeError(
                            "CUDA out of memory during VAE decode even after offloading.",
                            "ZImageLatentsDecoderNode",
                        ) from retry_exc
                    raise
            image = decoded[0]
            image = (image / 2 + 0.5).clamp(0, 1)

            image = image.detach().cpu().permute(1, 2, 0).float().numpy()
            image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
            image = np.ascontiguousarray(image)

        self.values["image"] = image
        return self.values
