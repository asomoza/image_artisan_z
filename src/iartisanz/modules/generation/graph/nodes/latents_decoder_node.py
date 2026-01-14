import numpy as np
import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg and "cuda" in msg


def _best_effort_free_vram_after_oom(mm) -> None:
    # Only attempt mitigation if a transformer exists.
    try:
        if mm.has("transformer"):
            mm.offload_to_cpu("transformer")
    except Exception:
        pass

    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            pass


class LatentsDecoderNode(Node):
    REQUIRED_INPUTS = ["vae", "latents"]
    OUTPUTS = ["image"]

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        vae = mm.resolve(self.vae)

        latents = self.latents.to(self.device, vae.dtype)
        latents = (latents / vae.config.scaling_factor) + vae.config.shift_factor

        try:
            decoded = vae.decode(latents, return_dict=False)[0]
        except Exception as e:
            # Reactive mitigation: if decode OOMs, free VRAM and retry once.
            if not _is_cuda_oom(e):
                raise

            _best_effort_free_vram_after_oom(mm)
            decoded = vae.decode(latents, return_dict=False)[0]
        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)

        image = image.detach().cpu().permute(1, 2, 0).float().numpy()
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)

        self.values["image"] = image
        return self.values
