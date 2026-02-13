import numpy as np
import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


class Flux2LatentsDecoderNode(Node):
    """Decodes Flux.2 Klein packed latents back to pixel images.

    Steps:
    1. Unpack sequence tokens to spatial layout using position IDs
    2. BatchNorm denormalization using VAE running statistics
    3. Unpatchify (reverse 2x2 patchification)
    4. VAE decode
    """

    REQUIRED_INPUTS = ["vae", "latents", "latent_ids"]
    OUTPUTS = ["image"]

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        vae = mm.resolve(self.vae)

        latents = self.latents.to(self.device, vae.dtype)
        latent_ids = self.latent_ids.to(self.device)

        # Step 1: Unpack sequence -> spatial using position IDs
        latents = self._unpack_latents_with_ids(latents, latent_ids)

        # Step 2: BatchNorm denormalization
        bn = vae.bn
        bn_mean = bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        bn_std = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(
            latents.device, latents.dtype
        )
        latents = latents * bn_std + bn_mean

        # Step 3: Unpatchify (B, C*4, H/2, W/2) -> (B, C, H, W)
        latents = self._unpatchify_latents(latents)

        # Step 4: VAE decode
        try:
            decoded = vae.decode(latents, return_dict=False)[0]
        except Exception as e:
            if not mm.is_cuda_oom(e):
                raise
            mm.free_vram_for_forward_pass(preserve=("vae",))
            decoded = vae.decode(latents, return_dict=False)[0]

        image = decoded[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(1, 2, 0).float().numpy()
        image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
        image = np.ascontiguousarray(image)

        self.values["image"] = image
        return self.values

    @staticmethod
    def _unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> torch.Tensor:
        """Scatter packed sequence tokens back to spatial layout using position IDs.

        Args:
            x: Packed latents of shape (B, seq_len, C).
            x_ids: Position IDs of shape (B, seq_len, 4) where dims are (T, H, W, L).

        Returns:
            Spatial latents of shape (B, C, H, W).
        """
        x_list = []
        for data, pos in zip(x, x_ids):
            _, ch = data.shape
            h_ids = pos[:, 1].to(torch.int64)
            w_ids = pos[:, 2].to(torch.int64)

            h = torch.max(h_ids) + 1
            w = torch.max(w_ids) + 1

            flat_ids = h_ids * w + w_ids

            out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
            out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

            out = out.view(h, w, ch).permute(2, 0, 1)
            x_list.append(out)

        return torch.stack(x_list, dim=0)

    @staticmethod
    def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
        """Reverse 2x2 patchification.

        (B, C*4, H/2, W/2) -> (B, C, H, W)
        """
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
        latents = latents.permute(0, 1, 4, 2, 5, 3)
        latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
        return latents
