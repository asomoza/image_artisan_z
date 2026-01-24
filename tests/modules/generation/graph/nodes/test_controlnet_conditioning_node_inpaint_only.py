"""Test ControlNet conditioning node with inpainting-only mode (no control_image)."""

import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.controlnet_conditioning_node import ControlNetConditioningNode


class DummyVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32

        class _Cfg:
            shift_factor = 0.0
            scaling_factor = 1.0

        self.config = _Cfg()

    def encode(self, x: torch.Tensor):
        b, _c, h, w = x.shape
        lh = max(1, h // 8)
        lw = max(1, w // 8)
        latents = torch.zeros((b, 4, lh, lw), device=x.device, dtype=x.dtype)

        class _Out:
            pass

        out = _Out()
        out.latents = latents
        return out


def test_controlnet_conditioning_node_inpaint_only_with_init_image_no_control_image():
    """Test inpainting-only mode: init_image provided, no control_image."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    # Only init_image, no control_image
    node.init_image = torch.zeros(1, 3, 32, 32)

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    # Should produce control latents using init_image as base
    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4)


def test_controlnet_conditioning_node_inpaint_only_with_init_and_mask():
    """Test inpainting-only mode: init_image and mask provided, no control_image."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    # Only init_image and mask, no control_image
    node.init_image = torch.zeros(1, 3, 32, 32)
    node.mask_image = torch.ones(1, 1, 32, 32)

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    # Should produce concatenated latents [init_image_latents, mask, init_image_latents]
    # = 4 + 1 + 4 = 9 channels
    latents = out["control_image_latents"]
    assert latents.shape == (1, 9, 1, 4, 4)
