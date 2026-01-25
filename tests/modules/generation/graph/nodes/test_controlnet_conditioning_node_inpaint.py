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


def test_controlnet_conditioning_node_no_mask_outputs_control_latents_only():
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    node.control_image = torch.zeros(1, 3, 32, 32)
    node.differential_diffusion_active = False

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4)
    assert out["control_mode"] == "standard_controlnet"


def test_controlnet_conditioning_node_mask_enables_inpaint_concat_with_explicit_init_image():
    """With control + mask + init_image (source) and NO differential diffusion -> 33-channel inpainting."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    node.control_image = torch.zeros(1, 3, 32, 32)
    node.mask_image = torch.ones(1, 1, 32, 32)
    node.init_image = torch.zeros(1, 3, 32, 32)  # source_image from LatentsNode
    node.differential_diffusion_active = False

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 9, 1, 4, 4), "Should use 33-channel inpainting context"
    assert out["control_mode"] == "controlnet_inpainting"
    assert out["spatial_mask"] is not None


def test_controlnet_conditioning_node_mask_without_init_is_spatial_mode():
    """With control + mask but NO init_image (source) -> Spatial ControlNet mode (not inpainting).

    This is the new behavior after refactoring. Inpainting mode requires init_image.
    Without init_image, the mask is used for spatial restriction only.
    """
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    node.control_image = torch.zeros(1, 3, 32, 32)
    node.mask_image = torch.ones(1, 1, 32, 32)
    node.differential_diffusion_active = False

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4), "Should use 16-channel with spatial restriction"
    assert out["control_mode"] == "spatial_controlnet"
    assert out["spatial_mask"] is not None
