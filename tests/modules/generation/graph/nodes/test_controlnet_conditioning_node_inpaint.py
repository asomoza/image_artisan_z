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

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4)


def test_controlnet_conditioning_node_mask_enables_inpaint_concat_with_explicit_init_image():
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
    node.init_image = torch.zeros(1, 3, 32, 32)

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 9, 1, 4, 4)


def test_controlnet_conditioning_node_mask_defaults_init_image_to_control_image():
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

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 9, 1, 4, 4)
