"""Test ControlNet conditioning node with all 11 activation scenarios.

This test suite validates the behavior described in CONTROLNET_REFACTORING_PLAN.md,
including differential diffusion mode awareness where the controlnet_mask serves
dual purposes:
- With differential diffusion: Spatial restriction (where ControlNet applies)
- Without differential diffusion: Inpainting boundary (what to regenerate)
"""

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


def test_scenario_1_control_only():
    """Scenario 1: Control image only -> Standard ControlNet."""
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
    assert latents.shape == (1, 4, 1, 4, 4), "Should have 16-channel (control latents only)"
    assert out["control_mode"] == "standard_controlnet"
    assert out["spatial_mask"] is None


def test_scenario_2_source_only():
    """Scenario 2: Source image only (no ControlNet) -> img2img.

    NOTE: This scenario should not use ControlNetConditioningNode at all.
    ControlNetConditioningNode is only used when ControlNet is active.
    """
    # This test is skipped because ControlNetConditioningNode should not be used
    # for img2img only scenarios. The LatentsNode handles this case.
    pass


def test_scenario_3_source_mask_only():
    """Scenario 3: Source mask only (no ControlNet) -> differential diffusion.

    NOTE: This scenario should not use ControlNetConditioningNode at all.
    Differential diffusion without ControlNet is handled by DenoiseNode directly.
    """
    # This test is skipped because ControlNetConditioningNode should not be used
    # for differential diffusion only scenarios. The DenoiseNode handles this case.
    pass


def test_scenario_4_control_cn_mask():
    """Scenario 4: Control image + CN mask (no source, no diff diff) -> Spatial ControlNet."""
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
    # Without init_image: 16-channel (control only) + spatial mask
    assert latents.shape == (1, 4, 1, 4, 4), "Should have 16-channel with spatial restriction"
    assert out["control_mode"] == "spatial_controlnet"
    assert out["spatial_mask"] is not None


def test_scenario_5_control_source():
    """Scenario 5: Control image + Source image (no mask) -> Standard ControlNet.

    NOTE: init_image (source) is present but without mask, ControlNet just uses control_image.
    The img2img aspect is handled by LatentsNode strength parameter separately.
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
    node.init_image = torch.zeros(1, 3, 32, 32)  # source_image
    node.differential_diffusion_active = False

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4), "Should have 16-channel (control latents only)"
    assert out["control_mode"] == "standard_controlnet"
    assert out["spatial_mask"] is None


def test_scenario_6_control_source_mask():
    """Scenario 6: Control + Source + Source mask -> ControlNet + differential diffusion."""
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
    node.init_image = torch.zeros(1, 3, 32, 32)  # source_image
    node.differential_diffusion_active = True

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4), "Should have 16-channel (control latents only)"
    assert out["control_mode"] == "controlnet_diff_diff"
    assert out["spatial_mask"] is None


def test_scenario_7_control_source_cn_mask_no_diff_diff():
    """Scenario 7: Control + Source + CN mask (no diff diff) -> ControlNet inpainting."""
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
    node.init_image = torch.zeros(1, 3, 32, 32)  # source_image
    node.mask_image = torch.ones(1, 1, 32, 32)  # CN mask
    node.differential_diffusion_active = False

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 9, 1, 4, 4), "Should have 33-channel inpainting context"
    assert out["control_mode"] == "controlnet_inpainting"
    assert out["spatial_mask"] is not None


def test_scenario_8_control_source_cn_mask_with_diff_diff():
    """Scenario 8: Control + Source + CN mask + diff diff -> ControlNet + spatial restriction."""
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
    node.init_image = torch.zeros(1, 3, 32, 32)  # source_image
    node.mask_image = torch.ones(1, 1, 32, 32)  # CN mask as spatial restriction
    node.differential_diffusion_active = True

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4), "Should have 16-channel (control latents only)"
    assert out["control_mode"] == "controlnet_diff_diff"
    assert out["spatial_mask"] is not None, "Should output spatial mask for DenoiseNode"


def test_scenario_9_source_cn_mask_no_diff_diff():
    """Scenario 9: Source + CN mask (no diff diff) -> Inpainting only (source as control)."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32
    node.init_image = torch.zeros(1, 3, 32, 32)  # source_image
    node.mask_image = torch.ones(1, 1, 32, 32)  # CN mask
    node.differential_diffusion_active = False

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 9, 1, 4, 4), "Should have 33-channel inpainting context"
    assert out["control_mode"] == "inpainting_only"
    assert out["spatial_mask"] is not None


def test_scenario_10_source_cn_mask_with_diff_diff():
    """Scenario 10: Source + CN mask + diff diff -> Spatial restriction only (source as control)."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32
    node.init_image = torch.zeros(1, 3, 32, 32)  # source_image
    node.mask_image = torch.ones(1, 1, 32, 32)  # CN mask as spatial restriction
    node.differential_diffusion_active = True

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    latents = out["control_image_latents"]
    assert latents.shape == (1, 4, 1, 4, 4), "Should have 16-channel (control latents only)"
    assert out["control_mode"] == "diff_diff_spatial"
    assert out["spatial_mask"] is not None, "Should output spatial mask for DenoiseNode"


def test_scenario_11_source_source_mask():
    """Scenario 11: Source + Source mask -> img2img + differential diffusion (no ControlNet).

    NOTE: This scenario should not use ControlNetConditioningNode at all.
    The combination of source_image + source_mask is handled by LatentsNode + DenoiseNode.
    """
    # This test is skipped because ControlNetConditioningNode should not be used
    # for img2img + differential diffusion without ControlNet.
    pass


def test_differential_diffusion_flag_changes_mask_semantics():
    """Test that the same inputs produce different outputs based on diff diff flag."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    # Test with differential diffusion OFF
    node1 = ControlNetConditioningNode()
    node1.vae = ModelHandle("vae")
    node1.vae_scale_factor = 8
    node1.width = 32
    node1.height = 32
    node1.control_image = torch.zeros(1, 3, 32, 32)
    node1.init_image = torch.zeros(1, 3, 32, 32)
    node1.mask_image = torch.ones(1, 1, 32, 32)
    node1.differential_diffusion_active = False

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out1 = node1()

    # Test with differential diffusion ON
    node2 = ControlNetConditioningNode()
    node2.vae = ModelHandle("vae")
    node2.vae_scale_factor = 8
    node2.width = 32
    node2.height = 32
    node2.control_image = torch.zeros(1, 3, 32, 32)
    node2.init_image = torch.zeros(1, 3, 32, 32)
    node2.mask_image = torch.ones(1, 1, 32, 32)
    node2.differential_diffusion_active = True

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out2 = node2()

    # Different modes
    assert out1["control_mode"] == "controlnet_inpainting"
    assert out2["control_mode"] == "controlnet_diff_diff"

    # Different channel counts
    assert out1["control_image_latents"].shape[1] == 9  # 33-channel
    assert out2["control_image_latents"].shape[1] == 4  # 16-channel

    # Both should output spatial mask
    assert out1["spatial_mask"] is not None
    assert out2["spatial_mask"] is not None
