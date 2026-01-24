"""Test ControlNet conditioning node with alpha channel detection."""

import numpy as np
import torch
from PIL import Image

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.controlnet_conditioning_node import (
    ControlNetConditioningNode,
    _composite_rgba_to_rgb,
    _extract_alpha_mask,
    _union_masks,
)


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


def test_extract_alpha_mask_pil():
    """Test extracting alpha mask from PIL RGBA image."""
    # Create RGBA image: opaque left half, transparent right half
    img = Image.new('RGBA', (100, 50))
    pixels = np.array(img)
    pixels[:, :50, 3] = 255  # Left half opaque (alpha=255)
    pixels[:, 50:, 3] = 0    # Right half transparent (alpha=0)
    img = Image.fromarray(pixels, mode='RGBA')

    mask = _extract_alpha_mask(img)

    assert mask is not None
    assert mask.mode == 'L'
    mask_array = np.array(mask)
    # Transparent should become white (255), opaque should become black (0)
    assert mask_array[25, 25] == 0, "Opaque region should be black (0)"
    assert mask_array[25, 75] == 255, "Transparent region should be white (255)"


def test_extract_alpha_mask_torch():
    """Test extracting alpha mask from torch RGBA tensor."""
    # Create RGBA tensor: opaque left half, transparent right half
    img = torch.zeros(1, 4, 50, 100)
    img[:, :3, :, :] = 0.5  # Gray RGB
    img[:, 3, :, :50] = 1.0  # Left half opaque (alpha=1.0)
    img[:, 3, :, 50:] = 0.0  # Right half transparent (alpha=0.0)

    mask = _extract_alpha_mask(img)

    assert mask is not None
    assert mask.mode == 'L'
    mask_array = np.array(mask)
    # Transparent should become white (255), opaque should become black (0)
    assert mask_array[25, 25] == 0, "Opaque region should be black (0)"
    assert mask_array[25, 75] == 255, "Transparent region should be white (255)"


def test_extract_alpha_mask_no_alpha():
    """Test that RGB images return None."""
    img = Image.new('RGB', (100, 50), color=(128, 128, 128))
    mask = _extract_alpha_mask(img)
    assert mask is None


def test_composite_rgba_to_rgb_pil():
    """Test compositing RGBA to RGB with PIL."""
    # Create RGBA image with transparency
    img = Image.new('RGBA', (100, 50), color=(255, 0, 0, 128))  # Semi-transparent red

    composited = _composite_rgba_to_rgb(img, background_color=(128, 128, 128))

    assert composited.mode == 'RGB'
    # Should be blend of red and gray
    pixels = np.array(composited)
    assert pixels.shape == (50, 100, 3)


def test_union_masks():
    """Test union of two masks."""
    # Create two masks
    mask1 = Image.new('L', (100, 50), color=0)
    pixels1 = np.array(mask1)
    pixels1[:, :50] = 255  # Left half white
    mask1 = Image.fromarray(pixels1, mode='L')

    mask2 = Image.new('L', (100, 50), color=0)
    pixels2 = np.array(mask2)
    pixels2[:, 50:] = 255  # Right half white
    mask2 = Image.fromarray(pixels2, mode='L')

    union = _union_masks(mask1, mask2)

    assert union is not None
    union_array = np.array(union)
    # Both halves should be white
    assert union_array[25, 25] == 255, "Left should be white (from mask1)"
    assert union_array[25, 75] == 255, "Right should be white (from mask2)"


def test_controlnet_conditioning_with_alpha_control_image():
    """Test that control_image with alpha auto-generates mask."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    # Create RGBA control image: opaque left, transparent right
    control_rgba = torch.zeros(1, 4, 32, 32)
    control_rgba[:, :3, :, :] = 0.5  # Gray RGB
    control_rgba[:, 3, :, :16] = 1.0  # Left opaque
    control_rgba[:, 3, :, 16:] = 0.0  # Right transparent
    node.control_image = control_rgba

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    # Should produce inpainting latents (9 channels: 4 + 1 + 4)
    latents = out["control_image_latents"]
    assert latents.shape[1] == 9, "Should have mask and init latents from alpha"


def test_controlnet_conditioning_with_alpha_init_image():
    """Test that init_image with alpha auto-generates mask."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    # Create RGBA init image: opaque left, transparent right
    init_rgba = torch.zeros(1, 4, 32, 32)
    init_rgba[:, :3, :, :] = 0.5  # Gray RGB
    init_rgba[:, 3, :, :16] = 1.0  # Left opaque
    init_rgba[:, 3, :, 16:] = 0.0  # Right transparent
    node.init_image = init_rgba

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    # Should produce inpainting latents (9 channels: 4 + 1 + 4)
    latents = out["control_image_latents"]
    assert latents.shape[1] == 9, "Should have mask and init latents from alpha"


def test_controlnet_conditioning_alpha_union_with_explicit_mask():
    """Test that alpha mask unions with explicit mask."""
    mm = get_model_manager()
    mm.clear()

    vae = DummyVAE()
    mm.register_active_model(model_id="test", vae=vae)

    node = ControlNetConditioningNode()
    node.vae = ModelHandle("vae")
    node.vae_scale_factor = 8
    node.width = 32
    node.height = 32

    # Create RGBA init image: transparent right half
    init_rgba = torch.zeros(1, 4, 32, 32)
    init_rgba[:, :3, :, :] = 0.5  # Gray RGB
    init_rgba[:, 3, :, :16] = 1.0  # Left opaque
    init_rgba[:, 3, :, 16:] = 0.0  # Right transparent
    node.init_image = init_rgba

    # Create explicit mask: top half masked
    explicit_mask = torch.zeros(1, 1, 32, 32)
    explicit_mask[:, :, :16, :] = 1.0  # Top half masked
    node.mask_image = explicit_mask

    with mm.device_scope(device="cpu", dtype=torch.float32):
        out = node()

    # Should produce inpainting latents with unioned mask
    # Union should have: top half (explicit) + right half (alpha)
    latents = out["control_image_latents"]
    assert latents.shape[1] == 9, "Should have mask and init latents"
