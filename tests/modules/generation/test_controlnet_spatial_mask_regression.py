"""Regression tests for ControlNet spatial mask loading and application.

This test suite ensures that spatial masks with alpha channel are correctly:
1. Loaded from files (ImageLoadNode with grayscale=True extracts alpha)
2. Processed through ControlNetConditioningNode
3. Applied to ControlNet blocks in DenoiseNode

These tests prevent regressions where masks become all-zero due to incorrect
handling of black-over-alpha convention used in the UI.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest
import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.controlnet_conditioning_node import ControlNetConditioningNode
from iartisanz.modules.generation.graph.nodes.denoise_node import DenoiseNode
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode


class DummyVAE(torch.nn.Module):
    """Minimal VAE for testing."""

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


class DummyControlNet(torch.nn.Module):
    """Minimal ControlNet that returns predictable block samples for testing."""

    def __init__(self):
        super().__init__()
        self._p = torch.nn.Parameter(torch.zeros(1))
        self.dtype = torch.float32

    def forward(self, sample, timestep, encoder_hidden_states, controlnet_cond, **kwargs):
        """Return predictable block samples for testing.

        Returns blocks with values of 1.0 so we can verify masking sets them to 0.0.
        """
        # Create dummy blocks at different resolutions
        b, c, h, w = sample.shape

        # Typical ControlNet returns ~15 blocks at different spatial resolutions
        blocks = []
        for i in range(15):
            # Alternate between 3D and 4D blocks to test dimension matching
            if i % 3 == 0:
                # 3D block [B, H, W]
                block_h, block_w = h * (2 ** (i % 3)), w * (2 ** (i % 3))
                block = torch.ones(b, block_h, block_w, device=sample.device, dtype=sample.dtype)
            else:
                # 4D block [B, C, H, W]
                block_h, block_w = h * (2 ** (i % 3)), w * (2 ** (i % 3))
                block = torch.ones(b, c, block_h, block_w, device=sample.device, dtype=sample.dtype)
            blocks.append(block)

        # Return format matching real ControlNet
        return (torch.zeros_like(sample), blocks)


def create_test_mask_image_with_alpha(path: str, width: int, height: int, painted_region: str = "left"):
    """Create a test mask image using black-over-alpha convention (UI format).

    Args:
        path: Path to save the mask PNG
        width: Image width
        height: Image height
        painted_region: Which region to paint - "left", "right", "top", "bottom", "all", or "none"

    The mask uses the UI convention:
    - Painted areas: black RGB (0,0,0) with alpha=255 (opaque)
    - Unpainted areas: any RGB with alpha=0 (transparent)

    When loaded with grayscale=True, ImageLoadNode should extract alpha channel:
    - Painted areas (alpha=255) -> normalized to 1.0 (white/apply ControlNet)
    - Unpainted areas (alpha=0) -> normalized to 0.0 (black/no ControlNet)
    """
    # Create BGRA image (OpenCV format)
    img = np.zeros((height, width, 4), dtype=np.uint8)

    # Set painted region based on parameter
    if painted_region == "left":
        img[:, :width // 2, 0:3] = 0  # Black RGB
        img[:, :width // 2, 3] = 255  # Opaque alpha
    elif painted_region == "right":
        img[:, width // 2:, 0:3] = 0  # Black RGB
        img[:, width // 2:, 3] = 255  # Opaque alpha
    elif painted_region == "top":
        img[:height // 2, :, 0:3] = 0  # Black RGB
        img[:height // 2, :, 3] = 255  # Opaque alpha
    elif painted_region == "bottom":
        img[height // 2:, :, 0:3] = 0  # Black RGB
        img[height // 2:, :, 3] = 255  # Opaque alpha
    elif painted_region == "all":
        img[:, :, 0:3] = 0  # Black RGB
        img[:, :, 3] = 255  # Opaque alpha
    elif painted_region == "none":
        # All transparent
        img[:, :, 3] = 0

    cv2.imwrite(path, img)


class TestImageLoadNodeMaskWithAlpha:
    """Test that ImageLoadNode correctly extracts alpha channel for masks."""

    def test_load_mask_with_alpha_extracts_alpha_channel(self):
        """ImageLoadNode with grayscale=True should extract alpha from RGBA images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "test_mask.png")
            create_test_mask_image_with_alpha(mask_path, 64, 64, painted_region="left")

            # Load mask with grayscale=True
            node = ImageLoadNode(path=mask_path, grayscale=True)
            node()

            mask = node.values["image"]

            # Verify shape [H, W, 1]
            assert mask.shape == (64, 64, 1), f"Expected shape (64, 64, 1), got {mask.shape}"

            # Verify values: painted region (alpha=255) -> 1.0, unpainted (alpha=0) -> 0.0
            assert mask[:, :32, 0].min() > 0.99, "Left half should be ~1.0 (painted region)"
            assert mask[:, 32:, 0].max() < 0.01, "Right half should be ~0.0 (unpainted region)"

    def test_load_mask_fully_painted_is_all_ones(self):
        """Fully painted mask should load as all 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "test_mask_full.png")
            create_test_mask_image_with_alpha(mask_path, 64, 64, painted_region="all")

            node = ImageLoadNode(path=mask_path, grayscale=True)
            node()

            mask = node.values["image"]
            assert mask.min() > 0.99, "Fully painted mask should be all ~1.0"
            assert mask.max() > 0.99, "Fully painted mask should be all ~1.0"

    def test_load_mask_unpainted_is_all_zeros(self):
        """Unpainted mask should load as all 0.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "test_mask_empty.png")
            create_test_mask_image_with_alpha(mask_path, 64, 64, painted_region="none")

            node = ImageLoadNode(path=mask_path, grayscale=True)
            node()

            mask = node.values["image"]
            assert mask.max() < 0.01, "Unpainted mask should be all ~0.0"
            assert mask.min() < 0.01, "Unpainted mask should be all ~0.0"

    def test_load_rgb_mask_without_alpha_uses_grayscale(self):
        """RGB mask without alpha should use standard grayscale conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "test_mask_rgb.png")
            # Create RGB image (no alpha): white left, black right
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :32, :] = 255  # White left
            cv2.imwrite(mask_path, img)

            node = ImageLoadNode(path=mask_path, grayscale=True)
            node()

            mask = node.values["image"]

            # White -> 1.0, Black -> 0.0
            assert mask[:, :32, 0].min() > 0.99, "White region should be ~1.0"
            assert mask[:, 32:, 0].max() < 0.01, "Black region should be ~0.0"


class TestControlNetSpatialMaskPipeline:
    """Test spatial mask through ControlNetConditioningNode."""

    def test_spatial_mask_passed_to_output(self):
        """Spatial mask should be passed through to DenoiseNode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "test_mask.png")
            create_test_mask_image_with_alpha(mask_path, 32, 32, painted_region="left")

            # Load mask
            mask_node = ImageLoadNode(path=mask_path, grayscale=True)
            mask_node()
            mask_numpy = mask_node.values["image"]

            # Setup ControlNet node
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
            node.mask_image = mask_numpy
            node.differential_diffusion_active = False

            with mm.device_scope(device="cpu", dtype=torch.float32):
                out = node()

            # Verify spatial mask is output
            assert out["spatial_mask"] is not None, "Should output spatial mask"
            assert out["control_mode"] == "spatial_controlnet"

            spatial_mask = out["spatial_mask"]
            assert spatial_mask.shape == (1, 1, 32, 32), f"Expected (1, 1, 32, 32), got {spatial_mask.shape}"

            # Verify mask values (left=1.0, right=0.0)
            left_mean = spatial_mask[:, :, :, :16].mean().item()
            right_mean = spatial_mask[:, :, :, 16:].mean().item()

            assert left_mean > 0.99, f"Left half should be ~1.0 (painted), got {left_mean}"
            assert right_mean < 0.01, f"Right half should be ~0.0 (unpainted), got {right_mean}"

    def test_spatial_mask_statistics(self):
        """Test mask statistics for different painted regions."""
        test_cases = [
            ("left", 0.5),  # Half painted
            ("all", 1.0),  # Fully painted
            ("none", 0.0),  # Not painted
        ]

        for painted_region, expected_mean in test_cases:
            with tempfile.TemporaryDirectory() as tmpdir:
                mask_path = os.path.join(tmpdir, f"test_mask_{painted_region}.png")
                create_test_mask_image_with_alpha(mask_path, 32, 32, painted_region=painted_region)

                # Load and process mask
                mask_node = ImageLoadNode(path=mask_path, grayscale=True)
                mask_node()
                mask_numpy = mask_node.values["image"]

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
                node.mask_image = mask_numpy
                node.differential_diffusion_active = False

                with mm.device_scope(device="cpu", dtype=torch.float32):
                    out = node()

                spatial_mask = out["spatial_mask"]
                actual_mean = spatial_mask.mean().item()

                assert abs(actual_mean - expected_mean) < 0.1, (
                    f"Region '{painted_region}': expected mean ~{expected_mean}, got {actual_mean}"
                )


class TestDenoiseNodeSpatialMaskApplication:
    """Test that DenoiseNode correctly applies spatial masks to ControlNet blocks."""

    def test_spatial_mask_zeros_out_masked_regions(self):
        """Spatial mask should zero out ControlNet blocks in unpainted regions.

        This tests the core spatial restriction behavior:
        - Painted regions (mask=1.0): Keep ControlNet blocks (multiply by 1.0)
        - Unpainted regions (mask=0.0): Zero out ControlNet blocks (multiply by 0.0)
        """
        # Create spatial mask: left painted (1.0), right unpainted (0.0)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, :, :8] = 1.0  # Left half

        # Create dummy ControlNet blocks with all 1.0 values as dict
        # After masking, left should stay 1.0, right should become 0.0
        block_samples = {
            0: torch.ones(1, 16, 16),  # 3D block
            1: torch.ones(1, 4, 16, 16),  # 4D block
        }

        # Apply spatial mask using DenoiseNode._apply_spatial_mask
        latent_shape = (1, 4, 16, 16)
        masked_blocks = DenoiseNode._apply_spatial_mask(block_samples, mask, latent_shape)

        # Verify left half kept values, right half zeroed
        for i, block in masked_blocks.items():
            if block.ndim == 3:
                left_mean = block[:, :, :8].mean().item()
                right_mean = block[:, :, 8:].mean().item()
            else:
                left_mean = block[:, :, :, :8].mean().item()
                right_mean = block[:, :, :, 8:].mean().item()

            assert left_mean > 0.99, f"Block {i}: Left (painted) should be ~1.0, got {left_mean}"
            assert right_mean < 0.01, f"Block {i}: Right (unpainted) should be ~0.0, got {right_mean}"

    def test_spatial_mask_handles_different_block_resolutions(self):
        """Spatial mask should correctly resize per-block for different resolutions."""
        # Create spatial mask at latent resolution (16x16)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, :, :8] = 1.0  # Left half

        # Create blocks at different resolutions (simulating different UNet layers)
        block_samples = {
            0: torch.ones(1, 8, 8),  # Smaller resolution 3D
            1: torch.ones(1, 16, 16),  # Same resolution 3D
            2: torch.ones(1, 32, 32),  # Larger resolution 3D
            3: torch.ones(1, 4, 8, 8),  # Smaller resolution 4D
            4: torch.ones(1, 4, 16, 16),  # Same resolution 4D
            5: torch.ones(1, 4, 32, 32),  # Larger resolution 4D
        }

        latent_shape = (1, 4, 16, 16)
        masked_blocks = DenoiseNode._apply_spatial_mask(block_samples, mask, latent_shape)

        # All blocks should be masked correctly despite different resolutions
        for i, block in masked_blocks.items():
            if block.ndim == 3:
                _, _, w = block.shape
                half_w = w // 2
                left_mean = block[:, :, :half_w].mean().item()
                right_mean = block[:, :, half_w:].mean().item()
            else:
                _, _, _, w = block.shape
                half_w = w // 2
                left_mean = block[:, :, :, :half_w].mean().item()
                right_mean = block[:, :, :, half_w:].mean().item()

            # Allow some tolerance for interpolation artifacts when resizing
            assert left_mean > 0.98, f"Block {i}: Left should be ~1.0, got {left_mean}"
            assert right_mean < 0.02, f"Block {i}: Right should be ~0.0, got {right_mean}"

    def test_spatial_mask_all_zeros_blocks_all_controlnet(self):
        """All-zero mask should block all ControlNet guidance."""
        mask = torch.zeros(1, 1, 16, 16)  # All unpainted
        block_samples = {
            0: torch.ones(1, 16, 16),
            1: torch.ones(1, 4, 16, 16),
        }

        latent_shape = (1, 4, 16, 16)
        masked_blocks = DenoiseNode._apply_spatial_mask(block_samples, mask, latent_shape)

        for block in masked_blocks.values():
            assert block.max().item() < 0.01, "All blocks should be zeroed with all-zero mask"

    def test_spatial_mask_all_ones_passes_all_controlnet(self):
        """All-one mask should pass all ControlNet guidance."""
        mask = torch.ones(1, 1, 16, 16)  # All painted
        block_samples = {
            0: torch.ones(1, 16, 16),
            1: torch.ones(1, 4, 16, 16),
        }

        latent_shape = (1, 4, 16, 16)
        masked_blocks = DenoiseNode._apply_spatial_mask(block_samples, mask, latent_shape)

        for block in masked_blocks.values():
            assert block.min().item() > 0.99, "All blocks should keep values with all-one mask"


class TestSpatialMaskEndToEnd:
    """End-to-end tests from file loading through mask application."""

    def test_end_to_end_mask_loading_and_application(self):
        """Test complete pipeline: file -> ImageLoadNode -> ControlNetConditioningNode -> DenoiseNode."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mask_path = os.path.join(tmpdir, "test_mask.png")
            # Create mask with top half painted
            create_test_mask_image_with_alpha(mask_path, 32, 32, painted_region="top")

            # 1. Load mask from file
            mask_node = ImageLoadNode(path=mask_path, grayscale=True)
            mask_node()
            mask_numpy = mask_node.values["image"]

            # Verify loaded mask is correct
            top_mean = mask_numpy[:16, :, 0].mean()
            bottom_mean = mask_numpy[16:, :, 0].mean()
            assert top_mean > 0.99, f"Top should be ~1.0, got {top_mean}"
            assert bottom_mean < 0.01, f"Bottom should be ~0.0, got {bottom_mean}"

            # 2. Process through ControlNetConditioningNode
            mm = get_model_manager()
            mm.clear()
            vae = DummyVAE()
            mm.register_active_model(model_id="test", vae=vae)

            cnet_node = ControlNetConditioningNode()
            cnet_node.vae = ModelHandle("vae")
            cnet_node.vae_scale_factor = 8
            cnet_node.width = 32
            cnet_node.height = 32
            cnet_node.control_image = torch.zeros(1, 3, 32, 32)
            cnet_node.mask_image = mask_numpy
            cnet_node.differential_diffusion_active = False

            with mm.device_scope(device="cpu", dtype=torch.float32):
                cnet_out = cnet_node()

            spatial_mask = cnet_out["spatial_mask"]
            assert spatial_mask is not None

            # Verify mask orientation
            top_mean = spatial_mask[:, :, :16, :].mean().item()
            bottom_mean = spatial_mask[:, :, 16:, :].mean().item()
            assert top_mean > 0.99, f"Top should be ~1.0, got {top_mean}"
            assert bottom_mean < 0.01, f"Bottom should be ~0.0, got {bottom_mean}"

            # 3. Apply to ControlNet blocks
            block_samples = {
                0: torch.ones(1, 32, 32),  # 3D block
                1: torch.ones(1, 4, 32, 32),  # 4D block
            }

            latent_shape = (1, 4, 32, 32)
            masked_blocks = DenoiseNode._apply_spatial_mask(block_samples, spatial_mask, latent_shape)

            # Verify blocks are correctly masked
            for block in masked_blocks.values():
                if block.ndim == 3:
                    top_mean = block[:, :16, :].mean().item()
                    bottom_mean = block[:, 16:, :].mean().item()
                else:
                    top_mean = block[:, :, :16, :].mean().item()
                    bottom_mean = block[:, :, 16:, :].mean().item()

                assert top_mean > 0.99, f"Top (painted) should be ~1.0, got {top_mean}"
                assert bottom_mean < 0.01, f"Bottom (unpainted) should be ~0.0, got {bottom_mean}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
