"""Test spatial mask reshaping for different layer resolutions."""

import sys
from pathlib import Path

import torch

from tests.fixtures.lora_test_fixtures import DummyLoRALayer, create_dummy_spatial_mask


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestMaskResizing:
    """Tests for mask resizing to different spatial dimensions."""

    def test_mask_resized_to_layer_spatial_dims(self):
        """Test that mask is properly resized to match layer spatial dimensions."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Create mask at one resolution
        mask_64x64 = create_dummy_spatial_mask(64, 64, "left_half")
        layer = DummyLoRALayer()

        # Act: Patch with different target resolution
        patch_lora_layer_with_spatial_mask(layer, mask_64x64, spatial_dims=(32, 32))

        # The mask should be internally resized to 32x32
        # Test by checking forward pass works with 32x32 spatial tokens
        hidden_states = torch.randn(1, 32 * 32, 768)  # 32x32 = 1024 tokens
        output = layer(hidden_states)

        # Assert: Output shape correct
        assert output.shape == (1, 1024, 768)

        # Cleanup
        unpatch_lora_layer(layer)

    def test_mask_handles_different_resolutions(self):
        """Test mask works correctly at different spatial resolutions."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Test multiple resolutions
        resolutions = [(16, 16), (32, 32), (64, 64), (128, 128)]

        for H, W in resolutions:
            layer = DummyLoRALayer()

            # Create mask at base resolution
            mask = create_dummy_spatial_mask(64, 64, "left_half")

            # Patch layer
            patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(H, W))

            # Test forward pass
            hidden_states = torch.randn(1, H * W, 768)
            output = layer(hidden_states)

            # Assert: Works without error
            assert output.shape == (1, H * W, 768), f"Failed at resolution {H}x{W}"

            # Cleanup before next iteration
            unpatch_lora_layer(layer)

    def test_mask_interpolation_preserves_regions(self):
        """Test that bilinear interpolation preserves mask regions reasonably."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Half mask at high resolution
        mask_128x128 = create_dummy_spatial_mask(128, 128, "left_half")
        layer = DummyLoRALayer()

        # Act: Resize to lower resolution
        patch_lora_layer_with_spatial_mask(layer, mask_128x128, spatial_dims=(32, 32))

        # Apply to hidden states
        hidden_states = torch.randn(1, 32 * 32, 768)
        output = layer(hidden_states)

        # Reshape to spatial
        spatial_output = output.view(1, 32, 32, 768)

        # Assert: Left half should still be active (most tokens non-zero)
        left_half = spatial_output[:, :, :16, :]
        right_half = spatial_output[:, :, 16:, :]

        # Interpolation may cause some bleeding, but should preserve general pattern
        left_nonzero_ratio = (left_half != 0.0).float().mean()
        right_nonzero_ratio = (right_half != 0.0).float().mean()

        assert left_nonzero_ratio > 0.8, "Left half should be mostly active"
        assert right_nonzero_ratio < 0.2, "Right half should be mostly blocked"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_upsample_mask_preserves_pattern(self):
        """Test that upsampling mask from low to high resolution works."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Small mask
        mask_16x16 = create_dummy_spatial_mask(16, 16, "left_half")
        layer = DummyLoRALayer()

        # Act: Upsample to higher resolution
        patch_lora_layer_with_spatial_mask(layer, mask_16x16, spatial_dims=(64, 64))

        # Apply to hidden states
        hidden_states = torch.randn(1, 64 * 64, 768)
        output = layer(hidden_states)

        # Reshape to spatial
        spatial_output = output.view(1, 64, 64, 768)

        # Assert: Pattern should be preserved
        left_half = spatial_output[:, :, :32, :]
        right_half = spatial_output[:, :, 32:, :]

        left_nonzero_ratio = (left_half != 0.0).float().mean()
        right_nonzero_ratio = (right_half != 0.0).float().mean()

        assert left_nonzero_ratio > 0.9, "Left half should be mostly active after upsample"
        assert right_nonzero_ratio < 0.1, "Right half should be mostly blocked after upsample"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_non_square_mask(self):
        """Test masking with non-square mask dimensions applied to square spatial."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Create non-square mask (32x128) with left half pattern
        mask = torch.zeros(1, 1, 32, 128)
        mask[:, :, :, :64] = 1.0  # Left half

        layer = DummyLoRALayer()

        # Act: Apply to square spatial dims (64x64 = 4096 tokens)
        # The mask will be resized to 64x64
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Forward pass with square spatial sequence
        hidden_states = torch.randn(1, 64 * 64, 768)
        output = layer(hidden_states)

        # Assert: Works without error
        assert output.shape == (1, 64 * 64, 768)

        # Reshape to spatial and verify pattern
        spatial_output = output.view(1, 64, 64, 768)
        left_half = spatial_output[:, :, :32, :]
        right_half = spatial_output[:, :, 32:, :]

        # Pattern should still hold (mask resized from 32x128 to 64x64)
        left_nonzero_ratio = (left_half != 0.0).float().mean()
        right_nonzero_ratio = (right_half != 0.0).float().mean()

        assert left_nonzero_ratio > 0.8, "Left half should be mostly active"
        assert right_nonzero_ratio < 0.2, "Right half should be mostly blocked"

        # Cleanup
        unpatch_lora_layer(layer)


class TestCenterMaskPreservation:
    """Tests for center mask pattern preservation across resolutions."""

    def test_center_mask_downsample(self):
        """Test center mask pattern is preserved when downsampling."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Large center mask
        mask_128x128 = create_dummy_spatial_mask(128, 128, "center")
        layer = DummyLoRALayer()

        # Act: Downsample to 32x32
        patch_lora_layer_with_spatial_mask(layer, mask_128x128, spatial_dims=(32, 32))

        hidden_states = torch.randn(1, 32 * 32, 768)
        output = layer(hidden_states)

        # Reshape to spatial
        spatial_output = output.view(1, 32, 32, 768)

        # Assert: Center should be active, edges should be blocked
        center = spatial_output[:, 8:24, 8:24, :]  # Center 16x16
        edges_top = spatial_output[:, :4, :, :]
        edges_bottom = spatial_output[:, -4:, :, :]
        edges_left = spatial_output[:, :, :4, :]
        edges_right = spatial_output[:, :, -4:, :]

        center_ratio = (center != 0.0).float().mean()
        edges_ratio = (
            sum(
                [
                    (edges_top != 0.0).float().mean(),
                    (edges_bottom != 0.0).float().mean(),
                    (edges_left != 0.0).float().mean(),
                    (edges_right != 0.0).float().mean(),
                ]
            )
            / 4
        )

        assert center_ratio > 0.8, "Center should be mostly active"
        assert edges_ratio < 0.3, "Edges should be mostly blocked"

        # Cleanup
        unpatch_lora_layer(layer)


class TestCheckerboardPattern:
    """Tests for checkerboard mask pattern handling."""

    def test_checkerboard_pattern_preserved(self):
        """Test that checkerboard pattern is reasonably preserved."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Checkerboard mask
        mask = create_dummy_spatial_mask(64, 64, "checkerboard")
        layer = DummyLoRALayer()

        # Act: Patch at same resolution
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        hidden_states = torch.randn(1, 64 * 64, 768)
        output = layer(hidden_states)

        # Reshape to spatial
        spatial_output = output.view(1, 64, 64, 768)

        # Assert: Roughly half should be zero, half non-zero
        nonzero_ratio = (spatial_output != 0.0).float().mean()

        # Checkerboard should result in ~50% active
        assert 0.4 < nonzero_ratio < 0.6, f"Checkerboard should be ~50% active, got {nonzero_ratio:.2f}"

        # Cleanup
        unpatch_lora_layer(layer)


class TestNonSquareLatentDimensions:
    """Tests for non-square aspect ratios (e.g., 1376x960 → 172x120 latents)."""

    def test_non_square_latent_dims_with_explicit_hint(self):
        """Test masking works correctly with non-square latent dimensions when hint is provided."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Simulate 1376x960 image → 172x120 latents
        latent_h, latent_w = 172, 120
        num_tokens = latent_h * latent_w  # 20640 tokens

        # Create mask at mask resolution (e.g., 256x256 from UI)
        mask = torch.zeros(1, 1, 256, 256)
        mask[:, :, :, :128] = 1.0  # Left half

        layer = DummyLoRALayer()

        # Act: Apply with explicit latent_spatial_dims hint
        patch_lora_layer_with_spatial_mask(
            layer, mask, spatial_dims=(256, 256), latent_spatial_dims=(latent_h, latent_w)
        )

        # Forward pass with non-square sequence
        hidden_states = torch.randn(1, num_tokens, 768)
        output = layer(hidden_states)

        # Assert: Output shape correct
        assert output.shape == (1, num_tokens, 768)

        # Reshape to non-square spatial layout
        spatial_output = output.view(1, latent_h, latent_w, 768)

        # Left half should be active, right half blocked
        left_half = spatial_output[:, :, : latent_w // 2, :]  # First 60 columns
        right_half = spatial_output[:, :, latent_w // 2 :, :]  # Last 60 columns

        left_nonzero_ratio = (left_half != 0.0).float().mean()
        right_nonzero_ratio = (right_half != 0.0).float().mean()

        assert left_nonzero_ratio > 0.8, f"Left half should be mostly active, got {left_nonzero_ratio:.2f}"
        assert right_nonzero_ratio < 0.2, f"Right half should be mostly blocked, got {right_nonzero_ratio:.2f}"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_non_square_latent_dims_portrait_orientation(self):
        """Test masking with portrait aspect ratio (e.g., 960x1376 → 120x172 latents)."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Portrait orientation (height > width in latent space)
        latent_h, latent_w = 120, 172
        num_tokens = latent_h * latent_w  # 20640 tokens

        # Top half mask
        mask = torch.zeros(1, 1, 256, 256)
        mask[:, :, :128, :] = 1.0  # Top half

        layer = DummyLoRALayer()

        # Act: Apply with explicit latent_spatial_dims hint
        patch_lora_layer_with_spatial_mask(
            layer, mask, spatial_dims=(256, 256), latent_spatial_dims=(latent_h, latent_w)
        )

        # Forward pass
        hidden_states = torch.randn(1, num_tokens, 768)
        output = layer(hidden_states)

        # Reshape to portrait spatial layout
        spatial_output = output.view(1, latent_h, latent_w, 768)

        # Top half should be active, bottom half blocked
        top_half = spatial_output[:, : latent_h // 2, :, :]
        bottom_half = spatial_output[:, latent_h // 2 :, :, :]

        top_nonzero_ratio = (top_half != 0.0).float().mean()
        bottom_nonzero_ratio = (bottom_half != 0.0).float().mean()

        assert top_nonzero_ratio > 0.8, f"Top half should be active, got {top_nonzero_ratio:.2f}"
        assert bottom_nonzero_ratio < 0.2, f"Bottom half should be blocked, got {bottom_nonzero_ratio:.2f}"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_non_square_with_joint_attention(self):
        """Test non-square latents with joint attention (image + text tokens)."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Non-square latents with text tokens
        latent_h, latent_w = 172, 120
        num_image_tokens = latent_h * latent_w  # 20640
        num_text_tokens = 64
        total_tokens = num_image_tokens + num_text_tokens  # 20704

        # All-zero mask (blocks image, should not block text)
        mask = create_dummy_spatial_mask(256, 256, "all_zeros")

        layer = DummyLoRALayer()

        # Act: Apply with explicit latent_spatial_dims
        patch_lora_layer_with_spatial_mask(
            layer, mask, spatial_dims=(256, 256), latent_spatial_dims=(latent_h, latent_w)
        )

        # Forward pass with joint attention
        hidden_states = torch.randn(1, total_tokens, 768)
        output = layer(hidden_states)

        # Assert: Image tokens blocked, text tokens pass through
        image_output = output[:, :num_image_tokens, :]
        text_output = output[:, num_image_tokens:, :]

        assert torch.all(image_output == 0.0), "Image tokens should be blocked"
        assert torch.any(text_output != 0.0), "Text tokens should pass through"

        # Cleanup
        unpatch_lora_layer(layer)


class TestJointAttention:
    """Tests for joint attention sequences (image + text tokens)."""

    def test_joint_attention_masks_only_image_tokens(self):
        """Test that joint attention sequences only mask image tokens."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Create mask that blocks everything
        mask = create_dummy_spatial_mask(64, 64, "all_zeros")
        layer = DummyLoRALayer()

        # Act: Apply to sequence with image + text tokens
        # 64x64 = 4096 image tokens + 32 text tokens = 4128 total
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Forward pass with joint attention sequence
        hidden_states = torch.randn(1, 4096 + 32, 768)
        output = layer(hidden_states)

        # Assert: Shape preserved
        assert output.shape == (1, 4128, 768)

        # Image tokens (first 4096) should be blocked
        image_output = output[:, :4096, :]
        assert torch.all(image_output == 0.0), "Image tokens should be blocked by mask"

        # Text tokens (last 32) should NOT be blocked
        text_output = output[:, 4096:, :]
        assert torch.any(text_output != 0.0), "Text tokens should pass through unchanged"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_joint_attention_with_partial_mask(self):
        """Test joint attention with partial mask (left half)."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Create left-half mask
        mask = create_dummy_spatial_mask(64, 64, "left_half")
        layer = DummyLoRALayer()

        # Act: Apply to joint attention sequence
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Forward pass: 4096 image + 32 text = 4128
        hidden_states = torch.randn(1, 4128, 768)
        output = layer(hidden_states)

        # Reshape image tokens to spatial
        image_output = output[:, :4096, :].view(1, 64, 64, 768)
        left_half = image_output[:, :, :32, :]
        right_half = image_output[:, :, 32:, :]

        # Left half should be active, right half blocked
        assert torch.any(left_half != 0.0), "Left half of image should be active"
        assert torch.all(right_half == 0.0), "Right half of image should be blocked"

        # Text tokens should be unchanged
        text_output = output[:, 4096:, :]
        assert torch.any(text_output != 0.0), "Text tokens should pass through"

        # Cleanup
        unpatch_lora_layer(layer)
