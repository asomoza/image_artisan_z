"""Test LoRA forward pass patching for spatial masking."""

import sys
from pathlib import Path

import pytest
import torch

from tests.fixtures.lora_test_fixtures import (
    DummyLoRALayer,
    DummyTransformerWithLoRA,
    create_dummy_hidden_states,
    create_dummy_spatial_mask,
)


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPatchLoraLayerWithSpatialMask:
    """Tests for patch_lora_layer_with_spatial_mask function."""

    def test_patch_lora_layer_with_uniform_mask(self):
        """Test patching a single LoRA layer with uniform mask (all ones)."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "all_ones")
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Get original output
        original_output = layer(hidden_states)

        # Act: Patch layer with all-ones mask (should not change output)
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Apply patched layer
        patched_output = layer(hidden_states)

        # Assert: Outputs should be identical (mask is all ones)
        torch.testing.assert_close(patched_output, original_output)

        # Cleanup
        unpatch_lora_layer(layer)

    def test_patch_lora_layer_with_zero_mask(self):
        """Test patching with all-zeros mask blocks LoRA completely."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "all_zeros")
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Act: Patch layer with all-zeros mask
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Apply patched layer
        patched_output = layer(hidden_states)

        # Assert: Output should be all zeros (LoRA blocked)
        assert torch.all(patched_output == 0.0), "All-zeros mask should block LoRA output"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_patch_lora_layer_with_half_mask(self):
        """Test patching with half mask applies LoRA to half of spatial regions."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "left_half")
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Act: Patch layer
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Apply patched layer
        patched_output = layer(hidden_states)

        # Reshape to spatial: [B, N, D] -> [B, H, W, D]
        B, N, D = patched_output.shape
        H, W = 64, 64
        spatial_output = patched_output.view(B, H, W, D)

        # Assert: Left half should have non-zero values, right half should be zero
        left_half = spatial_output[:, :, : W // 2, :]
        right_half = spatial_output[:, :, W // 2 :, :]

        assert torch.any(left_half != 0.0), "Left half should have LoRA effects"
        assert torch.all(right_half == 0.0), "Right half should be blocked"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_patch_preserves_layer_structure(self):
        """Test that patching doesn't break layer structure."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "center")

        # Store original attributes
        original_in_features = layer.in_features
        original_out_features = layer.out_features
        original_scaling = layer.scaling

        # Act: Patch layer
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Assert: Layer structure unchanged
        assert layer.in_features == original_in_features
        assert layer.out_features == original_out_features
        assert layer.scaling == original_scaling

        # Cleanup
        unpatch_lora_layer(layer)

    def test_unpatch_lora_layer_restores_original(self):
        """Test that unpatching restores original forward behavior."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "left_half")
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Get original output
        original_output = layer(hidden_states)

        # Act: Patch, then unpatch
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        patched_output = layer(hidden_states)

        unpatch_lora_layer(layer)
        restored_output = layer(hidden_states)

        # Assert: Restored output matches original
        torch.testing.assert_close(restored_output, original_output)

        # Assert: Patched output was different (mask had effect)
        assert not torch.allclose(patched_output, original_output), "Patched output should differ from original"

    def test_patch_with_3d_mask(self):
        """Test patching with 3D mask [1, H, W] (auto-expanded to 4D)."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        # Create 3D mask [1, H, W] instead of [1, 1, H, W]
        mask = torch.zeros(1, 64, 64)
        mask[:, :, :32] = 1.0  # Left half
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Act: Patch layer (should handle 3D mask)
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        patched_output = layer(hidden_states)

        # Reshape to spatial
        spatial_output = patched_output.view(1, 64, 64, 768)

        # Assert: Mask applied correctly
        left_half = spatial_output[:, :, :32, :]
        right_half = spatial_output[:, :, 32:, :]

        assert torch.any(left_half != 0.0), "Left half should have LoRA effects"
        assert torch.all(right_half == 0.0), "Right half should be blocked"

        # Cleanup
        unpatch_lora_layer(layer)

    def test_patch_with_gradient_mask(self):
        """Test patching with gradient mask produces gradient output."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "gradient_h")  # Horizontal gradient
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Get unmasked output for comparison
        unmasked_output = layer(hidden_states)

        # Act: Patch layer
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        patched_output = layer(hidden_states)

        # Reshape to spatial
        spatial_patched = patched_output.view(1, 64, 64, 768)
        spatial_unmasked = unmasked_output.view(1, 64, 64, 768)

        # Assert: Left edge (mask ~0) should be near zero
        left_edge_patched = spatial_patched[:, :, :4, :].abs().mean()
        left_edge_unmasked = spatial_unmasked[:, :, :4, :].abs().mean()

        # Assert: Right edge (mask ~1) should be near original
        right_edge_patched = spatial_patched[:, :, -4:, :]
        right_edge_unmasked = spatial_unmasked[:, :, -4:, :]

        # Left edge should be much smaller than right edge
        right_edge_mean = right_edge_patched.abs().mean()
        assert left_edge_patched < right_edge_mean * 0.2, "Gradient mask should produce gradient output"

        # Right edge should be close to unmasked
        torch.testing.assert_close(right_edge_patched, right_edge_unmasked, rtol=0.1, atol=0.01)

        # Cleanup
        unpatch_lora_layer(layer)

    def test_invalid_mask_shape_raises_error(self):
        """Test that invalid mask shapes raise ValueError."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
        )

        layer = DummyLoRALayer()

        # 2D mask should raise error
        mask_2d = torch.zeros(64, 64)
        with pytest.raises(ValueError, match="3D or 4D"):
            patch_lora_layer_with_spatial_mask(layer, mask_2d, spatial_dims=(64, 64))

        # 5D mask should raise error
        mask_5d = torch.zeros(1, 1, 1, 64, 64)
        with pytest.raises(ValueError, match="3D or 4D"):
            patch_lora_layer_with_spatial_mask(layer, mask_5d, spatial_dims=(64, 64))


class TestUnpatchLoraLayer:
    """Tests for unpatch_lora_layer function."""

    def test_unpatch_unpatched_layer_raises_error(self):
        """Test that unpatching an unpatched layer raises ValueError."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            unpatch_lora_layer,
        )

        layer = DummyLoRALayer()

        with pytest.raises(ValueError, match="was not patched"):
            unpatch_lora_layer(layer)

    def test_unpatch_after_unpatch_raises_error(self):
        """Test that double unpatching raises ValueError."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "all_ones")

        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        unpatch_lora_layer(layer)

        with pytest.raises(ValueError, match="was not patched"):
            unpatch_lora_layer(layer)


class TestUnpatchAllLoraLayers:
    """Tests for unpatch_all_lora_layers function."""

    def test_unpatch_all_cleans_registry(self):
        """Test that unpatch_all_lora_layers clears the global registry."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_layer_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange: Patch multiple layers
        layers = [DummyLoRALayer() for _ in range(3)]
        mask = create_dummy_spatial_mask(64, 64, "center")

        for layer in layers:
            patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Registry should have 3 entries
        assert get_patched_layer_count() == 3

        # Act: Unpatch all
        count = unpatch_all_lora_layers()

        # Assert: Registry is empty
        assert count == 3
        assert get_patched_layer_count() == 0

    def test_unpatch_all_restores_forward_methods(self):
        """Test that unpatch_all restores all forward methods."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange
        layers = [DummyLoRALayer() for _ in range(3)]
        mask = create_dummy_spatial_mask(64, 64, "left_half")
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Get original outputs
        original_outputs = [layer(hidden_states) for layer in layers]

        # Patch all layers
        for layer in layers:
            patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Act: Unpatch all
        unpatch_all_lora_layers()

        # Assert: All layers restored
        for layer, original_output in zip(layers, original_outputs):
            restored_output = layer(hidden_states)
            torch.testing.assert_close(restored_output, original_output)

    def test_unpatch_all_on_empty_registry(self):
        """Test that unpatch_all_lora_layers handles empty registry."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            unpatch_all_lora_layers,
        )

        # Ensure registry is empty
        unpatch_all_lora_layers()
        assert get_patched_layer_count() == 0

        # Act: Call again on empty registry
        count = unpatch_all_lora_layers()

        # Assert: Should return 0, not raise error
        assert count == 0


class TestPatchLoraAdapterWithSpatialMask:
    """Tests for patch_lora_adapter_with_spatial_mask function."""

    def test_patch_adapter_patches_all_layers(self):
        """Test that patching an adapter patches all its layers."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_adapter_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange
        transformer = DummyTransformerWithLoRA(num_layers=4)
        mask = create_dummy_spatial_mask(64, 64, "center")

        # Act: Patch adapter
        count = patch_lora_adapter_with_spatial_mask(transformer, "test_lora", mask)

        # Assert: All layers patched
        assert count == 4
        assert get_patched_layer_count() == 4

        # Cleanup
        unpatch_all_lora_layers()

    def test_patch_adapter_applies_mask_to_all_layers(self):
        """Test that mask is applied to all adapter layers."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_adapter_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange
        transformer = DummyTransformerWithLoRA(num_layers=4)
        mask = create_dummy_spatial_mask(64, 64, "all_zeros")  # Block everything
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Patch adapter
        patch_lora_adapter_with_spatial_mask(transformer, "test_lora", mask)

        # Act: Apply each layer
        for layer in transformer.layers:
            output = layer(hidden_states)
            # Assert: Output is zero (blocked by mask)
            assert torch.all(output == 0.0), "All layers should have zero output"

        # Cleanup
        unpatch_all_lora_layers()

    def test_patch_adapter_with_nonexistent_adapter(self):
        """Test patching nonexistent adapter returns 0 and logs warning."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_adapter_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange: Use a transformer without get_lora_layers returning empty
        class EmptyLoRATransformer:
            def get_lora_layers(self, adapter_name):
                return {}

        transformer = EmptyLoRATransformer()
        mask = create_dummy_spatial_mask(64, 64, "center")

        # Act: Patch nonexistent adapter
        count = patch_lora_adapter_with_spatial_mask(transformer, "nonexistent_adapter", mask)

        # Assert: No layers patched
        assert count == 0
        assert get_patched_layer_count() == 0

        # Cleanup
        unpatch_all_lora_layers()


class TestBatchHandling:
    """Tests for batch size handling in patching."""

    def test_single_mask_broadcast_to_batch(self):
        """Test that single mask broadcasts to larger batch."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "left_half")  # [1, 1, 64, 64]
        # Batch size 4
        hidden_states = create_dummy_hidden_states(4, 4096, 768)

        # Act: Patch layer
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        patched_output = layer(hidden_states)

        # Assert: All batches masked correctly
        spatial_output = patched_output.view(4, 64, 64, 768)

        for b in range(4):
            left_half = spatial_output[b, :, :32, :]
            right_half = spatial_output[b, :, 32:, :]

            assert torch.any(left_half != 0.0), f"Batch {b} left half should have values"
            assert torch.all(right_half == 0.0), f"Batch {b} right half should be zero"

        # Cleanup
        unpatch_lora_layer(layer)


class TestNonSpatialInputs:
    """Tests for non-spatial (2D) input handling like adaLN layers."""

    def test_2d_input_passes_through_unchanged(self):
        """Test that 2D inputs (adaLN layers) pass through without masking."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "all_zeros")  # Would block everything
        # 2D input [B, D] - simulating adaLN conditioning
        hidden_states_2d = torch.randn(2, 768)

        # Get original output
        original_output = layer(hidden_states_2d)

        # Act: Patch layer
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        patched_output = layer(hidden_states_2d)

        # Assert: 2D output should be identical (mask not applied)
        torch.testing.assert_close(patched_output, original_output)

        # Cleanup
        unpatch_lora_layer(layer)

    def test_3d_spatial_input_gets_masked(self):
        """Test that 3D spatial inputs still get masked correctly."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "all_zeros")  # Block everything
        # 3D input [B, N, D] - spatial hidden states
        hidden_states_3d = create_dummy_hidden_states(1, 4096, 768)

        # Get original output
        original_output = layer(hidden_states_3d)

        # Act: Patch layer
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        patched_output = layer(hidden_states_3d)

        # Assert: 3D output should be zero (mask applied)
        assert torch.all(patched_output == 0.0), "3D inputs should be masked"

        # Cleanup
        unpatch_lora_layer(layer)
