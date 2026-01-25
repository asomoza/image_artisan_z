"""Test cleanup and memory safety of LoRA spatial masking."""

import gc
import sys
from pathlib import Path

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


class TestRegistryManagement:
    """Tests for patch registry management."""

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
        unpatch_all_lora_layers()

        # Assert: Registry is empty
        assert get_patched_layer_count() == 0

    def test_repatch_same_layer_reuses_original(self):
        """Test that patching a layer twice uses the same original forward."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask1 = create_dummy_spatial_mask(64, 64, "left_half")
        mask2 = create_dummy_spatial_mask(64, 64, "right_half")
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Get original output
        original_output = layer(hidden_states)

        # Act: Patch twice
        patch_lora_layer_with_spatial_mask(layer, mask1, spatial_dims=(64, 64))
        assert get_patched_layer_count() == 1

        patch_lora_layer_with_spatial_mask(layer, mask2, spatial_dims=(64, 64))  # Repatch
        # Should still only be 1 entry (same layer)
        assert get_patched_layer_count() == 1

        # Unpatch once should restore original
        unpatch_lora_layer(layer)
        restored_output = layer(hidden_states)

        # Assert: Restored output matches original
        torch.testing.assert_close(restored_output, original_output)

    def test_multiple_layers_tracked_independently(self):
        """Test that multiple layers are tracked independently in registry."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer1 = DummyLoRALayer()
        layer2 = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "center")

        # Patch both layers
        patch_lora_layer_with_spatial_mask(layer1, mask, spatial_dims=(64, 64))
        patch_lora_layer_with_spatial_mask(layer2, mask, spatial_dims=(64, 64))

        assert get_patched_layer_count() == 2

        # Unpatch only layer1
        unpatch_lora_layer(layer1)
        assert get_patched_layer_count() == 1

        # Unpatch layer2
        unpatch_lora_layer(layer2)
        assert get_patched_layer_count() == 0


class TestMemorySafety:
    """Tests for memory safety and leak prevention."""

    def test_patch_unpatch_does_not_leak_memory(self):
        """Test that patching/unpatching doesn't leak references."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "center")

        # Force garbage collection to get baseline
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Act: Patch and unpatch multiple times
        for _ in range(10):
            patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
            unpatch_lora_layer(layer)

        # Force garbage collection
        gc.collect()
        final_objects = len(gc.get_objects())

        # Assert: No significant object growth
        # Allow some variance due to GC timing
        object_growth = final_objects - initial_objects
        assert object_growth < 100, f"Object count grew by {object_growth}, possible memory leak"

        # Ensure registry is clean
        assert get_patched_layer_count() == 0

    def test_registry_does_not_prevent_layer_gc(self):
        """Test that registry doesn't prevent garbage collection of unpached layers."""
        import weakref

        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Create layer and track with weak reference
        layer = DummyLoRALayer()
        weak_layer = weakref.ref(layer)
        mask = create_dummy_spatial_mask(64, 64, "center")

        # Patch and unpatch
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        unpatch_lora_layer(layer)

        # Delete strong reference
        del layer

        # Force garbage collection
        gc.collect()

        # Assert: Layer was garbage collected
        assert weak_layer() is None, "Layer should be garbage collected after unpatching"
        assert get_patched_layer_count() == 0


class TestConcurrentPatching:
    """Tests for concurrent/sequential patching scenarios."""

    def test_patch_multiple_adapters_same_transformer(self):
        """Test patching multiple adapters on the same transformer."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_adapter_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange
        transformer = DummyTransformerWithLoRA(num_layers=4)

        mask1 = create_dummy_spatial_mask(64, 64, "left_half")
        mask2 = create_dummy_spatial_mask(64, 64, "right_half")

        # Act: Patch two adapters (they share layers in this dummy implementation)
        count1 = patch_lora_adapter_with_spatial_mask(transformer, "adapter1", mask1)
        count2 = patch_lora_adapter_with_spatial_mask(transformer, "adapter2", mask2)

        # Assert: Both patching operations completed
        assert count1 == 4
        assert count2 == 4
        # Note: Same layers patched twice means they're re-patched with mask2

        # Cleanup
        unpatch_all_lora_layers()

    def test_cleanup_after_generation_cycle(self):
        """Test cleanup mimicking a generation cycle."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_adapter_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Simulate multiple generation cycles
        for cycle in range(3):
            transformer = DummyTransformerWithLoRA(num_layers=4)
            mask = create_dummy_spatial_mask(64, 64, "center")

            # Setup for generation
            count = patch_lora_adapter_with_spatial_mask(transformer, "lora", mask)
            assert count == 4
            assert get_patched_layer_count() >= 4

            # Simulate forward passes
            hidden_states = create_dummy_hidden_states(1, 4096, 768)
            for layer in transformer.layers:
                output = layer(hidden_states)
                assert output.shape == hidden_states.shape

            # Cleanup after generation
            unpatch_count = unpatch_all_lora_layers()
            assert unpatch_count >= 4
            assert get_patched_layer_count() == 0


class TestEdgeCases:
    """Tests for edge cases in cleanup."""

    def test_unpatch_after_layer_deleted(self):
        """Test behavior when layer is deleted before unpatching."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_layer_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange
        layer = DummyLoRALayer()
        mask = create_dummy_spatial_mask(64, 64, "center")

        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))
        assert get_patched_layer_count() == 1

        # Don't unpatch, just clear registry
        # In real scenario, this could happen if transformer is deleted
        # The cleanup should still work
        count = unpatch_all_lora_layers()

        # Assert: Cleanup succeeded
        assert count == 1
        assert get_patched_layer_count() == 0

    def test_patch_with_very_small_mask(self):
        """Test patching with very small mask dimensions."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: Very small mask
        mask = create_dummy_spatial_mask(4, 4, "left_half")
        layer = DummyLoRALayer()

        # Act: Patch with larger target resolution
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Should work without error
        hidden_states = create_dummy_hidden_states(1, 4096, 768)
        output = layer(hidden_states)

        assert output.shape == (1, 4096, 768)

        # Cleanup
        unpatch_lora_layer(layer)

    def test_patch_with_single_pixel_mask(self):
        """Test patching with 1x1 mask (edge case)."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            patch_lora_layer_with_spatial_mask,
            unpatch_lora_layer,
        )

        # Arrange: 1x1 mask (all ones)
        mask = torch.ones(1, 1, 1, 1)
        layer = DummyLoRALayer()

        # Act: Patch
        patch_lora_layer_with_spatial_mask(layer, mask, spatial_dims=(64, 64))

        # Should upsample mask to 64x64
        hidden_states = create_dummy_hidden_states(1, 4096, 768)
        output = layer(hidden_states)

        # Output should be non-zero (mask was all ones)
        assert torch.any(output != 0.0)

        # Cleanup
        unpatch_lora_layer(layer)
