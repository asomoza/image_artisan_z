"""Test ZImageDenoiseNode LoRA spatial mask extraction and application."""

import sys
from pathlib import Path

import torch

from tests.fixtures.lora_test_fixtures import DummyTransformerWithLoRA, create_dummy_spatial_mask


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestZImageDenoiseNodeExtractLoraMasks:
    """Tests for ZImageDenoiseNode._extract_lora_masks method."""

    def test_extract_masks_from_single_lora_3tuple(self):
        """Test extracting mask from a single LoRA 3-tuple."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        mask = create_dummy_spatial_mask(64, 64, "left_half")
        node.lora = ("test_lora", {"transformer": 0.75}, mask)

        # Act
        masks = node._extract_lora_masks()

        # Assert
        assert "test_lora" in masks
        torch.testing.assert_close(masks["test_lora"], mask)

    def test_extract_masks_from_single_lora_2tuple_legacy(self):
        """Test that 2-tuple (legacy) format returns no masks."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        node.lora = ("test_lora", {"transformer": 0.75})  # 2-tuple, no mask

        # Act
        masks = node._extract_lora_masks()

        # Assert: No masks extracted (legacy format)
        assert len(masks) == 0

    def test_extract_masks_from_single_lora_with_none_mask(self):
        """Test that 3-tuple with None mask returns no masks."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        node.lora = ("test_lora", {"transformer": 0.75}, None)  # 3-tuple with None

        # Act
        masks = node._extract_lora_masks()

        # Assert: No masks (mask was None)
        assert len(masks) == 0

    def test_extract_masks_from_multiple_loras(self):
        """Test extracting masks from multiple LoRAs."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        mask1 = create_dummy_spatial_mask(64, 64, "left_half")
        mask2 = create_dummy_spatial_mask(64, 64, "right_half")

        node.lora = [
            ("lora_1", {"transformer": 0.5}, mask1),
            ("lora_2", {"transformer": 0.7}, mask2),
            ("lora_3", {"transformer": 1.0}, None),  # No mask
        ]

        # Act
        masks = node._extract_lora_masks()

        # Assert
        assert len(masks) == 2
        assert "lora_1" in masks
        assert "lora_2" in masks
        assert "lora_3" not in masks
        torch.testing.assert_close(masks["lora_1"], mask1)
        torch.testing.assert_close(masks["lora_2"], mask2)

    def test_extract_masks_from_mixed_tuple_lengths(self):
        """Test extracting masks from mixed 2-tuple and 3-tuple formats."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        mask = create_dummy_spatial_mask(64, 64, "center")

        node.lora = [
            ("lora_1", {"transformer": 0.5}),  # 2-tuple (legacy)
            ("lora_2", {"transformer": 0.7}, mask),  # 3-tuple with mask
            ("lora_3", {"transformer": 1.0}, None),  # 3-tuple without mask
        ]

        # Act
        masks = node._extract_lora_masks()

        # Assert: Only lora_2 has a mask
        assert len(masks) == 1
        assert "lora_2" in masks

    def test_extract_masks_returns_empty_when_no_lora(self):
        """Test that empty dict is returned when no LoRA is set."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        node.lora = None

        # Act
        masks = node._extract_lora_masks()

        # Assert
        assert masks == {}


class TestZImageDenoiseNodeLoraPatching:
    """Tests for LoRA spatial mask patching in ZImageDenoiseNode."""

    def test_patch_lora_adapter_integrates_with_denoise(self):
        """Test that patching works with the denoise node pattern."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_adapter_with_spatial_mask,
            unpatch_all_lora_layers,
        )

        # Arrange
        transformer = DummyTransformerWithLoRA(num_layers=4)
        mask = create_dummy_spatial_mask(64, 64, "left_half")

        # Act: Patch (simulating what ZImageDenoiseNode does)
        count = patch_lora_adapter_with_spatial_mask(transformer, "test_lora", mask)

        # Assert
        assert count == 4
        assert get_patched_layer_count() == 4

        # Cleanup (simulating end of generation)
        unpatch_all_lora_layers()
        assert get_patched_layer_count() == 0

    def test_multiple_lora_masks_applied_independently(self):
        """Test that multiple LoRA masks can be applied independently."""
        from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
            get_patched_layer_count,
            patch_lora_adapter_with_spatial_mask,
            unpatch_all_lora_layers,
        )
        from tests.fixtures.lora_test_fixtures import create_dummy_hidden_states

        # Arrange: Two transformers (simulating two different LoRAs)
        transformer1 = DummyTransformerWithLoRA(num_layers=2)
        transformer2 = DummyTransformerWithLoRA(num_layers=3)

        mask1 = create_dummy_spatial_mask(64, 64, "left_half")
        mask2 = create_dummy_spatial_mask(64, 64, "right_half")

        # Act: Patch both
        count1 = patch_lora_adapter_with_spatial_mask(transformer1, "lora1", mask1)
        count2 = patch_lora_adapter_with_spatial_mask(transformer2, "lora2", mask2)

        # Assert: Both patched
        assert count1 == 2
        assert count2 == 3
        assert get_patched_layer_count() == 5

        # Verify masking works
        hidden_states = create_dummy_hidden_states(1, 4096, 768)

        # Transformer1 should have left-half active
        for layer in transformer1.layers:
            output = layer(hidden_states)
            spatial = output.view(1, 64, 64, 768)
            assert torch.any(spatial[:, :, :32, :] != 0)
            assert torch.all(spatial[:, :, 32:, :] == 0)

        # Transformer2 should have right-half active
        for layer in transformer2.layers:
            output = layer(hidden_states)
            spatial = output.view(1, 64, 64, 768)
            assert torch.all(spatial[:, :, :32, :] == 0)
            assert torch.any(spatial[:, :, 32:, :] != 0)

        # Cleanup
        unpatch_all_lora_layers()


class TestZImageDenoiseNodeBackwardCompatibility:
    """Tests for backward compatibility with 2-tuple LoRA format."""

    def test_2tuple_lora_still_works(self):
        """Test that the original 2-tuple LoRA format is still supported."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        node.lora = ("test_lora", {"transformer": 0.75})  # 2-tuple

        # Act: Extract masks (should return empty)
        masks = node._extract_lora_masks()

        # Assert: No masks, but no error
        assert masks == {}

    def test_2tuple_list_still_works(self):
        """Test that list of 2-tuples (original multi-LoRA format) still works."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        # Arrange
        node = ZImageDenoiseNode()
        node.lora = [
            ("lora_1", {"transformer": 0.5}),
            ("lora_2", {"transformer": 0.7}),
        ]

        # Act: Extract masks
        masks = node._extract_lora_masks()

        # Assert: No masks, but no error
        assert masks == {}
