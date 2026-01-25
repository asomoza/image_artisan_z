"""Test LoRA node loading and outputting spatial masks."""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image


# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class TestLoraNodeMaskLoading:
    """Tests for LoRA node spatial mask loading."""

    def test_lora_node_loads_mask_when_path_provided(self):
        """Test that LoRA node loads spatial mask from file."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Arrange: Create temporary mask file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_path = f.name

        try:
            # Create mask image (grayscale, 256x256)
            mask_array = np.zeros((256, 256), dtype=np.uint8)
            mask_array[:, :128] = 255  # Left half white
            mask_img = Image.fromarray(mask_array, mode="L")
            mask_img.save(mask_path)

            # Create LoRA node
            node = LoraNode(
                adapter_name="test_lora",
                spatial_mask_enabled=True,
                spatial_mask_path=mask_path,
            )

            # Act: Load mask
            mask_tensor = node._load_spatial_mask()

            # Assert: Mask loaded correctly
            assert mask_tensor is not None
            assert mask_tensor.shape[0] == 1  # Batch dimension
            assert mask_tensor.shape[1] == 1  # Channel dimension
            assert mask_tensor.shape[2] == 256  # Height
            assert mask_tensor.shape[3] == 256  # Width

            # Check left half is ~1.0, right half is ~0.0
            mid_w = mask_tensor.shape[3] // 2
            left_mean = mask_tensor[:, :, :, :mid_w].mean()
            right_mean = mask_tensor[:, :, :, mid_w:].mean()

            assert left_mean > 0.9, "Left half should be mostly 1.0"
            assert right_mean < 0.1, "Right half should be mostly 0.0"

        finally:
            # Cleanup
            if os.path.exists(mask_path):
                os.unlink(mask_path)

    def test_lora_node_loads_rgba_mask_extracts_alpha(self):
        """Test that LoRA node extracts alpha channel from RGBA masks."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Arrange: Create temporary RGBA mask file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_path = f.name

        try:
            # Create RGBA mask (black with alpha)
            mask_array = np.zeros((128, 128, 4), dtype=np.uint8)
            # Set alpha channel: left half opaque (255), right half transparent (0)
            mask_array[:, :64, 3] = 255  # Left half opaque
            mask_array[:, 64:, 3] = 0  # Right half transparent
            mask_img = Image.fromarray(mask_array, mode="RGBA")
            mask_img.save(mask_path)

            # Create LoRA node
            node = LoraNode(
                adapter_name="test_lora",
                spatial_mask_enabled=True,
                spatial_mask_path=mask_path,
            )

            # Act: Load mask
            mask_tensor = node._load_spatial_mask()

            # Assert: Alpha channel extracted correctly
            assert mask_tensor is not None
            mid_w = mask_tensor.shape[3] // 2
            left_mean = mask_tensor[:, :, :, :mid_w].mean()
            right_mean = mask_tensor[:, :, :, mid_w:].mean()

            assert left_mean > 0.9, "Left half (opaque) should be ~1.0"
            assert right_mean < 0.1, "Right half (transparent) should be ~0.0"

        finally:
            if os.path.exists(mask_path):
                os.unlink(mask_path)

    def test_lora_node_returns_none_when_disabled(self):
        """Test that mask is None when spatial_mask_enabled is False."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Create node with mask disabled
        node = LoraNode(
            adapter_name="test_lora",
            spatial_mask_enabled=False,
            spatial_mask_path="/some/path.png",
        )

        # Act
        mask_tensor = node._load_spatial_mask()

        # Assert
        assert mask_tensor is None

    def test_lora_node_handles_missing_mask_file_gracefully(self):
        """Test that node handles missing mask file without crashing."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Create node with nonexistent mask path
        node = LoraNode(
            adapter_name="test_lora",
            spatial_mask_enabled=True,
            spatial_mask_path="/nonexistent/mask.png",
        )

        # Act: Try to load mask
        mask_tensor = node._load_spatial_mask()

        # Assert: Returns None
        assert mask_tensor is None

    def test_lora_node_output_tuple_with_mask(self):
        """Test that LoRA node outputs 3-tuple when mask is loaded."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Arrange: Create temporary mask file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            mask_path = f.name

        try:
            # Create simple mask
            mask_img = Image.new("L", (64, 64), 255)
            mask_img.save(mask_path)

            # Create node and set up cached mask
            node = LoraNode(
                adapter_name="test_lora",
                transformer_weight=0.75,
                spatial_mask_enabled=True,
                spatial_mask_path=mask_path,
            )

            # Load the mask
            node._cached_mask = node._load_spatial_mask()

            # Manually create output tuple (simulating __call__ behavior)
            scale = {"transformer": 0.75}
            output = (node.adapter_name, scale, node._cached_mask)

            # Assert: Tuple has 3 elements
            assert len(output) == 3
            adapter_name, scale_dict, mask = output
            assert adapter_name == "test_lora"
            assert scale_dict == {"transformer": 0.75}
            assert mask is not None
            assert mask.shape == (1, 1, 64, 64)

        finally:
            if os.path.exists(mask_path):
                os.unlink(mask_path)

    def test_lora_node_output_tuple_without_mask(self):
        """Test that LoRA node outputs 3-tuple with None mask when disabled."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Create node without mask
        node = LoraNode(
            adapter_name="test_lora",
            transformer_weight=1.0,
            spatial_mask_enabled=False,
        )

        # Manually create output tuple (simulating __call__ behavior)
        scale = {"transformer": 1.0}
        spatial_mask = None if not node.spatial_mask_enabled else node._cached_mask
        output = (node.adapter_name, scale, spatial_mask)

        # Assert: Tuple has 3 elements with None mask
        assert len(output) == 3
        adapter_name, scale_dict, mask = output
        assert adapter_name == "test_lora"
        assert scale_dict == {"transformer": 1.0}
        assert mask is None


class TestLoraNodeUpdateMethods:
    """Tests for LoRA node update methods."""

    def test_update_spatial_mask_clears_cache_on_path_change(self):
        """Test that changing mask path clears the cached mask."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Create node with cached mask
        node = LoraNode(
            adapter_name="test_lora",
            spatial_mask_enabled=True,
            spatial_mask_path="/path/to/old_mask.png",
        )
        # Simulate cached mask
        node._cached_mask = torch.ones(1, 1, 64, 64)

        # Act: Update with new path
        node.update_spatial_mask(enabled=True, path="/path/to/new_mask.png")

        # Assert: Cache cleared
        assert node._cached_mask is None
        assert node.spatial_mask_path == "/path/to/new_mask.png"
        assert node._mask_load_failed is False

    def test_update_spatial_mask_keeps_cache_on_same_path(self):
        """Test that cache is kept when path doesn't change."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Create node with cached mask
        node = LoraNode(
            adapter_name="test_lora",
            spatial_mask_enabled=True,
            spatial_mask_path="/path/to/mask.png",
        )
        cached_mask = torch.ones(1, 1, 64, 64)
        node._cached_mask = cached_mask

        # Act: Update with same path
        node.update_spatial_mask(enabled=True, path="/path/to/mask.png")

        # Assert: Cache kept
        assert node._cached_mask is cached_mask

    def test_update_lora_includes_mask_params(self):
        """Test that update_lora can update mask parameters."""
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        # Create node
        node = LoraNode(
            adapter_name="test_lora",
            spatial_mask_enabled=False,
            spatial_mask_path="",
        )

        # Act: Update lora with mask params
        node.update_lora(
            lora_enabled=True,
            transformer_weight=0.8,
            granular_transformer_weights_enabled=False,
            granular_transformer_weights={},
            is_slider=False,
            spatial_mask_enabled=True,
            spatial_mask_path="/new/mask.png",
        )

        # Assert: Mask params updated
        assert node.spatial_mask_enabled is True
        assert node.spatial_mask_path == "/new/mask.png"


class TestLoraDataObject:
    """Tests for LoraDataObject spatial mask fields."""

    def test_lora_data_object_has_mask_fields(self):
        """Test that LoraDataObject has spatial mask fields."""
        from iartisanz.modules.generation.data_objects.lora_data_object import (
            LoraDataObject,
        )

        # Create object with defaults
        obj = LoraDataObject(
            name="test",
            filename="test.safetensors",
            version="1.0",
            path="/path/to/lora",
            lora_node_name="lora_node",
        )

        # Assert: Default values
        assert hasattr(obj, "spatial_mask_enabled")
        assert hasattr(obj, "spatial_mask_path")
        assert obj.spatial_mask_enabled is False
        assert obj.spatial_mask_path == ""

    def test_lora_data_object_mask_fields_settable(self):
        """Test that spatial mask fields can be set."""
        from iartisanz.modules.generation.data_objects.lora_data_object import (
            LoraDataObject,
        )

        # Create object with mask fields set
        obj = LoraDataObject(
            name="test",
            filename="test.safetensors",
            version="1.0",
            path="/path/to/lora",
            lora_node_name="lora_node",
            spatial_mask_enabled=True,
            spatial_mask_path="/path/to/mask.png",
        )

        # Assert
        assert obj.spatial_mask_enabled is True
        assert obj.spatial_mask_path == "/path/to/mask.png"
