"""Tests for ControlNet inpainting validation logic.

Tests various invalid and valid combinations of controlnet, init_image, and mask.
"""

import pytest

from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes.controlnet_conditioning_node import ControlNetConditioningNode
from iartisanz.modules.generation.graph.nodes.controlnet_model_node import ControlNetModelNode
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from iartisanz.modules.generation.graph.nodes.node import Node


class DummyNode(Node):
    """Dummy node for testing."""

    OUTPUTS = ["value"]

    def __init__(self, value=None):
        super().__init__()
        self._value = value

    def __call__(self):
        self.values["value"] = self._value


class TestControlNetInpaintingValidation:
    """Test validation of controlnet inpainting configurations."""

    def test_mask_without_init_image_raises_error(self):
        """Mask without init_image AND without control_image should raise ValueError.

        Note: Mask with control_image (without init_image) is valid - it's Spatial ControlNet mode.
        This test validates that mask ALONE (no control_image, no init_image) is invalid.
        """
        graph = ImageArtisanZNodeGraph()

        # Add controlnet nodes
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")

        # Add mask but no init_image and no control_image
        graph.add_node(ImageLoadNode(path="/fake/mask.png", grayscale=True), "control_mask_image")

        # Validate should raise error (mask alone is invalid)
        with pytest.raises(ValueError, match="ControlNet mask requires either a control image or init_image"):
            graph.validate_controlnet_inpainting()

    def test_mask_without_controlnet_raises_error(self):
        """Mask without controlnet should raise ValueError."""
        graph = ImageArtisanZNodeGraph()

        # Add mask and init_image nodes without controlnet_conditioning
        graph.add_node(ImageLoadNode(path="/fake/mask.png", grayscale=True), "control_mask_image")
        graph.add_node(ImageLoadNode(path="/fake/init.png"), "control_init_image")

        # Validate should raise error (no controlnet_conditioning node)
        with pytest.raises(ValueError, match="ControlNet mask requires a ControlNet to be configured"):
            graph.validate_controlnet_inpainting()

    def test_init_image_without_controlnet_raises_error(self):
        """Init image without controlnet should raise ValueError."""
        graph = ImageArtisanZNodeGraph()

        # Add init_image node without controlnet
        graph.add_node(ImageLoadNode(path="/fake/init.png"), "control_init_image")

        # Validate should raise error
        with pytest.raises(ValueError, match="ControlNet init_image requires a ControlNet to be configured"):
            graph.validate_controlnet_inpainting()

    def test_init_image_and_mask_without_controlnet_raises_error(self):
        """Init image and mask without controlnet should raise ValueError."""
        graph = ImageArtisanZNodeGraph()

        # Add both mask and init_image without controlnet
        graph.add_node(ImageLoadNode(path="/fake/mask.png", grayscale=True), "control_mask_image")
        graph.add_node(ImageLoadNode(path="/fake/init.png"), "control_init_image")

        # Validate should raise error - init_image check will trigger first
        with pytest.raises(ValueError, match="ControlNet (mask|init_image) requires a ControlNet to be configured"):
            graph.validate_controlnet_inpainting()

    def test_init_image_with_controlnet_no_mask_is_valid(self):
        """Init image with controlnet but no mask should be valid."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet and init_image
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")
        graph.add_node(ImageLoadNode(path="/fake/init.png"), "control_init_image")

        # Should not raise
        graph.validate_controlnet_inpainting()

    def test_init_image_and_mask_with_controlnet_is_valid(self):
        """Init image, mask, and controlnet (full inpainting setup) should be valid."""
        graph = ImageArtisanZNodeGraph()

        # Add full controlnet with inpainting
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")
        graph.add_node(ImageLoadNode(path="/fake/init.png"), "control_init_image")
        graph.add_node(ImageLoadNode(path="/fake/mask.png", grayscale=True), "control_mask_image")

        # Should not raise
        graph.validate_controlnet_inpainting()

    def test_no_controlnet_inpainting_is_valid(self):
        """No controlnet, init_image, or mask should be valid (normal operation)."""
        graph = ImageArtisanZNodeGraph()

        # Should not raise - no controlnet inpainting nodes present
        graph.validate_controlnet_inpainting()

    def test_only_mask_without_both_init_and_controlnet_raises_error(self):
        """Mask alone (no init, no controlnet, no control_image) should raise error."""
        graph = ImageArtisanZNodeGraph()

        # Add only mask
        graph.add_node(ImageLoadNode(path="/fake/mask.png", grayscale=True), "control_mask_image")

        # Should raise error about missing required inputs
        with pytest.raises(ValueError, match="ControlNet mask requires either a control image or init_image"):
            graph.validate_controlnet_inpainting()

    def test_controlnet_with_no_control_image_and_no_inpainting_raises_error(self):
        """ControlNet without control_image or inpainting should raise ValueError."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet conditioning but no control_image or inpainting
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")

        # Should raise error - controlnet needs either control image or inpainting
        with pytest.raises(ValueError, match="ControlNet requires either a control image or inpainting"):
            graph.validate_controlnet_inpainting()

    def test_controlnet_with_only_init_image_no_control_image_is_valid(self):
        """ControlNet with only init_image (no control_image, no mask) should be valid."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet with only init_image
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")
        graph.add_node(ImageLoadNode(path="/fake/init.png"), "control_init_image")

        # Should not raise - init_image without mask is valid for controlnet
        graph.validate_controlnet_inpainting()

    def test_controlnet_with_control_image_is_valid(self):
        """ControlNet with control_image (standard usage) should be valid."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet with control_image
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")
        graph.add_node(ImageLoadNode(path="/fake/control.png"), "control_image")

        # Should not raise - standard controlnet usage
        graph.validate_controlnet_inpainting()

    def test_controlnet_with_control_image_and_mask_is_valid(self):
        """ControlNet with control_image + mask (without init_image) should be valid (Spatial ControlNet).

        This is Scenario 2 from the refactoring plan: the mask acts as a spatial restriction,
        controlling where ControlNet guidance applies, rather than as an inpainting boundary.
        """
        graph = ImageArtisanZNodeGraph()

        # Add controlnet with control_image and mask (but NO init_image)
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")
        graph.add_node(ImageLoadNode(path="/fake/control.png"), "control_image")
        graph.add_node(ImageLoadNode(path="/fake/mask.png", grayscale=True), "control_mask_image")

        # Should not raise - Spatial ControlNet mode is valid
        graph.validate_controlnet_inpainting()

    def test_controlnet_with_control_image_and_inpainting_is_valid(self):
        """ControlNet with both control_image and inpainting should be valid."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet with control_image and full inpainting
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")
        graph.add_node(ImageLoadNode(path="/fake/control.png"), "control_image")
        graph.add_node(ImageLoadNode(path="/fake/init.png"), "control_init_image")
        graph.add_node(ImageLoadNode(path="/fake/mask.png", grayscale=True), "control_mask_image")

        # Should not raise - full setup is valid
        graph.validate_controlnet_inpainting()
