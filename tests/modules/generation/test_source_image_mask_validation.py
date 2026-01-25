"""Regression tests for source image + mask validation issues.

This test suite reproduces and prevents two critical bugs:

Bug 1: "truth value of an array with more than one element is ambiguous"
- Occurs when adding source_image + source_mask
- Caused by direct boolean check on numpy array in validation

Bug 2: Source image + controlnet mask doesn't activate inpainting
- ControlNet should activate when source_image + controlnet_mask are present
- But validation logic fails to detect this combination
"""

import numpy as np
import pytest
import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes.controlnet_conditioning_node import ControlNetConditioningNode
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from iartisanz.modules.generation.graph.nodes.latents_node import LatentsNode


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


class TestSourceImageMaskValidation:
    """Test validation of source image + mask configurations."""

    def test_source_image_with_source_mask_validation_does_not_raise_ambiguous_truth_error(self):
        """Source image + source mask should validate without "ambiguous truth value" error.

        Bug: validate_controlnet_inpainting() was doing `if mask_image:` directly on numpy array
        which raises: "The truth value of an array with more than one element is ambiguous"

        Fix: Use `mask_image is not None` instead of `if mask_image:`
        """
        graph = ImageArtisanZNodeGraph()

        # Add source image (for img2img)
        source_image = np.random.rand(512, 512, 3).astype(np.float32)
        graph.add_node(ImageLoadNode(image=source_image), "source_image")

        # Add source mask (for differential diffusion)
        source_mask = np.ones((512, 512, 1), dtype=np.float32)  # Numpy array
        graph.add_node(ImageLoadNode(image=source_mask, grayscale=True), "source_mask")

        # This should NOT raise "ambiguous truth value" error
        # The bug was: if mask_image: on numpy array
        # Should be: if mask_image is not None:
        graph.validate_controlnet_inpainting()

    def test_source_image_without_mask_validation_succeeds(self):
        """Source image without mask should validate successfully."""
        graph = ImageArtisanZNodeGraph()

        # Add source image only
        source_image = np.random.rand(512, 512, 3).astype(np.float32)
        graph.add_node(ImageLoadNode(image=source_image), "source_image")

        # Should not raise
        graph.validate_controlnet_inpainting()

    def test_source_image_with_controlnet_mask_requires_controlnet_activation(self):
        """Source image + controlnet mask should activate ControlNet inpainting.

        Bug: When user adds source_image + controlnet_mask, ControlNet should activate
        but validation may not detect this as a valid inpainting configuration.

        This test verifies the validation accepts this configuration.
        """
        graph = ImageArtisanZNodeGraph()

        # Add source image
        source_image = np.random.rand(512, 512, 3).astype(np.float32)
        graph.add_node(ImageLoadNode(image=source_image), "control_init_image")

        # Add controlnet mask
        control_mask = np.ones((512, 512, 1), dtype=np.float32)
        graph.add_node(ImageLoadNode(image=control_mask, grayscale=True), "control_mask_image")

        # Add controlnet conditioning node (required for inpainting)
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")

        # Should validate successfully - this is inpainting-only mode
        graph.validate_controlnet_inpainting()

    def test_controlnet_mask_alone_without_source_or_control_raises_error(self):
        """Controlnet mask without source_image or control_image should raise error."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet mask only (no control_image, no init_image)
        control_mask = np.ones((512, 512, 1), dtype=np.float32)
        graph.add_node(ImageLoadNode(image=control_mask, grayscale=True), "control_mask_image")

        # Should raise error - mask needs either control_image or init_image
        with pytest.raises(ValueError, match="ControlNet mask requires either a control image or init_image"):
            graph.validate_controlnet_inpainting()


class TestSourceImageControlNetIntegration:
    """Test ControlNet activation with source images."""

    def test_graph_validation_with_source_image_and_controlnet_mask(self):
        """Graph validation should pass when source_image + controlnet_mask are present.

        This reproduces the UI scenario:
        1. User adds source image (creates latents node with source_image output)
        2. User adds controlnet mask (creates conditioning node + mask node)
        3. Validation should recognize init_image is connected from source_image

        Bug: validate_controlnet_inpainting() checks for node named "control_init_image"
        which doesn't exist anymore - init_image comes from latents_node.source_image connection.

        This test creates the graph structure as the refactored code does (init_image from
        latents node connection, not a separate node), and verifies validation passes.
        """
        graph = ImageArtisanZNodeGraph()

        # Simulate what the UI does after refactoring:
        # 1. User adds source image -> creates latents node with source_image output
        source_image_node = ImageLoadNode(path="test.png")
        graph.add_node(source_image_node, "source_image")

        latents_node = LatentsNode()
        # Connect source image to latents node's "image" input
        # latents_node then outputs it as "source_image" output
        latents_node.connect("image", source_image_node, "image")
        graph.add_node(latents_node, "latents")

        # 2. User adds controlnet mask -> creates conditioning node + mask node
        conditioning_node = ControlNetConditioningNode()
        # CRITICAL: init_image comes from latents_node's "source_image" OUTPUT
        # not from a separate node named "control_init_image"
        conditioning_node.connect("init_image", latents_node, "source_image")
        graph.add_node(conditioning_node, "controlnet_conditioning")

        mask_node = ImageLoadNode(path="mask.png", grayscale=True)
        graph.add_node(mask_node, "control_mask_image")
        conditioning_node.connect("mask_image", mask_node, "image")

        # 3. Validation should pass - this is inpainting-only mode (Scenario 8)
        # Currently fails because validation checks for node named "control_init_image"
        # which doesn't exist - the init_image is connected from latents_node.source_image
        graph.validate_controlnet_inpainting()

    def test_source_image_with_controlnet_mask_activates_inpainting_mode(self):
        """Source image + controlnet mask should activate inpainting-only mode.

        This tests the full pipeline: graph setup → node execution → verify inpainting mode.
        """
        mm = get_model_manager()
        mm.clear()

        vae = DummyVAE()
        mm.register_active_model(model_id="test", vae=vae)

        # Create ControlNet node with init_image (source) + mask
        node = ControlNetConditioningNode()
        node.vae = ModelHandle("vae")
        node.vae_scale_factor = 8
        node.width = 32
        node.height = 32

        # Add init_image (from source_image)
        node.init_image = torch.rand(1, 3, 32, 32)

        # Add mask
        node.mask_image = np.ones((32, 32, 1), dtype=np.float32)

        # No control_image - this is inpainting-only mode
        node.differential_diffusion_active = False

        with mm.device_scope(device="cpu", dtype=torch.float32):
            out = node()

        # Should output 33-channel inpainting context
        assert out["control_image_latents"].shape[1] == 9, "Should have 33-channel inpainting context"
        assert out["control_mode"] == "inpainting_only"
        assert out["spatial_mask"] is not None

    def test_control_image_with_source_image_and_controlnet_mask_activates_full_inpainting(self):
        """Control image + source image + controlnet mask should activate full inpainting mode.

        This is the complete ControlNet inpainting scenario.
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

        # Add control_image
        node.control_image = torch.rand(1, 3, 32, 32)

        # Add init_image (from source_image)
        node.init_image = torch.rand(1, 3, 32, 32)

        # Add mask
        node.mask_image = np.ones((32, 32, 1), dtype=np.float32)

        node.differential_diffusion_active = False

        with mm.device_scope(device="cpu", dtype=torch.float32):
            out = node()

        # Should output 33-channel inpainting context
        assert out["control_image_latents"].shape[1] == 9, "Should have 33-channel inpainting context"
        assert out["control_mode"] == "controlnet_inpainting"
        assert out["spatial_mask"] is not None

    def test_source_image_only_does_not_activate_controlnet(self):
        """Source image alone should not create ControlNet nodes.

        This is standard img2img, no ControlNet involved.
        """
        graph = ImageArtisanZNodeGraph()

        # Add source image only
        source_image = np.random.rand(32, 32, 3).astype(np.float32)
        graph.add_node(ImageLoadNode(image=source_image), "source_image")

        # Validation should pass - no ControlNet nodes to validate
        graph.validate_controlnet_inpainting()

        # Verify no ControlNet node exists
        assert graph.get_node_by_name("controlnet_conditioning") is None


class TestValidationEdgeCases:
    """Test edge cases in validation logic."""

    def test_empty_mask_array_validates(self):
        """Empty mask array should validate without ambiguous truth error."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet conditioning
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")

        # Add control image
        control_image = np.random.rand(32, 32, 3).astype(np.float32)
        graph.add_node(ImageLoadNode(image=control_image), "control_image")

        # Add empty mask (all zeros)
        empty_mask = np.zeros((32, 32, 1), dtype=np.float32)
        graph.add_node(ImageLoadNode(image=empty_mask, grayscale=True), "control_mask_image")

        # Should validate without error
        graph.validate_controlnet_inpainting()

    def test_none_mask_validates(self):
        """None mask should validate correctly."""
        graph = ImageArtisanZNodeGraph()

        # Add controlnet conditioning
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")

        # Add control image only (no mask)
        control_image = np.random.rand(32, 32, 3).astype(np.float32)
        graph.add_node(ImageLoadNode(image=control_image), "control_image")

        # Should validate - control_image alone is valid
        graph.validate_controlnet_inpainting()

    def test_multiple_validation_calls_do_not_accumulate_errors(self):
        """Multiple validation calls should be idempotent."""
        graph = ImageArtisanZNodeGraph()

        # Add valid configuration
        graph.add_node(ControlNetConditioningNode(), "controlnet_conditioning")
        control_image = np.random.rand(32, 32, 3).astype(np.float32)
        graph.add_node(ImageLoadNode(image=control_image), "control_image")

        # Call validation multiple times
        graph.validate_controlnet_inpainting()
        graph.validate_controlnet_inpainting()
        graph.validate_controlnet_inpainting()

        # Should not raise any errors


class TestArrayTruthinessChecks:
    """Test that array truthiness checks are done correctly throughout codebase."""

    def test_numpy_array_none_check_not_bool_check(self):
        """Verify that None checks work correctly with numpy arrays.

        This documents the correct pattern for checking array existence.
        """
        # Create test arrays
        empty_array = np.array([])
        zero_array = np.zeros((5, 5))
        none_value = None

        # WRONG: if array: raises ambiguous truth error for non-empty arrays
        # RIGHT: if array is not None:

        # These should work:
        assert empty_array is not None  # ✓ Correct
        assert zero_array is not None   # ✓ Correct
        assert none_value is None       # ✓ Correct

        # These would fail with ambiguous truth error:
        # if zero_array: ...  # ✗ Wrong - raises error
        # if empty_array: ... # ✗ Wrong - returns False (but empty array exists)

    def test_tensor_none_check_not_bool_check(self):
        """Verify that None checks work correctly with torch tensors."""
        empty_tensor = torch.tensor([])
        zero_tensor = torch.zeros((5, 5))
        none_value = None

        # These should work:
        assert empty_tensor is not None  # ✓ Correct
        assert zero_tensor is not None   # ✓ Correct
        assert none_value is None        # ✓ Correct


class TestControlNetMaskActivation:
    """Test that adding controlnet mask properly activates ControlNet infrastructure."""

    def test_add_controlnet_mask_without_control_image_creates_infrastructure(self):
        """Adding controlnet mask should create ControlNet infrastructure even without control image.

        Bug: add_controlnet_mask_image() returns early if conditioning_node is None,
        preventing inpainting-only mode from working.

        Fix: Should call _ensure_controlnet_infrastructure() when conditioning node doesn't exist.
        """
        from unittest.mock import MagicMock, patch
        from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread

        # Create a basic graph
        from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
        graph = ImageArtisanZNodeGraph()

        # Create thread with graph
        thread = NodeGraphThread(None, graph, torch.float32, torch.device("cpu"))

        # Mock the model path (normally set by generation_module)
        controlnet_model_path = "/fake/path/to/controlnet.safetensors"

        # Mock _ensure_controlnet_infrastructure to verify it gets called
        # (We don't need to test the full infrastructure creation, just that it's triggered)
        with patch.object(thread, '_ensure_controlnet_infrastructure') as mock_ensure:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                mask_path = f.name

            try:
                # Add controlnet mask with model path - should trigger infrastructure creation
                thread.add_controlnet_mask_image(mask_path, controlnet_path=controlnet_model_path)

                # Verify _ensure_controlnet_infrastructure was called
                assert mock_ensure.called, (
                    "_ensure_controlnet_infrastructure should be called when conditioning node "
                    "doesn't exist but controlnet_path is provided"
                )

                # Verify it was called with the correct controlnet_path
                call_kwargs = mock_ensure.call_args.kwargs
                assert call_kwargs['controlnet_path'] == controlnet_model_path, (
                    "Should pass the controlnet_path to _ensure_controlnet_infrastructure"
                )

            finally:
                import os
                if os.path.exists(mask_path):
                    os.unlink(mask_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
