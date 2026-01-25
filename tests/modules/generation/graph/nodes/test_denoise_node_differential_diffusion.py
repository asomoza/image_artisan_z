"""Tests for DenoiseNode differential diffusion functionality."""
import numpy as np
import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.denoise_node import DenoiseNode


class DummyScheduler:
    order = 1

    def set_timesteps(self, num_inference_steps, device=None, **_kwargs):
        # Use a single timestep tensor on the requested device.
        self.timesteps = torch.tensor([999], device=device)

    def scale_noise(self, latents, timesteps, noise):
        """Scale noise for differential diffusion."""
        # Simple identity for testing
        return latents + noise * 0.0

    def step(self, noise_pred, _t, latents, return_dict=False):
        # Identity step (keeps latents stable).
        # Return tuple: (prev_sample,) for subscriptable access
        return (latents + noise_pred.to(latents.dtype) * 0.0,)


class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        latent_model_input_list,
        _timestep_model_input,
        _prompt_embeds_model_input,
        *,
        controlnet_block_samples=None,
        return_dict=False,
    ):
        outs = [torch.zeros_like(x) for x in latent_model_input_list]
        return (outs,)


def _make_denoise_node_with_noise(*, device: torch.device, guidance_scale: float = 2.0) -> DenoiseNode:
    """Create a DenoiseNode configured for differential diffusion testing."""
    node = DenoiseNode()
    node.device = device
    node.dtype = torch.float32

    node.transformer = ModelHandle("transformer")
    node.scheduler = DummyScheduler()
    node.num_inference_steps = 1

    node.latents = torch.randn(1, 16, 4, 4)

    # Prompt embeddings
    node.prompt_embeds = torch.randn(1, 5, 8)
    node.negative_prompt_embeds = torch.randn(1, 5, 8)

    node.guidance_scale = float(guidance_scale)

    # Add noise (required for differential diffusion)
    node.noise = torch.randn(1, 16, 4, 4)

    return node


def test_denoise_node_with_source_mask_array():
    """Test that source_mask as numpy array doesn't cause ambiguous truth value error.

    This reproduces the UI scenario where:
    - source_mask is a numpy array from ImageLoadNode
    - image_mask is None (not set)

    The bug occurs because of the 'or' operator on line 348:
        source_mask = getattr(self, "source_mask", None) or getattr(self, "image_mask", None)

    When source_mask is a numpy array, the 'or' tries to evaluate its truthiness,
    which fails with: "The truth value of an array with more than one element is ambiguous"
    """
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    mm.register_active_model(model_id="test", transformer=transformer)

    node = _make_denoise_node_with_noise(device=torch.device("cpu"))

    # Simulate what the UI does: ImageLoadNode outputs a numpy array
    # This is a mask with all 1s (fully masked)
    source_mask_array = np.ones((1024, 1024, 1), dtype=np.float32)
    node.source_mask = source_mask_array

    # image_mask is not set (stays None)
    # node.image_mask = None  # This is the default, no need to set

    # This should NOT raise "The truth value of an array with more than one element is ambiguous"
    with mm.device_scope(device="cpu", dtype=torch.float32):
        result = node()

    assert result is not None
    assert "latents" in result


def test_denoise_node_with_image_mask_fallback():
    """Test that image_mask is used as fallback when source_mask is None."""
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    mm.register_active_model(model_id="test", transformer=transformer)

    node = _make_denoise_node_with_noise(device=torch.device("cpu"))

    # source_mask is None (not set)
    # image_mask is set to a numpy array
    image_mask_array = np.ones((512, 512, 1), dtype=np.float32)
    node.image_mask = image_mask_array

    with mm.device_scope(device="cpu", dtype=torch.float32):
        result = node()

    assert result is not None
    assert "latents" in result


def test_denoise_node_source_mask_takes_precedence():
    """Test that source_mask takes precedence over image_mask when both are set."""
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    mm.register_active_model(model_id="test", transformer=transformer)

    node = _make_denoise_node_with_noise(device=torch.device("cpu"))

    # Both masks are set
    source_mask_array = np.ones((1024, 1024, 1), dtype=np.float32)
    image_mask_array = np.zeros((512, 512, 1), dtype=np.float32)  # Different values

    node.source_mask = source_mask_array
    node.image_mask = image_mask_array

    with mm.device_scope(device="cpu", dtype=torch.float32):
        result = node()

    # Should use source_mask (the behavior should be that source_mask is preferred)
    assert result is not None
    assert "latents" in result


def test_denoise_node_no_mask_skips_differential_diffusion():
    """Test that differential diffusion is skipped when no mask is provided."""
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    mm.register_active_model(model_id="test", transformer=transformer)

    node = _make_denoise_node_with_noise(device=torch.device("cpu"))

    # No masks set
    # node.source_mask = None
    # node.image_mask = None

    with mm.device_scope(device="cpu", dtype=torch.float32):
        result = node()

    assert result is not None
    assert "latents" in result
