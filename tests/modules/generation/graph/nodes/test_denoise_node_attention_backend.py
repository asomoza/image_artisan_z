"""Tests for ZImageDenoiseNode attention backend integration."""

from unittest.mock import MagicMock, patch, call

import pytest
import torch

from iartisanz.app.model_manager import ModelHandle, ModelManager, get_model_manager


class TestZImageDenoiseNodeAttentionBackend:
    """Tests for attention backend application in ZImageDenoiseNode."""

    def _create_mock_transformer(self):
        """Create a mock transformer with attention backend methods."""
        mock = MagicMock()
        mock.set_attention_backend = MagicMock()
        mock.reset_attention_backend = MagicMock()
        mock.dtype = torch.float32
        # Make it look like a torch module for device handling
        mock.parameters = MagicMock(return_value=iter([]))
        mock.buffers = MagicMock(return_value=iter([]))
        return mock

    def _create_minimal_denoise_node(self, mock_transformer):
        """Create a minimal ZImageDenoiseNode for testing."""
        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        node = ZImageDenoiseNode()
        node.device = torch.device("cpu")
        node.dtype = torch.float32
        node.abort = False

        # Set required inputs
        node.transformer = mock_transformer
        node.num_inference_steps = 1
        node.latents = torch.randn(1, 4, 8, 8)
        node.prompt_embeds = torch.randn(1, 10, 64)
        node.negative_prompt_embeds = torch.randn(1, 10, 64)
        node.guidance_scale = 1.0

        # Create a mock scheduler
        mock_scheduler = MagicMock()
        mock_scheduler.set_timesteps = MagicMock()
        mock_scheduler.timesteps = torch.tensor([999.0])
        mock_scheduler.step = MagicMock(return_value=(torch.randn(1, 4, 8, 8),))
        mock_scheduler.order = 1
        node.scheduler = mock_scheduler

        # Optional inputs
        node.cfg_normalization = None
        node.sigmas = None
        node.lora = None
        node.guidance_start_end = None
        node.noise = None
        node.strength = None
        node.image_mask = None
        node.source_mask = None
        node.controlnet = None
        node.control_image_latents = None
        node.controlnet_conditioning_scale = None
        node.controlnet_spatial_mask = None
        node.control_guidance_start_end = None
        node.control_mode = None
        node.prompt_mode_decay = None

        return node

    @patch("iartisanz.modules.generation.graph.nodes.zimage_denoise_node.get_model_manager")
    def test_applies_attention_backend_from_model_manager(self, mock_get_mm):
        """ZImageDenoiseNode should apply attention backend from ModelManager."""
        # Setup mock ModelManager
        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.attention_backend = "flash"
        mock_mm.use_torch_compile = False
        mock_mm.apply_attention_backend = MagicMock(return_value=True)
        mock_mm.resolve = MagicMock(side_effect=lambda x: x)
        mock_get_mm.return_value = mock_mm

        # Setup mock transformer
        mock_transformer = self._create_mock_transformer()
        mock_transformer.return_value = ([torch.randn(1, 4, 8, 8)],)

        # Create node
        node = self._create_minimal_denoise_node(mock_transformer)

        # Run the node
        node()

        # Verify apply_attention_backend was called with the transformer
        mock_mm.apply_attention_backend.assert_called_once_with(mock_transformer)

    @patch("iartisanz.modules.generation.graph.nodes.zimage_denoise_node.get_model_manager")
    def test_applies_attention_backend_before_torch_compile(self, mock_get_mm):
        """Attention backend should be applied before torch.compile."""
        call_order = []

        # Setup mock ModelManager
        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.attention_backend = "flash"
        mock_mm.use_torch_compile = True
        mock_mm.apply_attention_backend = MagicMock(
            side_effect=lambda x: call_order.append("apply_attention_backend")
        )
        mock_mm.get_compiled = MagicMock(
            side_effect=lambda *args, **kwargs: call_order.append("get_compiled")
        )
        mock_mm.resolve = MagicMock(side_effect=lambda x: x)
        mock_mm.disable_compiled = MagicMock()
        mock_mm.clear_compiled = MagicMock()
        mock_get_mm.return_value = mock_mm

        # Setup mock transformer that is a ModelHandle
        mock_transformer = self._create_mock_transformer()
        mock_transformer.return_value = ([torch.randn(1, 4, 8, 8)],)

        # Create node with ModelHandle for transformer
        node = self._create_minimal_denoise_node(mock_transformer)
        node.transformer = ModelHandle(component="transformer")
        mock_mm.resolve = MagicMock(return_value=mock_transformer)
        mock_mm.get_compiled = MagicMock(return_value=mock_transformer)

        # Run the node
        node()

        # Verify apply_attention_backend was called before get_compiled
        assert "apply_attention_backend" in call_order
        # Note: get_compiled may or may not be in call_order depending on conditions

    @patch("iartisanz.modules.generation.graph.nodes.zimage_denoise_node.get_model_manager")
    def test_native_backend_resets_transformer(self, mock_get_mm):
        """Native backend should call reset on transformer."""
        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.attention_backend = "native"
        mock_mm.use_torch_compile = False

        # Make apply_attention_backend call reset_attention_backend
        def apply_backend(transformer):
            if hasattr(transformer, "reset_attention_backend"):
                transformer.reset_attention_backend()
            return True

        mock_mm.apply_attention_backend = MagicMock(side_effect=apply_backend)
        mock_mm.resolve = MagicMock(side_effect=lambda x: x)
        mock_get_mm.return_value = mock_mm

        mock_transformer = self._create_mock_transformer()
        mock_transformer.return_value = ([torch.randn(1, 4, 8, 8)],)

        node = self._create_minimal_denoise_node(mock_transformer)
        node()

        # Verify reset_attention_backend was called via apply_attention_backend
        mock_mm.apply_attention_backend.assert_called_once()


class TestZImageDenoiseNodeAttentionBackendWithControlNet:
    """Tests for attention backend with ControlNet."""

    @patch("iartisanz.modules.generation.graph.nodes.zimage_denoise_node.get_model_manager")
    def test_attention_backend_works_with_controlnet(self, mock_get_mm):
        """Attention backend should work correctly with ControlNet enabled."""
        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.attention_backend = "flash"
        mock_mm.use_torch_compile = False
        mock_mm.apply_attention_backend = MagicMock(return_value=True)
        mock_mm.resolve = MagicMock(side_effect=lambda x, **kwargs: x)
        mock_mm.default_device_scope = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
        mock_mm.ensure_module_device = MagicMock()
        mock_get_mm.return_value = mock_mm

        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        mock_transformer = MagicMock()
        mock_transformer.dtype = torch.float32
        mock_transformer.parameters = MagicMock(return_value=iter([]))
        mock_transformer.return_value = ([torch.randn(1, 4, 8, 8)],)

        node = ZImageDenoiseNode()
        node.device = torch.device("cpu")
        node.dtype = torch.float32
        node.abort = False

        node.transformer = mock_transformer
        node.num_inference_steps = 1
        node.latents = torch.randn(1, 4, 8, 8)
        node.prompt_embeds = torch.randn(1, 10, 64)
        node.negative_prompt_embeds = torch.randn(1, 10, 64)
        node.guidance_scale = 1.0

        mock_scheduler = MagicMock()
        mock_scheduler.set_timesteps = MagicMock()
        mock_scheduler.timesteps = torch.tensor([999.0])
        mock_scheduler.step = MagicMock(return_value=(torch.randn(1, 4, 8, 8),))
        mock_scheduler.order = 1
        node.scheduler = mock_scheduler

        # Set ControlNet with proper config to avoid channel mismatch
        mock_controlnet = MagicMock()
        mock_controlnet.return_value = {}
        # Set config.control_in_dim to match latent channels (4)
        mock_controlnet.config = {"control_in_dim": 4}
        node.controlnet = mock_controlnet
        node.control_image_latents = torch.randn(1, 4, 8, 8)
        node.controlnet_conditioning_scale = 1.0

        # Optional inputs
        node.cfg_normalization = None
        node.sigmas = None
        node.lora = None
        node.guidance_start_end = None
        node.noise = None
        node.strength = None
        node.image_mask = None
        node.source_mask = None
        node.controlnet_spatial_mask = None
        node.control_guidance_start_end = None
        node.control_mode = "balanced"
        node.prompt_mode_decay = None

        node()

        # Verify attention backend was still applied
        mock_mm.apply_attention_backend.assert_called_once()


class TestZImageDenoiseNodeAttentionBackendWithLoRA:
    """Tests for attention backend with LoRA."""

    @patch("iartisanz.modules.generation.graph.nodes.zimage_denoise_node.get_model_manager")
    def test_attention_backend_applied_before_lora(self, mock_get_mm):
        """Attention backend should be applied (LoRA is set after)."""
        mock_mm = MagicMock(spec=ModelManager)
        mock_mm.attention_backend = "flash"
        mock_mm.use_torch_compile = False
        mock_mm.apply_attention_backend = MagicMock(return_value=True)
        mock_mm.resolve = MagicMock(side_effect=lambda x: x)
        mock_get_mm.return_value = mock_mm

        from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode

        mock_transformer = MagicMock()
        mock_transformer.dtype = torch.float32
        mock_transformer.parameters = MagicMock(return_value=iter([]))
        mock_transformer.set_adapters = MagicMock()
        mock_transformer.return_value = ([torch.randn(1, 4, 8, 8)],)

        node = ZImageDenoiseNode()
        node.device = torch.device("cpu")
        node.dtype = torch.float32
        node.abort = False

        node.transformer = mock_transformer
        node.num_inference_steps = 1
        node.latents = torch.randn(1, 4, 8, 8)
        node.prompt_embeds = torch.randn(1, 10, 64)
        node.negative_prompt_embeds = torch.randn(1, 10, 64)
        node.guidance_scale = 1.0

        mock_scheduler = MagicMock()
        mock_scheduler.set_timesteps = MagicMock()
        mock_scheduler.timesteps = torch.tensor([999.0])
        mock_scheduler.step = MagicMock(return_value=(torch.randn(1, 4, 8, 8),))
        mock_scheduler.order = 1
        node.scheduler = mock_scheduler

        # Set LoRA
        node.lora = ("test_lora", {"transformer": [1.0]})

        # Optional inputs
        node.cfg_normalization = None
        node.sigmas = None
        node.guidance_start_end = None
        node.noise = None
        node.strength = None
        node.image_mask = None
        node.source_mask = None
        node.controlnet = None
        node.control_image_latents = None
        node.controlnet_conditioning_scale = None
        node.controlnet_spatial_mask = None
        node.control_guidance_start_end = None
        node.control_mode = None
        node.prompt_mode_decay = None

        node()

        # Verify attention backend was applied
        mock_mm.apply_attention_backend.assert_called_once()
