"""Tests for Flux2EditImageEncodeNode."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from iartisanz.modules.generation.graph.nodes.flux2_edit_image_encode_node import (
    Flux2EditImageEncodeNode,
)


def _make_fake_vae(latent_channels=16):
    """Create a mock VAE that returns deterministic latents."""
    vae = MagicMock()
    vae.device = torch.device("cpu")
    vae.dtype = torch.float32

    class FakeConfig:
        batch_norm_eps = 1e-5

    vae.config = FakeConfig()

    # BN running stats
    bn = MagicMock()
    bn.running_mean = torch.zeros(latent_channels * 4)
    bn.running_var = torch.ones(latent_channels * 4)
    vae.bn = bn

    def encode_fn(x):
        # Produce latents with correct channel count (latent_channels)
        b, _, h, w = x.shape
        lat_h = h // 8  # vae_scale_factor=8
        lat_w = w // 8
        latents = torch.randn(b, latent_channels, lat_h, lat_w, dtype=x.dtype, device=x.device)

        dist = MagicMock()
        dist.mode.return_value = latents

        result = MagicMock()
        result.latent_dist = dist
        return result

    vae.encode = encode_fn
    return vae


def _make_node_with_images(images: list[np.ndarray | None]) -> Flux2EditImageEncodeNode:
    """Create an encode node with mock connections for the given images."""
    node = Flux2EditImageEncodeNode()
    node.device = torch.device("cpu")
    node.dtype = torch.float32

    # Mock required inputs via values dict
    node.values = {}

    # Set up connections manually: REQUIRED_INPUTS come from connected nodes,
    # but for testing we set attributes directly on the node's values.
    # The node uses getattr(self, "image_N") which goes through __getattr__
    # -> get_input_value. We'll mock that by putting values into connected nodes.

    # Instead, directly manipulate the node so __getattr__ returns what we want.
    # We'll create tiny mock nodes and wire them.
    from iartisanz.modules.generation.graph.nodes.node import Node

    # VAE handle
    vae_node = Node()
    vae_node.values = {"vae": "mock_vae_handle", "vae_scale_factor": 8}
    node.dependencies.append(vae_node)
    node.connections["vae"] = [(vae_node, "vae")]
    node.connections["vae_scale_factor"] = [(vae_node, "vae_scale_factor")]

    for i, img in enumerate(images):
        if img is not None:
            img_node = Node()
            img_node.OUTPUTS = ["image"]
            img_node.values = {"image": img}
            node.dependencies.append(img_node)
            node.connections[f"image_{i}"] = [(img_node, "image")]

    return node


def _make_dummy_image(h=32, w=32) -> np.ndarray:
    """Create a dummy RGB uint8 image."""
    return np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)


class TestFlux2EditImageEncodeNode:
    def test_no_images_returns_none(self):
        """No images connected -> outputs None for both."""
        node = _make_node_with_images([None, None, None, None])
        with patch("iartisanz.modules.generation.graph.nodes.flux2_edit_image_encode_node.get_model_manager"):
            result = node()

        assert result["image_latents"] is None
        assert result["image_latent_ids"] is None

    def test_single_image_encodes_and_packs(self):
        """Single image -> VAE encode, patchify, BN normalize, pack."""
        img = _make_dummy_image(32, 32)
        node = _make_node_with_images([img, None, None, None])

        fake_vae = _make_fake_vae(latent_channels=16)
        mm = MagicMock()
        mm.resolve.return_value = fake_vae

        with patch(
            "iartisanz.modules.generation.graph.nodes.flux2_edit_image_encode_node.get_model_manager",
            return_value=mm,
        ):
            result = node()

        latents = result["image_latents"]
        latent_ids = result["image_latent_ids"]

        assert latents is not None
        assert latent_ids is not None

        # Shape: (1, seq_len, C) where C = latent_channels * 4 = 64
        assert latents.dim() == 3
        assert latents.shape[0] == 1
        assert latents.shape[2] == 64  # 16 * 4

        # IDs: (1, seq_len, 4)
        assert latent_ids.dim() == 3
        assert latent_ids.shape[0] == 1
        assert latent_ids.shape[2] == 4
        assert latents.shape[1] == latent_ids.shape[1]

    def test_multiple_images_concatenates(self):
        """Multiple images -> tokens concatenated along sequence dim."""
        img0 = _make_dummy_image(32, 32)
        img1 = _make_dummy_image(32, 32)
        node = _make_node_with_images([img0, img1, None, None])

        fake_vae = _make_fake_vae(latent_channels=16)
        mm = MagicMock()
        mm.resolve.return_value = fake_vae

        with patch(
            "iartisanz.modules.generation.graph.nodes.flux2_edit_image_encode_node.get_model_manager",
            return_value=mm,
        ):
            result = node()

        latents = result["image_latents"]
        latent_ids = result["image_latent_ids"]

        # With 32x32 input, vae_scale_factor=8: latent spatial = 4x4
        # Patchified: 2x2, so seq_len per image = 2*2 = 4
        # Two images -> total seq_len = 8
        single_seq_len = 4  # (4/2) * (4/2)
        assert latents.shape[1] == single_seq_len * 2
        assert latent_ids.shape[1] == single_seq_len * 2

    def test_position_ids_time_offsets(self):
        """Verify T coordinates are 10, 20, 30 for sequential images."""
        img0 = _make_dummy_image(32, 32)
        img1 = _make_dummy_image(32, 32)
        img2 = _make_dummy_image(32, 32)
        node = _make_node_with_images([img0, img1, img2, None])

        fake_vae = _make_fake_vae(latent_channels=16)
        mm = MagicMock()
        mm.resolve.return_value = fake_vae

        with patch(
            "iartisanz.modules.generation.graph.nodes.flux2_edit_image_encode_node.get_model_manager",
            return_value=mm,
        ):
            result = node()

        latent_ids = result["image_latent_ids"]
        # Each image produces 4 tokens (2x2 spatial)
        tokens_per_image = 4

        # First image: T=10
        t_values_0 = latent_ids[0, :tokens_per_image, 0]
        assert (t_values_0 == 10).all()

        # Second image: T=20
        t_values_1 = latent_ids[0, tokens_per_image : tokens_per_image * 2, 0]
        assert (t_values_1 == 20).all()

        # Third image: T=30
        t_values_2 = latent_ids[0, tokens_per_image * 2 : tokens_per_image * 3, 0]
        assert (t_values_2 == 30).all()

    def test_patchify_roundtrip(self):
        """Patchify -> unpatchify returns original shape."""
        from iartisanz.modules.generation.graph.nodes.flux2_latents_decoder_node import (
            Flux2LatentsDecoderNode,
        )

        original = torch.randn(1, 16, 8, 8)
        patchified = Flux2EditImageEncodeNode._patchify_latents(original)

        assert patchified.shape == (1, 64, 4, 4)

        restored = Flux2LatentsDecoderNode._unpatchify_latents(patchified)
        assert restored.shape == original.shape
        assert torch.allclose(original, restored)
