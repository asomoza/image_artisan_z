"""Tests for edit image management in NodeGraphThread."""

import torch

from iartisanz.modules.generation.graph.new_graph import create_default_flux2_graph
from iartisanz.modules.generation.graph.nodes.flux2_edit_image_encode_node import (
    Flux2EditImageEncodeNode,
)
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread


def _make_thread():
    g = create_default_flux2_graph()
    return NodeGraphThread(None, g, torch.float32, torch.device("cpu"))


def test_add_edit_image_creates_nodes_and_wires():
    """Adding an edit image creates image node, encoder node, and wires to denoise."""
    thread = _make_thread()
    g = thread.node_graph

    thread.add_edit_image(0, "/tmp/edit0.png")

    # Image load node exists
    img_node = g.get_node_by_name("edit_image_0")
    assert img_node is not None
    assert isinstance(img_node, ImageLoadNode)
    assert img_node.path == "/tmp/edit0.png"

    # Encode node exists
    encode_node = g.get_node_by_name("edit_image_encode")
    assert encode_node is not None
    assert isinstance(encode_node, Flux2EditImageEncodeNode)

    # Encode node connected to image node
    assert "image_0" in encode_node.connections
    connected_node, connected_output = encode_node.connections["image_0"][0]
    assert connected_node is img_node
    assert connected_output == "image"

    # Encode node connected to model node
    assert "vae" in encode_node.connections
    assert "vae_scale_factor" in encode_node.connections

    # Denoise node connected to encode node
    denoise = g.get_node_by_name("denoise")
    assert "edit_image_latents" in denoise.connections
    assert "edit_image_latent_ids" in denoise.connections

    lat_node, lat_output = denoise.connections["edit_image_latents"][0]
    assert lat_node is encode_node
    assert lat_output == "image_latents"


def test_add_second_edit_image_reuses_encoder():
    """Adding a second edit image reuses the existing encoder node."""
    thread = _make_thread()
    g = thread.node_graph

    thread.add_edit_image(0, "/tmp/edit0.png")
    encode_node_1 = g.get_node_by_name("edit_image_encode")

    thread.add_edit_image(1, "/tmp/edit1.png")
    encode_node_2 = g.get_node_by_name("edit_image_encode")

    assert encode_node_1 is encode_node_2

    # Both image connections exist
    assert "image_0" in encode_node_2.connections
    assert "image_1" in encode_node_2.connections


def test_update_edit_image_updates_path():
    """Updating an edit image path updates the image load node."""
    thread = _make_thread()
    g = thread.node_graph

    thread.add_edit_image(0, "/tmp/edit0.png")
    thread.update_edit_image(0, "/tmp/edit0_v2.png")

    img_node = g.get_node_by_name("edit_image_0")
    assert img_node.path == "/tmp/edit0_v2.png"


def test_remove_edit_image_cleans_up():
    """Removing an edit image disconnects from encoder and removes node."""
    thread = _make_thread()
    g = thread.node_graph

    thread.add_edit_image(0, "/tmp/edit0.png")
    thread.add_edit_image(1, "/tmp/edit1.png")

    thread.remove_edit_image(0)

    # Image node 0 removed
    assert g.get_node_by_name("edit_image_0") is None

    # Image node 1 still exists
    assert g.get_node_by_name("edit_image_1") is not None

    # Encoder still exists (image 1 remains)
    encode_node = g.get_node_by_name("edit_image_encode")
    assert encode_node is not None
    assert "image_0" not in encode_node.connections
    assert "image_1" in encode_node.connections


def test_remove_last_edit_image_removes_encoder():
    """When all images are removed, the encoder node is also deleted."""
    thread = _make_thread()
    g = thread.node_graph

    thread.add_edit_image(0, "/tmp/edit0.png")
    thread.remove_edit_image(0)

    assert g.get_node_by_name("edit_image_0") is None
    assert g.get_node_by_name("edit_image_encode") is None

    # Denoise should not have edit_image connections
    denoise = g.get_node_by_name("denoise")
    assert "edit_image_latents" not in denoise.connections
    assert "edit_image_latent_ids" not in denoise.connections


def test_remove_all_edit_images():
    """remove_all_edit_images removes all image nodes and the encoder."""
    thread = _make_thread()
    g = thread.node_graph

    thread.add_edit_image(0, "/tmp/edit0.png")
    thread.add_edit_image(1, "/tmp/edit1.png")
    thread.add_edit_image(2, "/tmp/edit2.png")

    thread.remove_all_edit_images()

    for i in range(4):
        assert g.get_node_by_name(f"edit_image_{i}") is None
    assert g.get_node_by_name("edit_image_encode") is None

    denoise = g.get_node_by_name("denoise")
    assert "edit_image_latents" not in denoise.connections
    assert "edit_image_latent_ids" not in denoise.connections


def test_add_duplicate_edit_image_is_noop():
    """Adding an edit image to an already-occupied slot is a no-op."""
    thread = _make_thread()
    g = thread.node_graph

    thread.add_edit_image(0, "/tmp/edit0.png")
    img_node = g.get_node_by_name("edit_image_0")

    thread.add_edit_image(0, "/tmp/edit0_different.png")

    # Same node, same path (no-op)
    assert g.get_node_by_name("edit_image_0") is img_node
    assert img_node.path == "/tmp/edit0.png"
