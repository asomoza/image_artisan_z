"""Test ControlNet uses source_image from LatentsNode in new architecture."""

import torch

from iartisanz.modules.generation.graph.new_graph import create_default_graph
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread


def test_controlnet_uses_source_image_from_latents_node():
    """Test that ControlNet gets init_image from LatentsNode's source_image output."""
    g = create_default_graph()
    thread = NodeGraphThread(None, g, torch.float32, torch.device("cpu"))

    thread.add_controlnet(
        controlnet_path="/tmp/fake_controlnet.safetensors",
        control_image_path="/tmp/fake_control_image.png",
        conditioning_scale=0.75,
    )

    # Add regular source image
    thread.add_source_image("/tmp/fake_source.png", strength=0.5)

    conditioning = g.get_node_by_name("controlnet_conditioning")
    latents_node = g.get_node_by_name("latents")
    assert conditioning is not None
    assert latents_node is not None

    # ControlNet should get init_image from LatentsNode's source_image output
    assert "init_image" in conditioning.connections
    init_connections = conditioning.connections["init_image"]
    assert len(init_connections) == 1
    init_node, init_output = init_connections[0]
    assert init_node == latents_node
    assert init_output == "source_image"


def test_controlnet_mask_and_differential_diffusion_are_independent():
    """Test that ControlNet mask and differential diffusion source mask are separate."""
    g = create_default_graph()
    thread = NodeGraphThread(None, g, torch.float32, torch.device("cpu"))

    thread.add_controlnet(
        controlnet_path="/tmp/fake_controlnet.safetensors",
        control_image_path="/tmp/fake_control_image.png",
        conditioning_scale=0.75,
    )

    # Add regular source image + differential diffusion mask
    thread.add_source_image("/tmp/fake_source.png", strength=0.5)
    thread.add_source_image_mask("/tmp/fake_source_mask.png")

    # Add ControlNet spatial mask
    thread.add_controlnet_mask_image("/tmp/fake_control_mask.png")

    conditioning = g.get_node_by_name("controlnet_conditioning")
    denoise_node = g.get_node_by_name("denoise")
    assert conditioning is not None
    assert denoise_node is not None

    # ControlNet should have mask_image connected
    assert "mask_image" in conditioning.connections
    control_mask_node = g.get_node_by_name("control_mask_image")
    assert control_mask_node is not None

    # DenoiseNode should have source_mask for differential diffusion
    assert "source_mask" in denoise_node.connections
    source_mask_node = g.get_node_by_name("source_image_mask")
    assert source_mask_node is not None

    # Verify they are distinct nodes
    assert control_mask_node is not source_mask_node

    # DenoiseNode should also have controlnet_spatial_mask connection
    assert "controlnet_spatial_mask" in denoise_node.connections


def test_differential_diffusion_flag_connected_to_controlnet():
    """Test that differential_diffusion_active flag is connected to ControlNet."""
    g = create_default_graph()
    thread = NodeGraphThread(None, g, torch.float32, torch.device("cpu"))

    thread.add_controlnet(
        controlnet_path="/tmp/fake_controlnet.safetensors",
        control_image_path="/tmp/fake_control_image.png",
        conditioning_scale=0.75,
    )

    # Add source image + mask to activate differential diffusion
    thread.add_source_image("/tmp/fake_source.png", strength=0.5)
    thread.add_source_image_mask("/tmp/fake_source_mask.png")

    conditioning = g.get_node_by_name("controlnet_conditioning")
    diff_diff_flag = g.get_node_by_name("differential_diffusion_active")

    assert conditioning is not None
    assert diff_diff_flag is not None

    # ControlNet should have differential_diffusion_active input connected
    assert "differential_diffusion_active" in conditioning.connections
    flag_connections = conditioning.connections["differential_diffusion_active"]
    assert len(flag_connections) == 1
    flag_node, flag_output = flag_connections[0]
    assert flag_node == diff_diff_flag
    assert flag_output == "value"

    # Flag should be True when source mask is added
    from iartisanz.modules.generation.graph.nodes.boolean_node import BooleanNode
    assert isinstance(diff_diff_flag, BooleanNode)
    assert diff_diff_flag.value is True
