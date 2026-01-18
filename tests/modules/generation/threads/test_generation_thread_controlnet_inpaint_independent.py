import torch

from iartisanz.modules.generation.graph.new_graph import create_default_graph
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread


def test_controlnet_inpaint_inputs_are_independent_from_source_image_nodes():
    g = create_default_graph()
    thread = NodeGraphThread(None, g, torch.float32, torch.device("cpu"))

    thread.add_controlnet(
        controlnet_path="/tmp/fake_controlnet.safetensors",
        control_image_path="/tmp/fake_control_image.png",
        conditioning_scale=0.75,
    )

    # Add regular source image + differential diffusion mask.
    thread.add_source_image("/tmp/fake_source.png", strength=0.5)
    thread.add_source_image_mask("/tmp/fake_source_mask.png")

    conditioning = g.get_node_by_name("controlnet_conditioning")
    assert conditioning is not None

    # Ensure ControlNet conditioning does NOT automatically reuse these.
    assert "init_image" not in conditioning.connections
    assert "mask_image" not in conditioning.connections

    # Now add dedicated ControlNet inpaint inputs.
    thread.add_controlnet_init_image("/tmp/fake_init.png")
    thread.add_controlnet_mask_image("/tmp/fake_control_mask.png")

    assert "init_image" in conditioning.connections
    assert "mask_image" in conditioning.connections

    init_node = g.get_node_by_name("control_init_image")
    mask_node = g.get_node_by_name("control_mask_image")
    assert init_node is not None
    assert mask_node is not None

    # Verify they are distinct nodes from the img2img/diffusion ones.
    assert g.get_node_by_name("source_image") is not None
    assert g.get_node_by_name("source_image_mask") is not None
    assert init_node is not g.get_node_by_name("source_image")
    assert mask_node is not g.get_node_by_name("source_image_mask")
