import torch

from iartisanz.modules.generation.graph.new_graph import create_default_graph
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread


def test_add_controlnet_wires_mode_and_decay_nodes():
    g = create_default_graph()
    thread = NodeGraphThread(None, g, torch.float32, torch.device("cpu"))

    thread.add_controlnet(
        controlnet_path="/tmp/fake_controlnet.safetensors",
        control_image_path="/tmp/fake_control_image.png",
        conditioning_scale=0.75,
        control_mode="prompt",
        prompt_decay=0.4,
    )

    assert g.get_node_by_name("controlnet_control_mode") is not None
    assert g.get_node_by_name("controlnet_prompt_decay") is not None

    denoise = g.get_node_by_name("denoise")
    assert denoise is not None

    assert "control_mode" in denoise.connections
    assert "prompt_mode_decay" in denoise.connections


def test_remove_controlnet_deletes_mode_and_decay_nodes_and_disconnects():
    g = create_default_graph()
    thread = NodeGraphThread(None, g, torch.float32, torch.device("cpu"))

    thread.add_controlnet(
        controlnet_path="/tmp/fake_controlnet.safetensors",
        control_image_path="/tmp/fake_control_image.png",
        conditioning_scale=0.75,
        control_mode="balanced",
        prompt_decay=0.825,
    )

    denoise = g.get_node_by_name("denoise")
    assert denoise is not None

    assert "control_mode" in denoise.connections
    assert "prompt_mode_decay" in denoise.connections

    thread.remove_controlnet()

    assert g.get_node_by_name("controlnet_control_mode") is None
    assert g.get_node_by_name("controlnet_prompt_decay") is None

    assert "control_mode" not in denoise.connections
    assert "prompt_mode_decay" not in denoise.connections
