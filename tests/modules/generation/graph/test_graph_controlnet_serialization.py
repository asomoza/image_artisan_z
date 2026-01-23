import json

from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.new_graph import create_default_graph
from iartisanz.modules.generation.graph.nodes import NODE_CLASSES
from iartisanz.modules.generation.graph.nodes.choice_node import ChoiceNode
from iartisanz.modules.generation.graph.nodes.controlnet_conditioning_node import ControlNetConditioningNode
from iartisanz.modules.generation.graph.nodes.controlnet_model_node import ControlNetModelNode
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from iartisanz.modules.generation.graph.nodes.number_node import NumberNode


def test_graph_json_roundtrip_with_optional_controlnet_nodes():
    g = create_default_graph()

    controlnet_model = ControlNetModelNode(path="/tmp/fake_controlnet.safetensors")
    control_image = ImageLoadNode(path="/tmp/fake_control_image.png")
    init_image = ImageLoadNode(path="/tmp/fake_init_image.png")
    mask_image = ImageLoadNode(path="/tmp/fake_mask.png", grayscale=True)
    conditioning_scale = NumberNode(number=0.8)
    control_mode = ChoiceNode(value="prompt", choices=["balanced", "prompt", "controlnet"], default="balanced")
    prompt_decay = NumberNode(number=0.5)
    conditioning = ControlNetConditioningNode()

    g.add_node(controlnet_model, name="controlnet_model")
    g.add_node(control_image, name="control_image")
    g.add_node(init_image, name="control_init_image")
    g.add_node(mask_image, name="control_mask_image")
    g.add_node(conditioning_scale, name="controlnet_conditioning_scale")
    g.add_node(control_mode, name="controlnet_control_mode")
    g.add_node(prompt_decay, name="controlnet_prompt_decay")
    g.add_node(conditioning, name="controlnet_conditioning")

    model = g.get_node_by_name("model")
    width = g.get_node_by_name("image_width")
    height = g.get_node_by_name("image_height")
    denoise = g.get_node_by_name("denoise")

    controlnet_model.connect("transformer", model, "transformer")

    conditioning.connect("vae", model, "vae")
    conditioning.connect("vae_scale_factor", model, "vae_scale_factor")
    conditioning.connect("control_image", control_image, "image")
    conditioning.connect("init_image", init_image, "image")
    conditioning.connect("mask_image", mask_image, "image")
    conditioning.connect("width", width, "value")
    conditioning.connect("height", height, "value")

    denoise.connect("controlnet", controlnet_model, "controlnet")
    denoise.connect("control_image_latents", conditioning, "control_image_latents")
    denoise.connect("controlnet_conditioning_scale", conditioning_scale, "value")
    denoise.connect("control_mode", control_mode, "value")
    denoise.connect("prompt_mode_decay", prompt_decay, "value")

    payload = g.to_json()
    parsed = json.loads(payload)

    assert any(n.get("name") == "controlnet_model" for n in parsed.get("nodes", []))
    assert any(n.get("name") == "controlnet_conditioning" for n in parsed.get("nodes", []))
    assert any(n.get("name") == "controlnet_control_mode" for n in parsed.get("nodes", []))
    assert any(n.get("name") == "controlnet_prompt_decay" for n in parsed.get("nodes", []))

    g2 = ImageArtisanZNodeGraph()
    g2.from_json(payload, node_classes=NODE_CLASSES, callbacks=None)

    denoise2 = g2.get_node_by_name("denoise")
    assert denoise2 is not None

    # Ensure the wiring survives roundtrip.
    assert "controlnet" in denoise2.connections
    assert "control_image_latents" in denoise2.connections
    assert "controlnet_conditioning_scale" in denoise2.connections
    assert "control_mode" in denoise2.connections
    assert "prompt_mode_decay" in denoise2.connections

    conditioning2 = g2.get_node_by_name("controlnet_conditioning")
    assert conditioning2 is not None
    assert "mask_image" in conditioning2.connections
    assert "init_image" in conditioning2.connections
