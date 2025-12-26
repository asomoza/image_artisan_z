from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes.denoise_node import DenoiseNode
from iartisanz.modules.generation.graph.nodes.image_send_node import ImageSendNode
from iartisanz.modules.generation.graph.nodes.latents_decoder_node import LatentsDecoderNode
from iartisanz.modules.generation.graph.nodes.latents_node import LatentsNode
from iartisanz.modules.generation.graph.nodes.number_node import NumberNode
from iartisanz.modules.generation.graph.nodes.prompt_encode_node import PromptEncoderNode
from iartisanz.modules.generation.graph.nodes.scheduler_node import SchedulerNode
from iartisanz.modules.generation.graph.nodes.text_node import TextNode
from iartisanz.modules.generation.graph.nodes.zimage_model_node import ZImageModelNode


def create_default_graph():
    node_graph = ImageArtisanZNodeGraph()

    seed = NumberNode()
    node_graph.add_node(seed, "seed")

    image_width = NumberNode()
    node_graph.add_node(image_width, "image_width")

    image_height = NumberNode()
    node_graph.add_node(image_height, "image_height")

    num_inference_steps = NumberNode(number=9)
    node_graph.add_node(num_inference_steps, "num_inference_steps")

    guidance_scale = NumberNode(number=1.0)
    node_graph.add_node(guidance_scale, "guidance_scale")

    positive_prompt = TextNode()
    node_graph.add_node(positive_prompt, "positive_prompt")

    negative_prompt = TextNode()
    node_graph.add_node(negative_prompt, "negative_prompt")

    zimage_models = ZImageModelNode()
    node_graph.add_node(zimage_models, "model")

    scheduler = SchedulerNode(scheduler_data_object=SchedulerDataObject())
    node_graph.add_node(scheduler, "scheduler")

    prompts_encoder = PromptEncoderNode()
    prompts_encoder.connect("tokenizer", zimage_models, "tokenizer")
    prompts_encoder.connect("text_encoder", zimage_models, "text_encoder")
    prompts_encoder.connect("positive_prompt", positive_prompt, "value")
    prompts_encoder.connect("negative_prompt", negative_prompt, "value")
    node_graph.add_node(prompts_encoder, "prompts_encoder")

    latents = LatentsNode()
    latents.connect("seed", seed, "value")
    latents.connect("num_channels_latents", zimage_models, "num_channels_latents")
    latents.connect("width", image_width, "value")
    latents.connect("height", image_height, "value")
    latents.connect("vae_scale_factor", zimage_models, "vae_scale_factor")
    node_graph.add_node(latents, "latents")

    denoise = DenoiseNode()
    denoise.connect("transformer", zimage_models, "transformer")
    denoise.connect("num_inference_steps", num_inference_steps, "value")
    denoise.connect("latents", latents, "latents")
    denoise.connect("scheduler", scheduler, "scheduler")
    denoise.connect("prompt_embeds", prompts_encoder, "prompt_embeds")
    denoise.connect("negative_prompt_embeds", prompts_encoder, "negative_prompt_embeds")
    denoise.connect("guidance_scale", guidance_scale, "value")
    node_graph.add_node(denoise, "denoise")

    latents_decoder = LatentsDecoderNode()
    latents_decoder.connect("vae", zimage_models, "vae")
    latents_decoder.connect("latents", denoise, "latents")
    node_graph.add_node(latents_decoder, "decoder")

    image_send = ImageSendNode()
    image_send.connect("image", latents_decoder, "image")
    node_graph.add_node(image_send, "image_send")

    return node_graph
