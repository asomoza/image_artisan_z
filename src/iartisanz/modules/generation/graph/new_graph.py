from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes.flux2_denoise_node import Flux2DenoiseNode
from iartisanz.modules.generation.graph.nodes.flux2_latents_decoder_node import Flux2LatentsDecoderNode
from iartisanz.modules.generation.graph.nodes.flux2_latents_node import Flux2LatentsNode
from iartisanz.modules.generation.graph.nodes.flux2_model_node import Flux2ModelNode
from iartisanz.modules.generation.graph.nodes.flux2_prompt_encode_node import Flux2PromptEncoderNode
from iartisanz.modules.generation.graph.nodes.image_send_node import ImageSendNode
from iartisanz.modules.generation.graph.nodes.number_node import NumberNode
from iartisanz.modules.generation.graph.nodes.number_range_node import NumberRangeNode
from iartisanz.modules.generation.graph.nodes.scheduler_node import SchedulerNode
from iartisanz.modules.generation.graph.nodes.text_node import TextNode
from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode
from iartisanz.modules.generation.graph.nodes.zimage_latents_decoder_node import ZImageLatentsDecoderNode
from iartisanz.modules.generation.graph.nodes.zimage_latents_node import ZImageLatentsNode
from iartisanz.modules.generation.graph.nodes.zimage_model_node import ZImageModelNode
from iartisanz.modules.generation.graph.nodes.zimage_prompt_encode_node import ZImagePromptEncoderNode


def create_default_graph(model_type: int = 1):
    """Create the default node graph for Z-Image text-to-image generation.

    Args:
        model_type: Model type int. Z-Image Turbo (1) uses 9 steps and no CFG;
            Z-Image base (2) uses 50 steps and guidance 5.0.
    """
    from iartisanz.modules.generation.constants import MODEL_TYPE_DEFAULTS

    defaults = MODEL_TYPE_DEFAULTS.get(model_type, MODEL_TYPE_DEFAULTS[1])

    node_graph = ImageArtisanZNodeGraph()

    seed = NumberNode()
    node_graph.add_node(seed, "seed")

    image_width = NumberNode()
    node_graph.add_node(image_width, "image_width")

    image_height = NumberNode()
    node_graph.add_node(image_height, "image_height")

    num_inference_steps = NumberNode(number=defaults["num_inference_steps"])
    node_graph.add_node(num_inference_steps, "num_inference_steps")

    guidance_scale = NumberNode(number=defaults["guidance_scale"])
    node_graph.add_node(guidance_scale, "guidance_scale")

    guidance_start_end = NumberRangeNode(value=[0.0, 1.0])
    node_graph.add_node(guidance_start_end, "guidance_start_end")

    positive_prompt = TextNode()
    node_graph.add_node(positive_prompt, "positive_prompt")

    negative_prompt = TextNode()
    node_graph.add_node(negative_prompt, "negative_prompt")

    zimage_models = ZImageModelNode()
    node_graph.add_node(zimage_models, "model")

    scheduler = SchedulerNode(scheduler_data_object=SchedulerDataObject())
    node_graph.add_node(scheduler, "scheduler")

    prompts_encoder = ZImagePromptEncoderNode()
    prompts_encoder.connect("tokenizer", zimage_models, "tokenizer")
    prompts_encoder.connect("text_encoder", zimage_models, "text_encoder")
    prompts_encoder.connect("positive_prompt", positive_prompt, "value")
    prompts_encoder.connect("negative_prompt", negative_prompt, "value")
    node_graph.add_node(prompts_encoder, "prompts_encoder")

    latents = ZImageLatentsNode()
    latents.connect("seed", seed, "value")
    latents.connect("num_channels_latents", zimage_models, "num_channels_latents")
    latents.connect("width", image_width, "value")
    latents.connect("height", image_height, "value")
    latents.connect("vae_scale_factor", zimage_models, "vae_scale_factor")
    node_graph.add_node(latents, "latents")

    denoise = ZImageDenoiseNode()
    denoise.connect("transformer", zimage_models, "transformer")
    denoise.connect("num_inference_steps", num_inference_steps, "value")
    denoise.connect("latents", latents, "latents")
    denoise.connect("scheduler", scheduler, "scheduler")
    denoise.connect("prompt_embeds", prompts_encoder, "prompt_embeds")
    denoise.connect("negative_prompt_embeds", prompts_encoder, "negative_prompt_embeds")
    denoise.connect("guidance_scale", guidance_scale, "value")
    denoise.connect("guidance_start_end", guidance_start_end, "value")
    denoise.connect("positive_prompt_text", positive_prompt, "value")
    node_graph.add_node(denoise, "denoise")

    latents_decoder = ZImageLatentsDecoderNode()
    latents_decoder.connect("vae", zimage_models, "vae")
    latents_decoder.connect("latents", denoise, "latents")
    node_graph.add_node(latents_decoder, "decoder")

    image_send = ImageSendNode()
    image_send.connect("image", latents_decoder, "image")
    node_graph.add_node(image_send, "image_send")

    return node_graph


def create_default_flux2_graph(model_type: int = 3):
    """Create the default node graph for Flux.2 Klein text-to-image generation.

    Args:
        model_type: Model type int. Distilled variants (3, 5) use 4 steps and
            guidance 1.0; base variants (4, 6) use 30 steps and guidance 4.0.
    """
    from iartisanz.modules.generation.constants import MODEL_TYPE_DEFAULTS

    defaults = MODEL_TYPE_DEFAULTS.get(model_type, MODEL_TYPE_DEFAULTS[3])

    node_graph = ImageArtisanZNodeGraph()

    seed = NumberNode()
    node_graph.add_node(seed, "seed")

    image_width = NumberNode()
    node_graph.add_node(image_width, "image_width")

    image_height = NumberNode()
    node_graph.add_node(image_height, "image_height")

    num_inference_steps = NumberNode(number=defaults["num_inference_steps"])
    node_graph.add_node(num_inference_steps, "num_inference_steps")

    guidance_scale = NumberNode(number=defaults["guidance_scale"])
    node_graph.add_node(guidance_scale, "guidance_scale")

    guidance_start_end = NumberRangeNode(value=[0.0, 1.0])
    node_graph.add_node(guidance_start_end, "guidance_start_end")

    positive_prompt = TextNode()
    node_graph.add_node(positive_prompt, "positive_prompt")

    negative_prompt = TextNode()
    node_graph.add_node(negative_prompt, "negative_prompt")

    flux2_models = Flux2ModelNode()
    node_graph.add_node(flux2_models, "model")

    scheduler = SchedulerNode(scheduler_data_object=SchedulerDataObject())
    node_graph.add_node(scheduler, "scheduler")

    prompts_encoder = Flux2PromptEncoderNode()
    prompts_encoder.connect("tokenizer", flux2_models, "tokenizer")
    prompts_encoder.connect("text_encoder", flux2_models, "text_encoder")
    prompts_encoder.connect("positive_prompt", positive_prompt, "value")
    prompts_encoder.connect("negative_prompt", negative_prompt, "value")
    node_graph.add_node(prompts_encoder, "prompts_encoder")

    latents = Flux2LatentsNode()
    latents.connect("seed", seed, "value")
    latents.connect("num_channels_latents", flux2_models, "num_channels_latents")
    latents.connect("width", image_width, "value")
    latents.connect("height", image_height, "value")
    latents.connect("vae_scale_factor", flux2_models, "vae_scale_factor")
    node_graph.add_node(latents, "latents")

    denoise = Flux2DenoiseNode()
    denoise.connect("transformer", flux2_models, "transformer")
    denoise.connect("num_inference_steps", num_inference_steps, "value")
    denoise.connect("latents", latents, "latents")
    denoise.connect("latent_ids", latents, "latent_ids")
    denoise.connect("scheduler", scheduler, "scheduler")
    denoise.connect("prompt_embeds", prompts_encoder, "prompt_embeds")
    denoise.connect("text_ids", prompts_encoder, "text_ids")
    denoise.connect("negative_prompt_embeds", prompts_encoder, "negative_prompt_embeds")
    denoise.connect("negative_text_ids", prompts_encoder, "negative_text_ids")
    denoise.connect("guidance_scale", guidance_scale, "value")
    denoise.connect("guidance_start_end", guidance_start_end, "value")
    node_graph.add_node(denoise, "denoise")

    latents_decoder = Flux2LatentsDecoderNode()
    latents_decoder.connect("vae", flux2_models, "vae")
    latents_decoder.connect("latents", denoise, "latents")
    latents_decoder.connect("latent_ids", denoise, "latent_ids")
    node_graph.add_node(latents_decoder, "decoder")

    image_send = ImageSendNode()
    image_send.connect("image", latents_decoder, "image")
    node_graph.add_node(image_send, "image_send")

    return node_graph


def create_graph_for_model_type(model_type: int) -> ImageArtisanZNodeGraph:
    """Return the appropriate default graph for a given model type."""
    from iartisanz.modules.generation.constants import FLUX2_MODEL_TYPES

    if model_type in FLUX2_MODEL_TYPES:
        return create_default_flux2_graph(model_type)
    return create_default_graph(model_type)
