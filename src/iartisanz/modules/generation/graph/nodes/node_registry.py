from __future__ import annotations

from .denoise_node import DenoiseNode
from .image_send_node import ImageSendNode
from .latents_decoder_node import LatentsDecoderNode
from .latents_node import LatentsNode
from .lora_node import LoraNode
from .number_node import NumberNode
from .prompt_encode_node import PromptEncoderNode
from .scheduler_node import SchedulerNode
from .text_node import TextNode
from .zimage_model_node import ZImageModelNode


NODE_CLASSES = {
    "NumberNode": NumberNode,
    "TextNode": TextNode,
    "ZImageModelNode": ZImageModelNode,
    "SchedulerNode": SchedulerNode,
    "PromptEncoderNode": PromptEncoderNode,
    "LatentsNode": LatentsNode,
    "DenoiseNode": DenoiseNode,
    "LatentsDecoderNode": LatentsDecoderNode,
    "ImageSendNode": ImageSendNode,
    "LoraNode": LoraNode,
}
