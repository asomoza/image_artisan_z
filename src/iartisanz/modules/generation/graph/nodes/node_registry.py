from __future__ import annotations

from .boolean_node import BooleanNode
from .denoise_node import DenoiseNode
from .image_load_node import ImageLoadNode
from .image_send_node import ImageSendNode
from .latents_decoder_node import LatentsDecoderNode
from .latents_node import LatentsNode
from .lora_node import LoraNode
from .number_node import NumberNode
from .number_range_node import NumberRangeNode
from .prompt_encode_node import PromptEncoderNode
from .scheduler_node import SchedulerNode
from .text_node import TextNode
from .zimage_model_node import ZImageModelNode


NODE_CLASSES = {
    "BooleanNode": BooleanNode,
    "DenoiseNode": DenoiseNode,
    "ImageLoadNode": ImageLoadNode,
    "ImageSendNode": ImageSendNode,
    "LatentsDecoderNode": LatentsDecoderNode,
    "LatentsNode": LatentsNode,
    "LoraNode": LoraNode,
    "NumberNode": NumberNode,
    "NumberRangeNode": NumberRangeNode,
    "PromptEncoderNode": PromptEncoderNode,
    "SchedulerNode": SchedulerNode,
    "TextNode": TextNode,
    "ZImageModelNode": ZImageModelNode,
}
