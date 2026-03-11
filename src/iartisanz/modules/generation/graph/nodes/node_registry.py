from __future__ import annotations

from .boolean_node import BooleanNode
from .choice_node import ChoiceNode
from .controlnet_conditioning_node import ControlNetConditioningNode
from .controlnet_model_node import ControlNetModelNode
from .flux2_denoise_node import Flux2DenoiseNode
from .flux2_dev_model_node import Flux2DevModelNode
from .flux2_dev_prompt_encode_node import Flux2DevPromptEncoderNode
from .flux2_edit_image_encode_node import Flux2EditImageEncodeNode
from .flux2_inpaint_base_latents_node import Flux2InpaintBaseLatentsNode
from .flux2_latents_decoder_node import Flux2LatentsDecoderNode
from .flux2_latents_node import Flux2LatentsNode
from .flux2_model_node import Flux2ModelNode
from .flux2_prompt_encode_node import Flux2PromptEncoderNode
from .image_load_node import ImageLoadNode
from .image_send_node import ImageSendNode
from .lora_node import LoraNode
from .number_node import NumberNode
from .number_range_node import NumberRangeNode
from .scheduler_node import SchedulerNode
from .text_node import TextNode
from .zimage_denoise_node import ZImageDenoiseNode
from .zimage_latents_decoder_node import ZImageLatentsDecoderNode
from .zimage_latents_node import ZImageLatentsNode
from .zimage_model_node import ZImageModelNode
from .zimage_prompt_encode_node import ZImagePromptEncoderNode


NODE_CLASSES = {
    "BooleanNode": BooleanNode,
    "ChoiceNode": ChoiceNode,
    "ControlNetConditioningNode": ControlNetConditioningNode,
    "ControlNetModelNode": ControlNetModelNode,
    "Flux2DenoiseNode": Flux2DenoiseNode,
    "Flux2DevModelNode": Flux2DevModelNode,
    "Flux2DevPromptEncoderNode": Flux2DevPromptEncoderNode,
    "Flux2EditImageEncodeNode": Flux2EditImageEncodeNode,
    "Flux2InpaintBaseLatentsNode": Flux2InpaintBaseLatentsNode,
    "Flux2LatentsDecoderNode": Flux2LatentsDecoderNode,
    "Flux2LatentsNode": Flux2LatentsNode,
    "Flux2ModelNode": Flux2ModelNode,
    "Flux2PromptEncoderNode": Flux2PromptEncoderNode,
    "ZImageDenoiseNode": ZImageDenoiseNode,
    "ImageLoadNode": ImageLoadNode,
    "ImageSendNode": ImageSendNode,
    "ZImageLatentsDecoderNode": ZImageLatentsDecoderNode,
    "ZImageLatentsNode": ZImageLatentsNode,
    "LoraNode": LoraNode,
    "NumberNode": NumberNode,
    "NumberRangeNode": NumberRangeNode,
    "ZImagePromptEncoderNode": ZImagePromptEncoderNode,
    "SchedulerNode": SchedulerNode,
    "TextNode": TextNode,
    "ZImageModelNode": ZImageModelNode,
    # Backward-compat aliases for saved graph JSON deserialization
    "DenoiseNode": ZImageDenoiseNode,
    "LatentsDecoderNode": ZImageLatentsDecoderNode,
    "LatentsNode": ZImageLatentsNode,
    "PromptEncoderNode": ZImagePromptEncoderNode,
}
