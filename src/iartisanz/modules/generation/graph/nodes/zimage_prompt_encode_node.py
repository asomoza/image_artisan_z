import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


class ZImagePromptEncoderNode(Node):
    PRIORITY = 0
    REQUIRED_INPUTS = ["tokenizer", "text_encoder", "positive_prompt"]
    OPTIONAL_INPUTS = ["negative_prompt"]
    OUTPUTS = ["prompt_embeds", "negative_prompt_embeds"]

    CHAT_TEMPLATE = {"role": "user", "content": ""}

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()

        with mm.use_components("text_encoder", device=self.device):
            tokenizer = mm.resolve(self.tokenizer)
            text_encoder = mm.resolve(self.text_encoder)

            negative_prompt = self.negative_prompt if self.negative_prompt is not None else ""

            prompt_embeds = self.encode_prompt(self.positive_prompt, tokenizer=tokenizer, text_encoder=text_encoder)
            negative_prompt_embeds = self.encode_prompt(
                negative_prompt,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
            )

            self.values["prompt_embeds"] = prompt_embeds.to("cpu").detach().clone()
            self.values["negative_prompt_embeds"] = negative_prompt_embeds.to("cpu").detach().clone()

        return self.values

    def encode_prompt(self, prompt, *, tokenizer, text_encoder, max_sequence_length=512):
        message = {**self.CHAT_TEMPLATE, "content": prompt}

        text_encoder_prompt = tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        text_inputs = tokenizer(
            text_encoder_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        # Keep the text encoder wherever ModelManager placed it (often CPU).
        # Move input tensors to the encoder device rather than forcing graph device.
        encoder_device = next(text_encoder.parameters()).device
        text_input_ids = text_inputs.input_ids.to(encoder_device)
        prompt_masks = text_inputs.attention_mask.to(encoder_device).bool()

        prompt_embeds = text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        prompt_embeds = prompt_embeds[0, prompt_masks[0]]

        return prompt_embeds
