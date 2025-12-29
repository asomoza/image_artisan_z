import torch

from iartisanz.modules.generation.graph.nodes.node import Node


class PromptEncoderNode(Node):
    PRIORITY = 0
    REQUIRED_INPUTS = ["tokenizer", "text_encoder", "positive_prompt"]
    OPTIONAL_INPUTS = ["negative_prompt"]
    OUTPUTS = ["prompt_embeds", "negative_prompt_embeds"]

    CHAT_TEMPLATE = {"role": "user", "content": ""}

    @torch.inference_mode()
    def __call__(self):
        negative_prompt = self.negative_prompt if self.negative_prompt is not None else ""

        prompt_embeds = self.encode_prompt(self.positive_prompt)
        negative_prompt_embeds = self.encode_prompt(negative_prompt)

        self.values["prompt_embeds"] = prompt_embeds.to("cpu").detach().clone()
        self.values["negative_prompt_embeds"] = negative_prompt_embeds.to("cpu").detach().clone()

        return self.values

    def encode_prompt(self, prompt, max_sequence_length=512):
        message = {**self.CHAT_TEMPLATE, "content": prompt}

        text_encoder_prompt = self.tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )

        text_inputs = self.tokenizer(
            text_encoder_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(self.device)
        prompt_masks = text_inputs.attention_mask.to(self.device).bool()

        prompt_embeds = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2]

        prompt_embeds = prompt_embeds[0, prompt_masks[0]]

        return prompt_embeds
