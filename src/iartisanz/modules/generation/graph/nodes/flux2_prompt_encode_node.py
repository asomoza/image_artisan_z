import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


class Flux2PromptEncoderNode(Node):
    """Prompt encoder for Flux.2 Klein models.

    Uses Qwen3ForCausalLM (generative model) and extracts hidden states
    from layers 9, 18, 27.  Stacks and reshapes to produce
    (B, seq_len, 3 * hidden_dim) embeddings.

    Also generates 4D text position IDs for the transformer.
    """

    PRIORITY = 0
    REQUIRED_INPUTS = ["tokenizer", "text_encoder", "positive_prompt"]
    OPTIONAL_INPUTS = ["negative_prompt"]
    OUTPUTS = ["prompt_embeds", "negative_prompt_embeds", "text_ids", "negative_text_ids"]

    CHAT_TEMPLATE = {"role": "user", "content": ""}
    HIDDEN_STATES_LAYERS = (9, 18, 27)

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()

        tokenizer = mm.resolve(self.tokenizer)
        text_encoder = mm.resolve(self.text_encoder)

        negative_prompt = self.negative_prompt if self.negative_prompt is not None else ""

        prompt_embeds = self.encode_prompt(self.positive_prompt, tokenizer=tokenizer, text_encoder=text_encoder)
        negative_prompt_embeds = self.encode_prompt(negative_prompt, tokenizer=tokenizer, text_encoder=text_encoder)

        text_ids = self._prepare_text_ids(prompt_embeds)
        negative_text_ids = self._prepare_text_ids(negative_prompt_embeds)

        self.values["prompt_embeds"] = prompt_embeds.to("cpu").detach().clone()
        self.values["negative_prompt_embeds"] = negative_prompt_embeds.to("cpu").detach().clone()
        self.values["text_ids"] = text_ids.to("cpu").detach().clone()
        self.values["negative_text_ids"] = negative_text_ids.to("cpu").detach().clone()

        return self.values

    def encode_prompt(self, prompt, *, tokenizer, text_encoder, max_sequence_length=512):
        message = {**self.CHAT_TEMPLATE, "content": prompt}

        text_encoder_prompt = tokenizer.apply_chat_template(
            [message],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        text_inputs = tokenizer(
            text_encoder_prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        encoder_device = next(text_encoder.parameters()).device
        input_ids = text_inputs.input_ids.to(encoder_device)
        attention_mask = text_inputs.attention_mask.to(encoder_device)

        output = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # Stack hidden states from intermediate layers
        out = torch.stack(
            [output.hidden_states[k] for k in self.HIDDEN_STATES_LAYERS],
            dim=1,
        )

        dtype = next(text_encoder.parameters()).dtype
        out = out.to(dtype=dtype)

        # (B, num_layers, seq_len, hidden_dim) -> (B, seq_len, num_layers * hidden_dim)
        batch_size, num_channels, seq_len, hidden_dim = out.shape
        prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

        return prompt_embeds

    @staticmethod
    def _prepare_text_ids(x: torch.Tensor) -> torch.Tensor:
        """Generate 4D position coordinates (T, H, W, L) for text embeddings.

        Args:
            x: Text embeddings of shape (B, L, D).

        Returns:
            Position IDs of shape (B, L, 4).
        """
        B, L, _ = x.shape
        out_ids = []

        for _i in range(B):
            t = torch.arange(1)
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)
            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)
