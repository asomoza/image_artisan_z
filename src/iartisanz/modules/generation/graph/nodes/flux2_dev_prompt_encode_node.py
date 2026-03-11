import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


SYSTEM_MESSAGE = (
    "You are an AI that reasons about image descriptions. You give structured responses "
    "focusing on object relationships, object attribution and actions without speculation."
)


def _format_input(prompt: str, system_message: str) -> list[dict]:
    """Format a single prompt into Mistral structured chat messages.

    PixtralProcessor's apply_chat_template expects structured content
    (list of dicts with type/text keys), not plain strings.
    """
    return [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [{"type": "text", "text": prompt}]},
    ]


class Flux2DevPromptEncoderNode(Node):
    """Prompt encoder for Flux.2 Dev models.

    Uses Mistral3ForConditionalGeneration with PixtralProcessor (AutoProcessor).
    Extracts hidden states from layers 10, 20, 30 and stacks them to produce
    (B, seq_len, 3 * hidden_dim) embeddings.

    Also generates 4D text position IDs for the transformer.
    """

    PRIORITY = 0
    REQUIRED_INPUTS = ["tokenizer", "text_encoder", "positive_prompt"]
    OPTIONAL_INPUTS = ["negative_prompt"]
    OUTPUTS = ["prompt_embeds", "negative_prompt_embeds", "text_ids", "negative_text_ids"]

    HIDDEN_STATES_LAYERS = (10, 20, 30)

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()

        with mm.use_components("text_encoder", device=self.device):
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
        messages = _format_input(prompt, SYSTEM_MESSAGE)

        inputs = tokenizer.apply_chat_template(
            [messages],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_sequence_length,
        )

        encoder_device = next(text_encoder.parameters()).device
        input_ids = inputs["input_ids"].to(encoder_device)
        attention_mask = inputs["attention_mask"].to(encoder_device)

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
