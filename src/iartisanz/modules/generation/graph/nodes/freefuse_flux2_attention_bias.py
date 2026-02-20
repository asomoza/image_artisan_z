"""FreeFuse attention bias for Flux2 Klein LoRA anti-bleeding.

Implements attention bias injection to suppress cross-attention between a LoRA's
image region and other LoRAs' text tokens, while encouraging attention between a
LoRA's image region and its own text tokens. Based on FreeFuse (arXiv: 2510.23515v2).

Flux2 sequence layout: [text_tokens | image_tokens]

Key differences from Z-Image:
- Sequence layout is [text | image] (Z-Image uses [image | text])
- No SEQ_MULTI_OF=32 padding — actual sequence lengths used directly
- Dual-stream (double blocks) + single-stream (single blocks) architecture
- Real-valued RoPE via diffusers apply_rotary_emb (Z-Image uses complex-valued)
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def find_trigger_word_positions_flux2(
    tokenizer,
    prompt_text: str,
    trigger_words_str: str,
    max_sequence_length: int = 512,
) -> list[int]:
    """Find token positions for trigger words in the Flux2-tokenized prompt.

    Replicates the same tokenization path as Flux2PromptEncoderNode.encode_prompt():
    1. Apply chat template
    2. Tokenize with padding="max_length"
    3. Map trigger words via character offsets in the templated text

    Args:
        tokenizer: The Qwen tokenizer used for the prompt.
        prompt_text: The full prompt text (before chat template).
        trigger_words_str: Comma-separated trigger words (e.g. "woman, lady").
        max_sequence_length: Max token length (must match encode_prompt).

    Returns:
        Sorted list of 0-based token indices.
    """
    if not trigger_words_str or not prompt_text:
        return []

    words = [w.strip() for w in trigger_words_str.split(",") if w.strip()]
    if not words:
        return []

    # Apply the same chat template as Flux2PromptEncoderNode
    templated_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt_text}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Tokenize with offset mapping
    try:
        encoding = tokenizer(
            templated_text,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_offsets_mapping=True,
        )
        offsets = encoding.get("offset_mapping")
    except Exception:
        offsets = None

    positions: list[int] = []
    used_chars: set[int] = set()
    used_tokens: set[int] = set()

    for word in words:
        # Find character span of trigger word in the templated text
        char_start = templated_text.find(word)
        if char_start == -1:
            logger.warning(
                "[FreeFuse Flux2] Trigger word '%s' not found in templated prompt, skipping",
                word,
            )
            continue

        char_end = char_start + len(word)

        # Skip if overlapping with already-claimed characters
        if any(c in used_chars for c in range(char_start, char_end)):
            continue

        if offsets is not None:
            word_tokens = _tokens_from_offsets(offsets, char_start, char_end, used_tokens)
        else:
            word_tokens = _tokens_from_prefix_length(
                tokenizer, templated_text, char_start, char_end, used_tokens
            )

        if word_tokens:
            positions.extend(word_tokens)
            used_tokens.update(word_tokens)
            used_chars.update(range(char_start, char_end))
        else:
            logger.warning(
                "[FreeFuse Flux2] Trigger word '%s' could not be mapped to tokens, skipping",
                word,
            )

    return sorted(positions)


def _tokens_from_offsets(
    offsets: list[tuple[int, int]],
    char_start: int,
    char_end: int,
    used_tokens: set[int],
) -> list[int]:
    """Map a character span to token indices using offset mapping."""
    result = []
    for tok_idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_idx in used_tokens:
            continue
        if tok_end <= char_start or tok_start >= char_end:
            continue
        # Skip padding tokens (both offsets are 0)
        if tok_start == 0 and tok_end == 0 and tok_idx > 0:
            continue
        result.append(tok_idx)
    return result


def _tokens_from_prefix_length(
    tokenizer,
    templated_text: str,
    char_start: int,
    char_end: int,
    used_tokens: set[int],
) -> list[int]:
    """Map a character span to token indices using prefix token counts."""
    try:
        prefix_ids = tokenizer.encode(templated_text[:char_start], add_special_tokens=False)
        prefix_plus_word_ids = tokenizer.encode(templated_text[:char_end], add_special_tokens=False)

        tok_start = len(prefix_ids)
        tok_end = len(prefix_plus_word_ids)

        if tok_end <= tok_start:
            return []

        return [i for i in range(tok_start, tok_end) if i not in used_tokens]
    except Exception:
        return []


def construct_flux2_attention_bias(
    lora_masks: dict[str, torch.Tensor],
    token_pos_maps: dict[str, list[int]],
    txt_seq_len: int,
    img_seq_len: int,
    bias_scale: float = 5.0,
    positive_bias_scale: float = 1.0,
    bidirectional: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Construct FreeFuse attention bias matrix for Flux2.

    Flux2 sequence layout: [text_tokens | image_tokens]

    Args:
        lora_masks: {adapter_name: flat_image_mask (1, img_seq_len)} with values 0-1.
        token_pos_maps: {adapter_name: [text_token_indices]} 0-based in text seq.
        txt_seq_len: Number of text tokens.
        img_seq_len: Number of image tokens.
        bias_scale: Negative suppression strength.
        positive_bias_scale: Positive encouragement strength.
        bidirectional: Whether to apply text->image bias too.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Attention bias tensor of shape (1, txt_seq_len + img_seq_len, txt_seq_len + img_seq_len).
    """
    total_seq = txt_seq_len + img_seq_len
    bias = torch.zeros(1, total_seq, total_seq, device=device, dtype=dtype)

    # Build text token ownership: -1 = unowned, lora_idx otherwise
    adapter_names = sorted(lora_masks.keys())
    text_owner = torch.full((txt_seq_len,), -1, dtype=torch.long, device=device)
    for lora_idx, name in enumerate(adapter_names):
        for pos in token_pos_maps.get(name, []):
            if pos < txt_seq_len:
                text_owner[pos] = lora_idx

    for lora_idx, name in enumerate(adapter_names):
        if name not in lora_masks or name not in token_pos_maps:
            continue

        # Image mask: (1, img_seq_len)
        img_mask = lora_masks[name]
        if img_mask.shape[1] < img_seq_len:
            pad = torch.zeros(1, img_seq_len - img_mask.shape[1], device=device, dtype=dtype)
            img_mask = torch.cat([img_mask, pad], dim=1)
        img_mask = img_mask[:, :img_seq_len].to(device=device, dtype=dtype)

        # Text masks for this LoRA vs others
        same_text = (text_owner == lora_idx).float().unsqueeze(0).to(device=device, dtype=dtype)
        other_text = (
            ((text_owner >= 0) & (text_owner != lora_idx))
            .float()
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )

        # Image->Text suppression: image tokens attending to OTHER LoRAs' text tokens
        # bias[:, txt_len:, :txt_len] is the image-query, text-key quadrant
        bias[:, txt_seq_len:, :txt_seq_len] += (
            img_mask.unsqueeze(2) * other_text.unsqueeze(1) * (-bias_scale)
        )

        # Image->Text encouragement: image tokens attending to OWN text tokens
        bias[:, txt_seq_len:, :txt_seq_len] += (
            img_mask.unsqueeze(2) * same_text.unsqueeze(1) * positive_bias_scale
        )

        if bidirectional:
            # Text->Image suppression: own text tokens attending to NOT-this-LoRA image regions
            not_this_img = 1.0 - img_mask
            bias[:, :txt_seq_len, txt_seq_len:] += (
                same_text.unsqueeze(2) * not_this_img.unsqueeze(1) * (-bias_scale)
            )

            # Text->Image encouragement: own text tokens attending to this LoRA's image region
            bias[:, :txt_seq_len, txt_seq_len:] += (
                same_text.unsqueeze(2) * img_mask.unsqueeze(1) * positive_bias_scale
            )

    return bias


class _Flux2BiasInjectionProcessor:
    """Wrapper processor that injects FreeFuse attention bias into Flux2 double-stream blocks.

    Replaces the original Flux2Attention processor. The wrapper adds the bias
    to the attention mask, then delegates to the original processor.
    """

    def __init__(self, original_processor, attention_bias: torch.Tensor):
        self._original = original_processor
        self._bias = attention_bias
        if hasattr(original_processor, "_attention_backend"):
            self._attention_backend = original_processor._attention_backend
        if hasattr(original_processor, "_parallel_config"):
            self._parallel_config = original_processor._parallel_config

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        attention_mask = self._inject_bias(attention_mask, hidden_states, encoder_hidden_states)
        return self._original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    def _inject_bias(self, attention_mask, hidden_states, encoder_hidden_states):
        """Combine existing attention mask with additive bias."""
        # Determine total sequence length: [text | image] for double-stream
        img_len = hidden_states.shape[1]
        txt_len = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
        S = txt_len + img_len
        B = hidden_states.shape[0]

        bias = self._bias
        if bias.shape[0] < B:
            bias = bias.expand(B, -1, -1)

        # Slice/pad bias to match actual sequence length
        if bias.shape[1] < S:
            pad_s = S - bias.shape[1]
            bias = F.pad(bias, (0, pad_s, 0, pad_s), value=0.0)
        bias = bias[:, :S, :S]

        # Convert to 4D: (B, 1, S, S)
        bias_4d = bias.unsqueeze(1).to(device=hidden_states.device, dtype=hidden_states.dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            # 2D bool mask -> 4D float mask + bias
            float_mask = torch.zeros(
                B, 1, 1, S, device=attention_mask.device, dtype=hidden_states.dtype
            )
            float_mask.masked_fill_(
                ~attention_mask.bool().unsqueeze(1).unsqueeze(1),
                torch.finfo(hidden_states.dtype).min,
            )
            return float_mask + bias_4d

        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask + bias_4d

        return bias_4d


class _Flux2SingleBiasInjectionProcessor:
    """Wrapper processor that injects FreeFuse attention bias into Flux2 single-stream blocks.

    Single-stream blocks operate on the already-concatenated [text | image] sequence.
    """

    def __init__(self, original_processor, attention_bias: torch.Tensor):
        self._original = original_processor
        self._bias = attention_bias
        if hasattr(original_processor, "_attention_backend"):
            self._attention_backend = original_processor._attention_backend
        if hasattr(original_processor, "_parallel_config"):
            self._parallel_config = original_processor._parallel_config

    def __call__(
        self,
        attn,
        hidden_states,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        attention_mask = self._inject_bias(attention_mask, hidden_states)
        return self._original(
            attn,
            hidden_states,
            attention_mask=attention_mask,
            image_rotary_emb=image_rotary_emb,
        )

    def _inject_bias(self, attention_mask, hidden_states):
        """Combine existing attention mask with additive bias."""
        S = hidden_states.shape[1]
        B = hidden_states.shape[0]

        bias = self._bias
        if bias.shape[0] < B:
            bias = bias.expand(B, -1, -1)

        if bias.shape[1] < S:
            pad_s = S - bias.shape[1]
            bias = F.pad(bias, (0, pad_s, 0, pad_s), value=0.0)
        bias = bias[:, :S, :S]

        bias_4d = bias.unsqueeze(1).to(device=hidden_states.device, dtype=hidden_states.dtype)

        if attention_mask is not None and attention_mask.ndim == 2:
            float_mask = torch.zeros(
                B, 1, 1, S, device=attention_mask.device, dtype=hidden_states.dtype
            )
            float_mask.masked_fill_(
                ~attention_mask.bool().unsqueeze(1).unsqueeze(1),
                torch.finfo(hidden_states.dtype).min,
            )
            return float_mask + bias_4d

        if attention_mask is not None and attention_mask.ndim == 4:
            return attention_mask + bias_4d

        return bias_4d


def patch_flux2_attention_with_bias(
    transformer, attention_bias: torch.Tensor, last_half_only: bool = True
) -> list:
    """Replace Flux2 transformer attention processors with bias-injecting wrappers.

    Patches both double-stream (transformer_blocks) and single-stream
    (single_transformer_blocks) attention processors.

    Args:
        transformer: The Flux2 transformer model.
        attention_bias: Bias tensor of shape (1, total_seq, total_seq).
        last_half_only: If True, only patch the last half of each block list.

    Returns:
        List of (attn_module, original_processor) tuples for cleanup.
    """
    patched = []

    # Double-stream blocks
    double_blocks = getattr(transformer, "transformer_blocks", None)
    if double_blocks is not None:
        n = len(double_blocks)
        start = n // 2 if last_half_only else 0
        for i in range(start, n):
            block = double_blocks[i]
            attn_module = block.attn
            original_proc = attn_module.processor
            attn_module.processor = _Flux2BiasInjectionProcessor(original_proc, attention_bias)
            patched.append((attn_module, original_proc))

    # Single-stream blocks
    single_blocks = getattr(transformer, "single_transformer_blocks", None)
    if single_blocks is not None:
        n = len(single_blocks)
        start = n // 2 if last_half_only else 0
        for i in range(start, n):
            block = single_blocks[i]
            attn_module = block.attn
            original_proc = attn_module.processor
            attn_module.processor = _Flux2SingleBiasInjectionProcessor(
                original_proc, attention_bias
            )
            patched.append((attn_module, original_proc))

    return patched


def unpatch_flux2_attention_bias(patched: list) -> None:
    """Restore original attention processors."""
    for attn_module, original_proc in patched:
        attn_module.processor = original_proc
