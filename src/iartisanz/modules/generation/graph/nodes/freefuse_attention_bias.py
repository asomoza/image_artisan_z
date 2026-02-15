"""FreeFuse attention bias for LoRA anti-bleeding.

Implements attention bias injection to suppress cross-attention between a LoRA's
image region and other LoRAs' text tokens, while encouraging attention between a
LoRA's image region and its own text tokens. Based on FreeFuse (arXiv: 2510.23515v2).

Z-Image sequence layout (basic mode): [image_tokens | caption_tokens]
"""

import logging

import torch


logger = logging.getLogger(__name__)

# Internal padding alignment used by Z-Image transformer
_SEQ_MULTI_OF = 32


def _ceil_to_multiple(n: int, m: int) -> int:
    return n + (-n) % m


def find_trigger_word_positions(
    tokenizer,
    prompt_text: str,
    trigger_words_str: str,
) -> list[int]:
    """Find token positions for trigger words in the tokenized prompt.

    Uses character-span mapping rather than token-subsequence matching,
    because subword tokenizers (SentencePiece, BPE) produce different
    tokens for the same word depending on surrounding context.

    Args:
        tokenizer: The tokenizer used for the prompt.
        prompt_text: The full prompt text.
        trigger_words_str: Comma-separated trigger words (e.g. "woman, lady").

    Returns:
        List of 0-based token indices in the caption embedding sequence.
    """
    if not trigger_words_str or not prompt_text:
        return []

    words = [w.strip() for w in trigger_words_str.split(",") if w.strip()]
    if not words:
        return []

    positions: list[int] = []
    used_chars: set[int] = set()
    used_tokens: set[int] = set()

    # Try offset-mapping approach first (fast tokenizers support this)
    offsets = _get_token_offsets(tokenizer, prompt_text)

    for word in words:
        # Find character span of trigger word in prompt
        char_start = prompt_text.find(word)
        if char_start == -1:
            logger.warning(
                f"[FreeFuse] Trigger word '{word}' not found in prompt text, skipping"
            )
            continue

        char_end = char_start + len(word)

        # Skip if overlapping with already-claimed characters
        if any(c in used_chars for c in range(char_start, char_end)):
            continue

        if offsets is not None:
            # Use offset mapping: find tokens whose character span overlaps
            word_tokens = _tokens_from_offsets(offsets, char_start, char_end, used_tokens)
        else:
            # Fallback: prefix-length method
            word_tokens = _tokens_from_prefix_length(
                tokenizer, prompt_text, char_start, char_end, used_tokens
            )

        if word_tokens:
            positions.extend(word_tokens)
            used_tokens.update(word_tokens)
            used_chars.update(range(char_start, char_end))
        else:
            logger.warning(
                f"[FreeFuse] Trigger word '{word}' could not be mapped to tokens, skipping"
            )

    return sorted(positions)


def _get_token_offsets(tokenizer, text: str) -> list[tuple[int, int]] | None:
    """Try to get (char_start, char_end) offsets for each token.

    Returns None if the tokenizer doesn't support offset mapping.
    """
    try:
        encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoding.get("offset_mapping")
        if offsets and len(offsets) == len(encoding["input_ids"]):
            return offsets
    except Exception:
        pass
    return None


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
        result.append(tok_idx)
    return result


def _tokens_from_prefix_length(
    tokenizer,
    prompt_text: str,
    char_start: int,
    char_end: int,
    used_tokens: set[int],
) -> list[int]:
    """Map a character span to token indices using prefix token counts.

    Tokenizes the prompt up to char_start and up to char_end, then the
    trigger word tokens are the difference in token count. This works
    regardless of tokenizer type because it preserves the full left context.
    """
    try:
        prefix_ids = tokenizer.encode(prompt_text[:char_start], add_special_tokens=False)
        prefix_plus_word_ids = tokenizer.encode(prompt_text[:char_end], add_special_tokens=False)

        tok_start = len(prefix_ids)
        tok_end = len(prefix_plus_word_ids)

        if tok_end <= tok_start:
            return []

        result = [i for i in range(tok_start, tok_end) if i not in used_tokens]
        return result
    except Exception:
        return []


def construct_attention_bias(
    lora_masks: dict[str, torch.Tensor],
    token_pos_maps: dict[str, list[int]],
    img_seq_len: int,
    txt_seq_len: int,
    bias_scale: float = 5.0,
    positive_bias_scale: float = 1.0,
    bidirectional: bool = True,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """Construct FreeFuse attention bias matrix.

    Args:
        lora_masks: {adapter_name: flat_image_mask (1, N_img)} with values 0-1.
        token_pos_maps: {adapter_name: [text_token_indices]} 0-based in caption seq.
        img_seq_len: Number of image tokens (including SEQ_MULTI_OF padding).
        txt_seq_len: Number of text tokens (including SEQ_MULTI_OF padding).
        bias_scale: Negative suppression strength.
        positive_bias_scale: Positive encouragement strength.
        bidirectional: Whether to apply text→image bias too.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Attention bias tensor of shape (1, total_seq, total_seq).
    """
    total_seq = img_seq_len + txt_seq_len
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

        # Image mask: (1, N_img) → broadcast to (1, img_seq_len)
        img_mask = lora_masks[name]  # (1, N_img)
        if img_mask.shape[1] < img_seq_len:
            # Pad with zeros for the SEQ_MULTI_OF alignment padding
            pad = torch.zeros(1, img_seq_len - img_mask.shape[1], device=device, dtype=dtype)
            img_mask = torch.cat([img_mask, pad], dim=1)
        img_mask = img_mask[:, :img_seq_len].to(device=device, dtype=dtype)  # (1, img_seq_len)

        # Text masks for this LoRA vs others
        same_text = (text_owner == lora_idx).float().unsqueeze(0).to(device=device, dtype=dtype)  # (1, txt_seq_len)
        other_text = ((text_owner >= 0) & (text_owner != lora_idx)).float().unsqueeze(0).to(
            device=device, dtype=dtype
        )  # (1, txt_seq_len)

        # Image→Text suppression: image tokens attending to OTHER LoRAs' text tokens
        # bias[:, :img, img:] += img_mask.T @ other_text * (-bias_scale)
        bias[:, :img_seq_len, img_seq_len:] += (
            img_mask.unsqueeze(2) * other_text.unsqueeze(1) * (-bias_scale)
        )

        # Image→Text encouragement: image tokens attending to OWN text tokens
        # bias[:, :img, img:] += img_mask.T @ same_text * positive_bias_scale
        bias[:, :img_seq_len, img_seq_len:] += (
            img_mask.unsqueeze(2) * same_text.unsqueeze(1) * positive_bias_scale
        )

        if bidirectional:
            # Text→Image suppression: own text tokens attending to NOT-this-LoRA image regions
            not_this_img = (1.0 - img_mask)  # (1, img_seq_len)
            bias[:, img_seq_len:, :img_seq_len] += (
                same_text.unsqueeze(2) * not_this_img.unsqueeze(1) * (-bias_scale)
            )

            # Text→Image encouragement: own text tokens attending to this LoRA's image region
            bias[:, img_seq_len:, :img_seq_len] += (
                same_text.unsqueeze(2) * img_mask.unsqueeze(1) * positive_bias_scale
            )

    return bias


class _BiasInjectionProcessor:
    """Wrapper processor that injects FreeFuse attention bias.

    Replaces the original attention processor on each layer.  The wrapper
    converts the 2D bool padding mask into a 4D float mask with bias added,
    then delegates to the original processor.

    Note: We replace the processor *instance* rather than patching
    ``proc.__call__`` because Python resolves ``__call__`` on the **class**
    for implicit calls (``proc(...)``), not on the instance dict.
    """

    def __init__(self, original_processor, attention_bias: torch.Tensor):
        self._original = original_processor
        self._bias = attention_bias
        # Copy attributes that callers might inspect
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
        freqs_cis=None,
    ):
        if attention_mask is not None and attention_mask.ndim == 2:
            B, S = attention_mask.shape
            dt = hidden_states.dtype

            # Convert 2D bool padding mask → 4D float mask (B, 1, S, S)
            bool_4d = attention_mask.bool().unsqueeze(1).unsqueeze(1)  # (B, 1, 1, S)
            float_mask = torch.zeros(
                B, 1, S, S, device=attention_mask.device, dtype=dt
            )
            float_mask.masked_fill_(~bool_4d, torch.finfo(dt).min)

            # Slice bias to match actual sequence length
            bias_slice = self._bias[:, :S, :S]
            if bias_slice.shape[0] < B:
                bias_slice = bias_slice.expand(B, -1, -1)
            bias_4d = bias_slice.unsqueeze(1).to(
                device=float_mask.device, dtype=dt
            )

            attention_mask = float_mask + bias_4d

        return self._original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            freqs_cis=freqs_cis,
        )


def patch_attention_with_bias(
    transformer, attention_bias: torch.Tensor, last_half_only: bool = True
) -> list:
    """Replace transformer attention processors with bias-injecting wrappers.

    Each layer's processor is swapped out for a :class:`_BiasInjectionProcessor`
    that converts the 2D bool padding mask into a 4D float mask with the
    FreeFuse bias added, then delegates to the original processor.

    Args:
        transformer: The Z-Image transformer model.
        attention_bias: Bias tensor of shape (B, total_seq, total_seq).
        last_half_only: If True, only patch the last half of layers (matching
            FreeFuse reference ``attention_bias_blocks="last_half"``).

    Returns:
        List of (layer_attention, original_processor) tuples for cleanup.
    """
    patched = []
    n_layers = len(transformer.layers)
    start_idx = n_layers // 2 if last_half_only else 0

    for i in range(start_idx, n_layers):
        layer = transformer.layers[i]
        attn_module = layer.attention
        original_proc = attn_module.processor
        attn_module.processor = _BiasInjectionProcessor(original_proc, attention_bias)
        patched.append((attn_module, original_proc))

    return patched


def unpatch_attention_bias(patched: list) -> None:
    """Restore original attention processors."""
    for attn_module, original_proc in patched:
        attn_module.processor = original_proc
