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

    # Tokenize full prompt (without special tokens — they are added by the encoder)
    full_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

    positions: list[int] = []
    used: set[int] = set()

    for word in words:
        word_ids = tokenizer.encode(word, add_special_tokens=False)
        if not word_ids:
            logger.warning(f"[FreeFuse] Trigger word '{word}' produced no tokens, skipping")
            continue

        # Sliding window search for subsequence
        found = False
        for start in range(len(full_ids) - len(word_ids) + 1):
            if start in used:
                continue
            if full_ids[start : start + len(word_ids)] == word_ids:
                new_positions = list(range(start, start + len(word_ids)))
                # Skip if overlapping with already-claimed positions
                if any(p in used for p in new_positions):
                    continue
                positions.extend(new_positions)
                used.update(new_positions)
                found = True
                break

        if not found:
            logger.warning(
                f"[FreeFuse] Trigger word '{word}' not found in prompt tokens, skipping"
            )

    return sorted(positions)


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


def patch_attention_with_bias(transformer, attention_bias: torch.Tensor) -> list:
    """Monkey-patch transformer attention processors to inject FreeFuse bias.

    Intercepts each layer's attention processor to convert the 2D bool padding mask
    into a 4D float mask with the FreeFuse bias added. The processor's own 2D→4D
    conversion is then skipped (since ndim != 2).

    Args:
        transformer: The Z-Image transformer model.
        attention_bias: Bias tensor of shape (B, total_seq, total_seq).

    Returns:
        List of (processor, original_call) tuples for cleanup.
    """
    patched = []
    for layer in transformer.layers:
        proc = layer.attention.processor
        original_call = proc.__call__

        def make_patched(orig, bias):
            def patched_call(
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
                    bias_slice = bias[:, :S, :S]
                    if bias_slice.shape[0] < B:
                        bias_slice = bias_slice.expand(B, -1, -1)
                    bias_4d = bias_slice.unsqueeze(1).to(
                        device=float_mask.device, dtype=dt
                    )

                    attention_mask = float_mask + bias_4d

                return orig(
                    attn,
                    hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    freqs_cis=freqs_cis,
                )

            return patched_call

        proc.__call__ = make_patched(original_call, attention_bias)
        patched.append((proc, original_call))

    return patched


def unpatch_attention_bias(patched: list) -> None:
    """Restore original attention processor __call__ methods."""
    for proc, original_call in patched:
        proc.__call__ = original_call
