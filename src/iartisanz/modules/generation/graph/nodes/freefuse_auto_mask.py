"""FreeFuse Phase B: Auto-mask extraction from cross-attention similarity maps.

When LoRAs have trigger words but no user-painted spatial masks, this module
derives masks automatically using a two-phase denoise approach:

1. Phase 1: A few denoise steps **without LoRA** collect cross-attention
   similarity maps showing which image regions correspond to which trigger words.
2. The maps are processed into binary spatial masks via background thresholding,
   morphological cleanup, and balanced argmax assignment.
3. Phase 2 (in denoise node): Full denoise with LoRA + derived masks for
   attention bias, reusing Phase A's FreeFuse infrastructure.

Based on FreeFuse (arXiv: 2510.23515v2).

Z-Image sequence layout (basic mode): [image_tokens | caption_tokens]
"""

import logging

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn


logger = logging.getLogger(__name__)

# Internal padding alignment used by Z-Image transformer
_SEQ_MULTI_OF = 32


def _ceil_to_multiple(n: int, m: int) -> int:
    return n + (-n) % m


# ---------------------------------------------------------------------------
# Sim map collector: replaces one layer's attention processor
# ---------------------------------------------------------------------------

class _SimMapCollectorProcessor:
    """Replacement attention processor that collects cross-attention sim maps.

    Replicates standard ``ZSingleStreamAttnProcessor`` logic while capturing
    Q, K, and hidden_states for concept similarity map extraction.

    Note: We replace the processor *instance* rather than patching
    ``proc.__call__`` because Python resolves ``__call__`` on the **class**
    for implicit calls (``proc(...)``), not on the instance dict.
    """

    def __init__(
        self,
        original_processor,
        token_pos_maps: dict[str, list[int]],
        img_seq_len: int,
        cap_seq_len: int,
        text_start_idx: int,
        top_k_ratio: float = 0.1,
    ):
        self._original = original_processor
        self.token_pos_maps = token_pos_maps
        self.img_seq_len = img_seq_len
        self.cap_seq_len = cap_seq_len
        self.text_start_idx = text_start_idx
        self.top_k_ratio = top_k_ratio
        self.cal_concept_sim_map = False
        self.concept_sim_maps = None
        self.total_seq = None
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
        # Track actual sequence length for downstream use
        self.total_seq = hidden_states.shape[1]

        # QKV projections
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        # QK normalization (RMSNorm)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # RoPE (complex-valued, Z-Image style)
        if freqs_cis is not None:
            query = _apply_rotary_emb(query, freqs_cis)
            key = _apply_rotary_emb(key, freqs_cis)

        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)

        # Attention mask: 2D -> 4D
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # SDPA
        hidden_states_out = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )

        hidden_states_out = hidden_states_out.flatten(2, 3)
        hidden_states_out = hidden_states_out.to(dtype)

        # Extract concept sim maps on last Phase 1 step
        if self.cal_concept_sim_map:
            self.concept_sim_maps = _extract_concept_sim_maps(
                query,
                key,
                hidden_states_out,
                self.token_pos_maps,
                self.img_seq_len,
                self.cap_seq_len,
                self.text_start_idx,
                self.top_k_ratio,
            )

        # Output projection
        output = attn.to_out[0](hidden_states_out)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)

        return output


def install_sim_map_collector(
    transformer,
    block_idx: int,
    token_pos_maps: dict[str, list[int]],
    img_seq_len: int,
    cap_seq_len: int,
    top_k_ratio: float = 0.1,
) -> dict:
    """Install a similarity map collector on one transformer layer.

    Replaces the attention processor with a :class:`_SimMapCollectorProcessor`
    that replicates standard ``ZSingleStreamAttnProcessor`` logic while
    capturing Q, K, and hidden_states for sim map extraction.

    Args:
        transformer: The Z-Image transformer model.
        block_idx: Which layer to install the collector on.
        token_pos_maps: {adapter_name: [text_token_indices]} for trigger words.
        img_seq_len: Number of image tokens (pre-padding).
        cap_seq_len: Number of caption tokens (pre-padding).
        top_k_ratio: Fraction of image tokens to select as "core" tokens.

    Returns:
        State dict for removal via :func:`remove_sim_map_collector`.
    """
    layer = transformer.layers[block_idx]
    attn_module = layer.attention
    original_proc = attn_module.processor

    # Text tokens start after the *padded* image tokens in the unified sequence.
    text_start_idx = _ceil_to_multiple(img_seq_len, _SEQ_MULTI_OF)

    collector = _SimMapCollectorProcessor(
        original_proc,
        token_pos_maps=token_pos_maps,
        img_seq_len=img_seq_len,
        cap_seq_len=cap_seq_len,
        text_start_idx=text_start_idx,
        top_k_ratio=top_k_ratio,
    )
    attn_module.processor = collector

    return {
        "attn_module": attn_module,
        "original_proc": original_proc,
        "collector": collector,
    }


def remove_sim_map_collector(state: dict) -> None:
    """Restore original attention processor."""
    state["attn_module"].processor = state["original_proc"]


# ---------------------------------------------------------------------------
# RoPE helper (matches ZSingleStreamAttnProcessor's inline implementation)
# ---------------------------------------------------------------------------

def _apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Complex-valued RoPE for Z-Image."""
    with torch.amp.autocast("cuda", enabled=False):
        x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x * freqs_cis).flatten(3)
        return x_out.type_as(x_in)


# ---------------------------------------------------------------------------
# Concept similarity map extraction
# ---------------------------------------------------------------------------

def _extract_concept_sim_maps(
    query: torch.Tensor,
    key: torch.Tensor,
    hidden_states_out: torch.Tensor,
    token_pos_maps: dict[str, list[int]],
    img_seq_len: int,
    cap_seq_len: int,
    text_start_idx: int,
    top_k_ratio: float = 0.1,
) -> dict[str, torch.Tensor] | None:
    """Extract per-concept spatial similarity maps from cross-attention.

    Algorithm per batch element:

    1. Split unified sequence into image Q and text K
    2. Cross-attention: ``img_Q @ concept_K`` -> softmax -> per-image-token score
    3. Contrastive scoring: ``score *= n_concepts; score -= sum(other_scores)``
    4. Top-k selection -> concept attention: ``core_hidden @ all_img_hidden``
    5. EOS background channel from last caption token

    Args:
        query: ``(B, seq, heads, head_dim)`` post-RoPE query.
        key: ``(B, seq, heads, head_dim)`` post-RoPE key.
        hidden_states_out: ``(B, seq, dim)`` post-SDPA output.
        token_pos_maps: ``{adapter_name: [text_token_indices]}``.
        img_seq_len: Image token count (pre-padding).
        cap_seq_len: Caption token count (pre-padding).
        text_start_idx: Start index of text tokens in the unified sequence
            (= padded image length, since each component is individually
            padded to SEQ_MULTI_OF before concatenation).
        top_k_ratio: Fraction of image tokens for core selection.

    Returns:
        ``{adapter_name: (B, img_seq_len, 1), "__eos__": (B, img_seq_len, 1)}``
        or ``None`` if no maps could be extracted.
    """
    concept_sim_maps: dict[str, torch.Tensor] = {}
    B = query.shape[0]
    total_seq = query.shape[1]

    if img_seq_len > total_seq or text_start_idx > total_seq:
        logger.error(
            "[FreeFuse] Sequence out of bounds: img_len=%d, text_start=%d, seq=%d",
            img_seq_len, text_start_idx, total_seq,
        )
        return None

    # The transformer may truncate/pool the caption before building the
    # unified sequence, so prompt_embeds.shape[1] can be much larger than
    # the actual caption region.  Derive the real caption length from the
    # sequence that the transformer actually produced.
    available_cap = total_seq - text_start_idx
    cap_len_actual = min(cap_seq_len, available_cap)
    if cap_len_actual < 1:
        logger.warning(
            "[FreeFuse] No caption tokens in sequence (text_start=%d, seq=%d)",
            text_start_idx, total_seq,
        )
        return None

    if cap_len_actual != cap_seq_len:
        logger.debug(
            "[FreeFuse] Caption length adjusted: prompt_embeds=%d, actual_in_seq=%d",
            cap_seq_len, cap_len_actual,
        )

    for b in range(B):
        x_len = img_seq_len
        cap_len = cap_len_actual

        # Split unified sequence [image_padded | caption_padded].
        # Image tokens are at [0, x_len); text tokens start at text_start_idx
        # (= padded image length) not at x_len.
        img_query = query[b : b + 1, :x_len]                                    # (1, x_len, H, D)
        img_hidden = hidden_states_out[b : b + 1, :x_len]                       # (1, x_len, dim)
        txt_key = key[b : b + 1, text_start_idx : text_start_idx + cap_len]     # (1, cap_len, H, D)

        scale = 1.0 / 1000.0

        # -- Step 1: cross-attention scores per concept --
        all_cross_attn_scores: dict[str, torch.Tensor] = {}

        for adapter_name, positions in token_pos_maps.items():
            if not positions:
                continue
            # Clamp to valid range within the sliced text key
            valid_pos = [p for p in positions if p < cap_len]
            if not valid_pos:
                continue
            pos_t = torch.tensor(valid_pos, device=query.device)
            concept_key = txt_key[:, pos_t]  # (1, n_concept, H, D)

            weights = torch.einsum(
                "bihd,bjhd->bhij", img_query, concept_key
            ) * scale
            weights = F.softmax(weights, dim=2)
            scores = weights.mean(dim=1).mean(dim=-1)  # (1, img_len)
            all_cross_attn_scores[adapter_name] = scores

        # -- Step 2: contrastive top-k + concept attention --
        n_concepts = len(all_cross_attn_scores)
        for adapter_name in list(all_cross_attn_scores.keys()):
            scores = all_cross_attn_scores[adapter_name] * n_concepts
            for other in all_cross_attn_scores:
                if other != adapter_name:
                    scores = scores - all_cross_attn_scores[other]

            k = max(1, int(x_len * top_k_ratio))
            _, topk_idx = torch.topk(scores, k, dim=-1)

            expanded = topk_idx.unsqueeze(-1).expand(
                -1, -1, img_hidden.shape[-1]
            )
            core = torch.gather(img_hidden, dim=1, index=expanded)

            sim = core @ img_hidden.transpose(-1, -2)       # (1, k, img_len)
            sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (1, img_len, 1)
            sim_map = F.softmax(sim_avg / 4000.0, dim=1)

            if b == 0:
                concept_sim_maps[adapter_name] = sim_map
            else:
                concept_sim_maps[adapter_name] = torch.cat(
                    [concept_sim_maps[adapter_name], sim_map], dim=0
                )

        # -- EOS background: use last caption token --
        eos_pos = torch.tensor([cap_len - 1], device=query.device)
        eos_key = txt_key[:, eos_pos]  # (1, 1, H, D)

        weights = torch.einsum("bihd,bjhd->bhij", img_query, eos_key) * scale
        weights = F.softmax(weights, dim=2)
        eos_scores = weights.mean(dim=1).mean(dim=-1)  # (1, img_len)

        k = max(1, int(x_len * top_k_ratio))
        _, topk_idx = torch.topk(eos_scores, k, dim=-1)

        expanded = topk_idx.unsqueeze(-1).expand(-1, -1, img_hidden.shape[-1])
        core = torch.gather(img_hidden, dim=1, index=expanded)

        sim = core @ img_hidden.transpose(-1, -2)
        sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
        sim_map = F.softmax(sim_avg / 4000.0, dim=1)

        if b == 0:
            concept_sim_maps["__eos__"] = sim_map
        else:
            concept_sim_maps["__eos__"] = torch.cat(
                [concept_sim_maps["__eos__"], sim_map], dim=0
            )

    return concept_sim_maps if concept_sim_maps else None


# ---------------------------------------------------------------------------
# Sim map processing -> binary masks
# ---------------------------------------------------------------------------

def process_sim_maps(
    concept_sim_maps: dict[str, torch.Tensor],
    h_tokens: int,
    w_tokens: int,
    img_token_count: int,
    eos_bg_scale: float = 0.95,
    opening_kernel_size: int = 2,
    closing_kernel_size: int = 2,
) -> dict[str, torch.Tensor]:
    """Convert raw concept similarity maps to binary LoRA masks.

    Pipeline:

    1. Validate maps (NaN check, sync pending CUDA errors)
    2. Pop EOS as background channel, scale by ``eos_bg_scale``
    3. Stack concept maps, argmax with background -> foreground mask
    4. Morphological opening + closing on foreground
    5. Balanced argmax among concepts for spatial ownership
    6. Combine balanced assignment with foreground mask

    Args:
        concept_sim_maps: ``{adapter_name: (B, N, 1), "__eos__": (B, N, 1)}``.
        h_tokens: Height in token space.
        w_tokens: Width in token space.
        img_token_count: Total image token count (h_tokens * w_tokens).
        eos_bg_scale: Scale factor for EOS background channel.
        opening_kernel_size: Kernel for morphological opening.
        closing_kernel_size: Kernel for morphological closing.

    Returns:
        ``{adapter_name: (1, img_token_count)}`` float masks.
    """
    if concept_sim_maps is None:
        return {}

    # Force pending CUDA errors to surface here with a clear traceback,
    # rather than at an unrelated later operation.
    if any(t.is_cuda for t in concept_sim_maps.values()):
        torch.cuda.synchronize()

    # Validate: drop maps that contain NaN (numerical issues in SDPA)
    clean = {}
    for name, tensor in concept_sim_maps.items():
        if torch.isnan(tensor).any():
            logger.warning(
                "[FreeFuse] Dropping sim map '%s': contains NaN values", name,
            )
        else:
            clean[name] = tensor
    concept_sim_maps = clean
    if not concept_sim_maps:
        return {}

    # Pop background channel
    if "__eos__" in concept_sim_maps:
        bg_raw = concept_sim_maps.pop("__eos__")[:, :img_token_count, :]
    else:
        bg_raw = None

    # Stack concept maps -> (B, C, N, 1)
    names = list(concept_sim_maps.keys())
    if not names:
        return {}

    maps_list = [concept_sim_maps[n][:, :img_token_count, :] for n in names]
    stacked = torch.stack(maps_list, dim=1)   # (B, C, N, 1)
    B, C, N, _ = stacked.shape
    squeezed = stacked.squeeze(-1)            # (B, C, N)

    # Build background channel
    if bg_raw is not None:
        bg_channel = bg_raw.squeeze(-1) * eos_bg_scale  # (B, N)
    else:
        bg_channel = squeezed.mean(dim=1)  # fallback: average

    # Argmax with background -> foreground mask
    with_bg = torch.cat([squeezed, bg_channel.unsqueeze(1)], dim=1)  # (B, C+1, N)
    bg_argmax = with_bg.argmax(dim=1)
    raw_fg = (bg_argmax != C).float()

    # Morphological cleanup
    fg_mask = morphological_clean_mask(
        raw_fg, h_tokens, w_tokens,
        opening_kernel_size=opening_kernel_size,
        closing_kernel_size=closing_kernel_size,
    ).bool()

    # Balanced argmax among concepts
    max_indices = stabilized_balanced_argmax(squeezed, h_tokens, w_tokens)

    # Combine: balanced assignment x foreground mask
    lora_masks = {}
    for idx, name in enumerate(names):
        lora_masks[name] = (max_indices == idx).float() * fg_mask.float()

    return lora_masks


# ---------------------------------------------------------------------------
# Morphological cleanup
# ---------------------------------------------------------------------------

def morphological_clean_mask(
    mask: torch.Tensor,
    h: int,
    w: int,
    opening_kernel_size: int = 2,
    closing_kernel_size: int = 2,
) -> torch.Tensor:
    """Apply morphological opening + closing to clean a binary mask.

    Args:
        mask: ``(B, N)`` flat binary mask.
        h: Height in tokens.
        w: Width in tokens.
        opening_kernel_size: Kernel for opening (erode -> dilate).
        closing_kernel_size: Kernel for closing (dilate -> erode).

    Returns:
        Cleaned mask ``(B, N)``.
    """
    B = mask.shape[0]
    mask_2d = mask.view(B, 1, h, w)

    def dilate(x, ks):
        p = ks // 2
        o = F.max_pool2d(x, kernel_size=ks, stride=1, padding=p)
        if o.shape[-2:] != x.shape[-2:]:
            o = F.interpolate(o, size=x.shape[-2:], mode="nearest")
        return o

    def erode(x, ks):
        p = ks // 2
        o = 1.0 - F.max_pool2d(1.0 - x, kernel_size=ks, stride=1, padding=p)
        if o.shape[-2:] != x.shape[-2:]:
            o = F.interpolate(o, size=x.shape[-2:], mode="nearest")
        return o

    # Opening: erode -> dilate (removes small foreground noise)
    if opening_kernel_size > 1:
        mask_2d = dilate(erode(mask_2d, opening_kernel_size), opening_kernel_size)

    # Closing: dilate -> erode (fills small holes)
    if closing_kernel_size > 1:
        mask_2d = erode(dilate(mask_2d, closing_kernel_size), closing_kernel_size)

    return mask_2d.view(B, -1)


# ---------------------------------------------------------------------------
# Stabilized balanced argmax
# ---------------------------------------------------------------------------

def stabilized_balanced_argmax(
    logits: torch.Tensor,
    h: int,
    w: int,
    target_count: int | None = None,
    max_iter: int = 15,
    lr: float = 0.0001,
    gravity_weight: float = 1e-6,
    spatial_weight: float = 4e-5,
    momentum: float = 0.2,
) -> torch.Tensor:
    """Balanced argmax with spatial regularization.

    Iteratively adjusts per-concept biases to balance spatial ownership
    while maintaining spatial coherence through neighbor voting and
    gravity toward concept centroids.

    Args:
        logits: ``(B, C, N)`` concept similarity scores.
        h: Height in tokens.
        w: Width in tokens.
        target_count: Target tokens per concept (default: N/C).
        max_iter: Number of balancing iterations.
        lr: Learning rate for bias adjustment.
        gravity_weight: Weight for centroid gravity.
        spatial_weight: Weight for neighbor voting.
        momentum: EMA momentum for running probabilities.

    Returns:
        ``(B, N)`` concept indices.
    """
    B, C, N = logits.shape
    device = logits.device

    # Spatial grid
    y_range = torch.linspace(-1, 1, steps=h, device=device)
    x_range = torch.linspace(-1, 1, steps=w, device=device)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    flat_y = grid_y.reshape(1, 1, N)
    flat_x = grid_x.reshape(1, 1, N)

    if target_count is None:
        target_count = N / C

    bias = torch.zeros(B, C, 1, device=device)

    def linear_normalize(x, dim=1):
        x_min = x.min(dim=dim, keepdim=True)[0]
        x_max = x.max(dim=dim, keepdim=True)[0]
        return (x - x_min) / (x_max - x_min + 1e-8)

    running_probs = linear_normalize(logits, dim=1)
    logit_range = (logits.max() - logits.min()).item()
    logit_scale = max(logit_range, 1e-4)
    effective_lr = lr * logit_scale
    max_bias = logit_scale * 10.0

    # Neighbor voting kernel
    neighbor_kernel = torch.ones(C, 1, 3, 3, device=device, dtype=logits.dtype) / 8.0
    neighbor_kernel[:, :, 1, 1] = 0

    current_logits = logits.clone()

    for i in range(max_iter):
        probs = linear_normalize(current_logits - bias, dim=1)
        running_probs = (1 - momentum) * probs + momentum * running_probs
        mass = running_probs.sum(dim=2, keepdim=True) + 1e-6
        center_y = (running_probs * flat_y).sum(dim=2, keepdim=True) / mass
        center_x = (running_probs * flat_x).sum(dim=2, keepdim=True) / mass

        dist_sq = (flat_y - center_y) ** 2 + (flat_x - center_x) ** 2

        hard_indices = torch.argmax(current_logits - bias, dim=1)
        hard_indices = hard_indices.clamp(0, C - 1)  # safety: NaN→garbage guard
        hard_counts = F.one_hot(hard_indices, num_classes=C).float().sum(dim=1)

        diff = hard_counts - target_count
        cur_lr = effective_lr * (0.95 ** i)
        bias += torch.sign(diff).unsqueeze(2) * cur_lr
        bias = torch.clamp(bias, -max_bias, max_bias)

        if spatial_weight > 0:
            probs_img = running_probs.view(B, C, h, w).float()
            neighbor_votes = F.conv2d(
                probs_img, neighbor_kernel.float(), padding=1, groups=C
            )
            neighbor_votes = neighbor_votes.to(logits.dtype).view(B, C, N)
        else:
            neighbor_votes = torch.zeros_like(logits)

        current_logits = (
            logits - bias
            + neighbor_votes * spatial_weight
            - dist_sq * gravity_weight
        )

    return torch.argmax(current_logits, dim=1)
