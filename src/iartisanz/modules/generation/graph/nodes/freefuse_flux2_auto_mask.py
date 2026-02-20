"""FreeFuse Phase B: Auto-mask extraction from cross-attention similarity maps for Flux2.

When LoRAs have trigger words but no user-painted spatial masks, this module
derives masks automatically using a two-phase denoise approach:

1. Phase 1: A few denoise steps **without LoRA** collect cross-attention
   similarity maps showing which image regions correspond to which trigger words.
2. The maps are processed into binary spatial masks via background thresholding,
   morphological cleanup, and balanced argmax assignment.
3. Phase 2 (in denoise node): Full denoise with LoRA + derived masks for
   attention bias, reusing Phase A's FreeFuse infrastructure.

Based on FreeFuse (arXiv: 2510.23515v2).

Flux2 sequence layout: [text_tokens | image_tokens]
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sim map collector for double-stream blocks (Flux2Attention)
# ---------------------------------------------------------------------------

class _Flux2SimMapCollectorProcessor:
    """Replacement processor for Flux2 double-stream blocks that collects sim maps.

    Replicates standard Flux2AttnProcessor logic while capturing Q, K, and
    hidden_states for concept similarity map extraction.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        original_processor,
        token_pos_maps: dict[str, list[int]],
        txt_seq_len: int,
        top_k_ratio: float = 0.1,
        eos_token_index: int | None = None,
    ):
        self._original = original_processor
        self.token_pos_maps = token_pos_maps
        self.txt_seq_len = txt_seq_len
        self.top_k_ratio = top_k_ratio
        self.eos_token_index = eos_token_index
        self.cal_concept_sim_map = False
        self.concept_sim_maps: dict[str, torch.Tensor] | None = None
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
        """Forward pass replicating Flux2AttnProcessor with sim-map collection."""
        # QKV projections
        if attn.fused_projections:
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
                enc_query, enc_key, enc_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
            else:
                enc_query = enc_key = enc_value = None
        else:
            query = attn.to_q(hidden_states)
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
                enc_query = attn.add_q_proj(encoder_hidden_states)
                enc_key = attn.add_k_proj(encoder_hidden_states)
                enc_value = attn.add_v_proj(encoder_hidden_states)
            else:
                enc_query = enc_key = enc_value = None

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if enc_query is not None:
            enc_query = enc_query.unflatten(-1, (attn.heads, -1))
            enc_key = enc_key.unflatten(-1, (attn.heads, -1))
            enc_value = enc_value.unflatten(-1, (attn.heads, -1))
            enc_query = attn.norm_added_q(enc_query)
            enc_key = attn.norm_added_k(enc_key)

            # Concatenate: [text, image]
            txt_img_query = torch.cat([enc_query, query], dim=1)
            txt_img_key = torch.cat([enc_key, key], dim=1)
            txt_img_value = torch.cat([enc_value, value], dim=1)
        else:
            txt_img_query = query
            txt_img_key = key
            txt_img_value = value

        # Apply RoPE
        if image_rotary_emb is not None:
            txt_img_query = apply_rotary_emb(txt_img_query, image_rotary_emb, sequence_dim=1)
            txt_img_key = apply_rotary_emb(txt_img_key, image_rotary_emb, sequence_dim=1)

        # Attention mask handling
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        # SDPA
        attn_output = dispatch_attention_fn(
            txt_img_query,
            txt_img_key,
            txt_img_value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        if encoder_hidden_states is not None:
            txt_len = encoder_hidden_states.shape[1]
            img_len = hidden_states.shape[1]

            enc_out, img_out = attn_output.split_with_sizes([txt_len, img_len], dim=1)

            # Extract concept sim maps on trigger step
            if self.cal_concept_sim_map and self.token_pos_maps:
                enc_key_rope = txt_img_key[:, :txt_len, :, :]
                img_query_rope = txt_img_query[:, txt_len:, :, :]

                self.concept_sim_maps = _extract_concept_sim_maps(
                    img_query_rope, enc_key_rope, img_out,
                    self.token_pos_maps, self.top_k_ratio,
                    self.eos_token_index,
                )

            # Output projections
            img_out = attn.to_out[0](img_out)
            img_out = attn.to_out[1](img_out)
            enc_out = attn.to_add_out(enc_out)

            return img_out, enc_out
        else:
            return attn_output


# ---------------------------------------------------------------------------
# Sim map collector for single-stream blocks (Flux2ParallelSelfAttention)
# ---------------------------------------------------------------------------

class _Flux2SingleSimMapCollectorProcessor:
    """Replacement processor for Flux2 single-stream blocks that collects sim maps.

    Single-stream blocks operate on the already-concatenated [text | image] sequence.
    """

    _attention_backend = None
    _parallel_config = None

    def __init__(
        self,
        original_processor,
        token_pos_maps: dict[str, list[int]],
        txt_seq_len: int,
        top_k_ratio: float = 0.1,
        eos_token_index: int | None = None,
    ):
        self._original = original_processor
        self.token_pos_maps = token_pos_maps
        self.txt_seq_len = txt_seq_len
        self.top_k_ratio = top_k_ratio
        self.eos_token_index = eos_token_index
        self.cal_concept_sim_map = False
        self.concept_sim_maps: dict[str, torch.Tensor] | None = None
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
        """Forward pass replicating Flux2SingleStreamAttnProcessor with sim-map collection."""
        # Fused QKV + MLP projection
        proj_output = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            proj_output,
            [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor],
            dim=-1,
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]

        attn_output = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        attn_output = attn_output.flatten(2, 3)
        attn_output = attn_output.to(query.dtype)

        # Extract sim maps
        if self.cal_concept_sim_map and self.token_pos_maps and self.txt_seq_len > 0:
            txt_len = self.txt_seq_len
            txt_key = key[:, :txt_len, :, :]
            img_query = query[:, txt_len:, :, :]
            img_attn_out = attn_output[:, txt_len:, :]

            self.concept_sim_maps = _extract_concept_sim_maps(
                img_query, txt_key, img_attn_out,
                self.token_pos_maps, self.top_k_ratio,
                self.eos_token_index,
            )

        # MLP activation
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Fused output projection
        combined_output = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        output = attn.to_out(combined_output)

        return output


# ---------------------------------------------------------------------------
# Concept similarity map extraction (shared by both block types)
# ---------------------------------------------------------------------------

def _extract_concept_sim_maps(
    img_query: torch.Tensor,
    txt_key: torch.Tensor,
    img_hidden_out: torch.Tensor,
    token_pos_maps: dict[str, list[int]],
    top_k_ratio: float,
    eos_token_index: int | None,
) -> dict[str, torch.Tensor] | None:
    """Extract per-concept spatial similarity maps from cross-attention.

    Algorithm:
    1. Cross-attention: img_query @ concept_text_key -> per-image-token scores
    2. Contrastive scoring: score *= n_concepts; score -= sum(other_scores)
    3. Top-k selection -> concept attention: core_hidden @ all_img_hidden
    4. EOS background channel

    Args:
        img_query: (B, img_len, heads, head_dim) post-RoPE image query.
        txt_key: (B, txt_len, heads, head_dim) post-RoPE text key.
        img_hidden_out: (B, img_len, dim) post-attention image output.
        token_pos_maps: {adapter_name: [text_token_indices]}.
        top_k_ratio: Fraction of image tokens for core selection.
        eos_token_index: Index of EOS token for background detection.

    Returns:
        {adapter_name: (B, img_len, 1), "__eos__": (B, img_len, 1)} or None.
    """
    concept_sim_maps: dict[str, torch.Tensor] = {}
    img_len = img_query.shape[1]
    txt_len = txt_key.shape[1]
    scale = 1.0 / 1000.0

    # Step 1: Cross-attention scores per concept
    all_cross_attn_scores: dict[str, torch.Tensor] = {}

    for adapter_name, positions in token_pos_maps.items():
        if not positions:
            continue
        valid_pos = [p for p in positions if p < txt_len]
        if not valid_pos:
            continue
        pos_t = torch.tensor(valid_pos, device=txt_key.device)
        concept_key = txt_key[:, pos_t]  # (B, n_concept, H, D)

        weights = torch.einsum("bihd,bjhd->bhij", img_query, concept_key) * scale
        weights = F.softmax(weights, dim=2)
        scores = weights.mean(dim=1).mean(dim=-1)  # (B, img_len)
        all_cross_attn_scores[adapter_name] = scores

    if not all_cross_attn_scores:
        return None

    # Step 2: Contrastive top-k + concept attention
    n_concepts = len(all_cross_attn_scores)
    for adapter_name in list(all_cross_attn_scores.keys()):
        scores = all_cross_attn_scores[adapter_name] * n_concepts
        for other in all_cross_attn_scores:
            if other != adapter_name:
                scores = scores - all_cross_attn_scores[other]

        k = max(1, int(img_len * top_k_ratio))
        _, topk_idx = torch.topk(scores, k, dim=-1)

        expanded = topk_idx.unsqueeze(-1).expand(-1, -1, img_hidden_out.shape[-1])
        core = torch.gather(img_hidden_out, dim=1, index=expanded)

        sim = core @ img_hidden_out.transpose(-1, -2)  # (B, k, img_len)
        sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)  # (B, img_len, 1)
        sim_map = F.softmax(sim_avg / 4000.0, dim=1)
        concept_sim_maps[adapter_name] = sim_map

    # EOS background
    if eos_token_index is not None and eos_token_index < txt_len:
        eos_pos = torch.tensor([eos_token_index], device=txt_key.device)
        eos_key = txt_key[:, eos_pos]

        weights = torch.einsum("bihd,bjhd->bhij", img_query, eos_key) * scale
        weights = F.softmax(weights, dim=2)
        eos_scores = weights.mean(dim=1).mean(dim=-1)

        k = max(1, int(img_len * top_k_ratio))
        _, topk_idx = torch.topk(eos_scores, k, dim=-1)

        expanded = topk_idx.unsqueeze(-1).expand(-1, -1, img_hidden_out.shape[-1])
        core = torch.gather(img_hidden_out, dim=1, index=expanded)

        sim = core @ img_hidden_out.transpose(-1, -2)
        sim_avg = sim.mean(dim=1, keepdim=True).transpose(1, 2)
        concept_sim_maps["__eos__"] = F.softmax(sim_avg / 4000.0, dim=1)

    return concept_sim_maps if concept_sim_maps else None


# ---------------------------------------------------------------------------
# Install / remove collector
# ---------------------------------------------------------------------------

def install_flux2_sim_map_collector(
    transformer,
    block_idx: int,
    token_pos_maps: dict[str, list[int]],
    txt_seq_len: int,
    top_k_ratio: float = 0.1,
    use_single_stream: bool = False,
) -> dict:
    """Install a similarity map collector on one Flux2 transformer block.

    Args:
        transformer: The Flux2 transformer model.
        block_idx: Which block to install the collector on.
        token_pos_maps: {adapter_name: [text_token_indices]}.
        txt_seq_len: Number of text tokens.
        top_k_ratio: Fraction of image tokens for core selection.
        use_single_stream: If True, install on single_transformer_blocks instead
            of transformer_blocks (double-stream).

    Returns:
        State dict for removal via remove_flux2_sim_map_collector().
    """
    # Find EOS token index (last occupied text token position)
    tokenizer = None  # Not needed — we use the positions directly
    eos_idx = None
    # Simple heuristic: use text_seq_len - 1 as fallback EOS position
    if txt_seq_len > 0:
        eos_idx = txt_seq_len - 1

    if use_single_stream:
        blocks = transformer.single_transformer_blocks
        block = blocks[block_idx]
        attn_module = block.attn
        original_proc = attn_module.processor

        collector = _Flux2SingleSimMapCollectorProcessor(
            original_proc,
            token_pos_maps=token_pos_maps,
            txt_seq_len=txt_seq_len,
            top_k_ratio=top_k_ratio,
            eos_token_index=eos_idx,
        )
    else:
        blocks = transformer.transformer_blocks
        block = blocks[block_idx]
        attn_module = block.attn
        original_proc = attn_module.processor

        collector = _Flux2SimMapCollectorProcessor(
            original_proc,
            token_pos_maps=token_pos_maps,
            txt_seq_len=txt_seq_len,
            top_k_ratio=top_k_ratio,
            eos_token_index=eos_idx,
        )

    attn_module.processor = collector

    return {
        "attn_module": attn_module,
        "original_proc": original_proc,
        "collector": collector,
    }


def remove_flux2_sim_map_collector(state: dict) -> None:
    """Restore original attention processor."""
    state["attn_module"].processor = state["original_proc"]
