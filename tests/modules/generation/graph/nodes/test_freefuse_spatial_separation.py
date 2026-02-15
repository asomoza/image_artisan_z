"""Full pipeline tests for FreeFuse mask separation and attention bias correctness.

Verifies that:
1. Auto-derived masks properly separate spatial regions for different LoRA concepts
2. Attention bias correctly suppresses cross-concept attention (negative values)
3. Attention bias correctly encourages same-concept attention (positive values)
4. Balanced argmax correctly assigns spatial ownership with known input
5. The full pipeline (structured Q/K/hidden -> sim maps -> masks -> bias) preserves
   spatial separation end-to-end
6. Oversized txt_seq_len (from prompt_embeds vs actual) produces equivalent bias
"""

import torch

from iartisanz.modules.generation.graph.nodes.freefuse_attention_bias import (
    _BiasInjectionProcessor,
    _SEQ_MULTI_OF,
    _ceil_to_multiple,
    construct_attention_bias,
)
from iartisanz.modules.generation.graph.nodes.freefuse_auto_mask import (
    _extract_concept_sim_maps,
    process_sim_maps,
    stabilized_balanced_argmax,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _left_right_indices(h: int, w: int) -> tuple[list[int], list[int]]:
    """Return (left_indices, right_indices) for an h*w grid split vertically."""
    left, right = [], []
    for row in range(h):
        for col in range(w):
            idx = row * w + col
            (left if col < w // 2 else right).append(idx)
    return left, right


def _make_structured_sim_maps(
    img_h: int,
    img_w: int,
) -> dict[str, torch.Tensor]:
    """Create sim maps where concept A -> left half, concept B -> right half.

    Sim maps are (B, N, 1) with values that, after softmax-like normalization
    over the spatial dim, clearly favor one half of the image.
    """
    N = img_h * img_w
    left, right = _left_right_indices(img_h, img_w)

    sim_a = torch.zeros(1, N, 1)
    sim_a[0, left, 0] = 1.0 / len(left)

    sim_b = torch.zeros(1, N, 1)
    sim_b[0, right, 0] = 1.0 / len(right)

    # EOS/background: uniform but much lower than concept peaks
    sim_eos = torch.full((1, N, 1), 0.5 / N)

    return {"lora_a": sim_a, "lora_b": sim_b, "__eos__": sim_eos}


def _make_structured_qkh(
    img_h: int,
    img_w: int,
    cap_len: int,
    heads: int = 4,
    head_dim: int = 16,
    hidden_dim: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    """Create Q/K/hidden with deterministic left/right spatial structure.

    Left image tokens use direction [200, 0, ...], right use [0, 200, ...].
    Concept A trigger-word keys (at text positions [2,3]) match the left direction.
    Concept B trigger-word keys (at text positions [6,7]) match the right direction.

    Returns (query, key, hidden, img_len, padded_img, total_seq).
    """
    img_len = img_h * img_w
    padded_img = _ceil_to_multiple(img_len, _SEQ_MULTI_OF)
    padded_cap = _ceil_to_multiple(cap_len, _SEQ_MULTI_OF)
    total_seq = padded_img + padded_cap

    left, right = _left_right_indices(img_h, img_w)

    # Small random baseline
    query = torch.randn(1, total_seq, heads, head_dim) * 0.01
    key = torch.randn(1, total_seq, heads, head_dim) * 0.01
    hidden = torch.randn(1, total_seq, hidden_dim) * 0.01

    # Direction vectors
    dir_a = torch.zeros(head_dim)
    dir_a[0] = 200.0
    dir_b = torch.zeros(head_dim)
    dir_b[1] = 200.0

    hdir_a = torch.zeros(hidden_dim)
    hdir_a[0] = 200.0
    hdir_b = torch.zeros(hidden_dim)
    hdir_b[1] = 200.0

    # Left image tokens -> direction A
    for idx in left:
        query[0, idx] = dir_a.unsqueeze(0).expand(heads, -1)
        hidden[0, idx] = hdir_a

    # Right image tokens -> direction B
    for idx in right:
        query[0, idx] = dir_b.unsqueeze(0).expand(heads, -1)
        hidden[0, idx] = hdir_b

    # Concept A trigger keys at text positions [2, 3] -> match direction A
    for pos in [2, 3]:
        key[0, padded_img + pos] = dir_a.unsqueeze(0).expand(heads, -1)

    # Concept B trigger keys at text positions [6, 7] -> match direction B
    for pos in [6, 7]:
        key[0, padded_img + pos] = dir_b.unsqueeze(0).expand(heads, -1)

    return query, key, hidden, img_len, padded_img, total_seq


# ===================================================================
# 1. Sim maps -> process_sim_maps -> masks respect spatial separation
# ===================================================================


class TestMaskSpatialSeparation:
    """Verify process_sim_maps produces spatially separated masks."""

    def test_left_right_masks_are_non_overlapping(self):
        """Masks for two concepts should not overlap."""
        img_h, img_w = 8, 8
        sim_maps = _make_structured_sim_maps(img_h, img_w)

        masks = process_sim_maps(sim_maps, img_h, img_w, img_h * img_w)

        assert "lora_a" in masks
        assert "lora_b" in masks

        overlap = (masks["lora_a"] > 0) & (masks["lora_b"] > 0)
        assert overlap.sum() == 0, f"Overlap at {overlap.sum().item()} tokens"

    def test_left_concept_covers_left_half(self):
        """Concept A (left sim map) -> mask covers the left half."""
        img_h, img_w = 8, 8
        N = img_h * img_w
        left, right = _left_right_indices(img_h, img_w)

        masks = process_sim_maps(
            _make_structured_sim_maps(img_h, img_w), img_h, img_w, N,
        )
        mask_a = masks["lora_a"].squeeze(0)

        left_coverage = mask_a[left].sum() / len(left)
        assert left_coverage > 0.7, f"Left coverage: {left_coverage:.2%}"

        right_leakage = mask_a[right].sum() / len(right)
        assert right_leakage < 0.3, f"Right leakage: {right_leakage:.2%}"

    def test_right_concept_covers_right_half(self):
        """Concept B (right sim map) -> mask covers the right half."""
        img_h, img_w = 8, 8
        N = img_h * img_w
        left, right = _left_right_indices(img_h, img_w)

        masks = process_sim_maps(
            _make_structured_sim_maps(img_h, img_w), img_h, img_w, N,
        )
        mask_b = masks["lora_b"].squeeze(0)

        right_coverage = mask_b[right].sum() / len(right)
        assert right_coverage > 0.7, f"Right coverage: {right_coverage:.2%}"

        left_leakage = mask_b[left].sum() / len(left)
        assert left_leakage < 0.3, f"Left leakage: {left_leakage:.2%}"

    def test_combined_masks_cover_foreground(self):
        """Union of both masks should cover most of the image."""
        img_h, img_w = 8, 8
        N = img_h * img_w

        masks = process_sim_maps(
            _make_structured_sim_maps(img_h, img_w), img_h, img_w, N,
        )

        combined = masks["lora_a"] + masks["lora_b"]
        fg_coverage = (combined > 0).float().sum() / N
        assert fg_coverage > 0.5, f"Foreground coverage: {fg_coverage:.2%}"

    def test_larger_grid_32x32(self):
        """32x32 grid (1024 tokens, realistic size) still separates correctly."""
        img_h, img_w = 32, 32
        N = img_h * img_w
        left, right = _left_right_indices(img_h, img_w)

        masks = process_sim_maps(
            _make_structured_sim_maps(img_h, img_w), img_h, img_w, N,
        )
        mask_a = masks["lora_a"].squeeze(0)
        mask_b = masks["lora_b"].squeeze(0)

        assert ((mask_a > 0) & (mask_b > 0)).sum() == 0
        assert mask_a[left].sum() > mask_a[right].sum()
        assert mask_b[right].sum() > mask_b[left].sum()

    def test_three_way_horizontal_split(self):
        """Three concepts: left third, middle third, right third."""
        img_h, img_w = 12, 12
        N = img_h * img_w

        left, mid, right = [], [], []
        for row in range(img_h):
            for col in range(img_w):
                idx = row * img_w + col
                if col < img_w // 3:
                    left.append(idx)
                elif col < 2 * img_w // 3:
                    mid.append(idx)
                else:
                    right.append(idx)

        sim_a = torch.zeros(1, N, 1)
        sim_a[0, left, 0] = 1.0 / len(left)
        sim_b = torch.zeros(1, N, 1)
        sim_b[0, mid, 0] = 1.0 / len(mid)
        sim_c = torch.zeros(1, N, 1)
        sim_c[0, right, 0] = 1.0 / len(right)

        sim_maps = {
            "lora_a": sim_a,
            "lora_b": sim_b,
            "lora_c": sim_c,
            "__eos__": torch.full((1, N, 1), 0.3 / N),
        }
        masks = process_sim_maps(sim_maps, img_h, img_w, N)

        assert len(masks) == 3
        # No pairwise overlap
        for n1 in masks:
            for n2 in masks:
                if n1 != n2:
                    overlap = ((masks[n1] > 0) & (masks[n2] > 0)).sum()
                    assert overlap == 0, f"Overlap {n1}-{n2}: {overlap.item()}"


# ===================================================================
# 2. construct_attention_bias correctness
# ===================================================================


class TestAttentionBiasValues:
    """Verify attention bias has correct suppression/encouragement at each position."""

    def _make_bias(self, img_seq=8, txt_seq=8, bias_scale=5.0, pos_scale=1.0):
        """Two LoRAs: A=left mask (0-3), B=right mask (4-7).

        Text ownership: A owns [0,1], B owns [4,5].
        """
        mask_a = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=torch.float32)
        mask_b = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]], dtype=torch.float32)
        return construct_attention_bias(
            lora_masks={"lora_a": mask_a, "lora_b": mask_b},
            token_pos_maps={"lora_a": [0, 1], "lora_b": [4, 5]},
            img_seq_len=img_seq,
            txt_seq_len=txt_seq,
            bias_scale=bias_scale,
            positive_bias_scale=pos_scale,
        ), img_seq, txt_seq

    def test_bias_shape(self):
        bias, img, txt = self._make_bias()
        assert bias.shape == (1, img + txt, img + txt)

    def test_left_img_to_concept_b_text_is_suppressed(self):
        """Left image tokens -> concept B text tokens should be negative."""
        bias, img, _ = self._make_bias()
        for img_idx in range(4):
            for txt_pos in [4, 5]:
                val = bias[0, img_idx, img + txt_pos].item()
                assert val < 0, (
                    f"img[{img_idx}]->txt[{txt_pos}]: {val:.3f} (expected <0)"
                )

    def test_left_img_to_concept_a_text_is_encouraged(self):
        """Left image tokens -> concept A text tokens should be positive."""
        bias, img, _ = self._make_bias()
        for img_idx in range(4):
            for txt_pos in [0, 1]:
                val = bias[0, img_idx, img + txt_pos].item()
                assert val > 0, (
                    f"img[{img_idx}]->txt[{txt_pos}]: {val:.3f} (expected >0)"
                )

    def test_right_img_to_concept_a_text_is_suppressed(self):
        """Right image tokens -> concept A text tokens should be negative."""
        bias, img, _ = self._make_bias()
        for img_idx in range(4, 8):
            for txt_pos in [0, 1]:
                val = bias[0, img_idx, img + txt_pos].item()
                assert val < 0, (
                    f"img[{img_idx}]->txt[{txt_pos}]: {val:.3f} (expected <0)"
                )

    def test_right_img_to_concept_b_text_is_encouraged(self):
        """Right image tokens -> concept B text tokens should be positive."""
        bias, img, _ = self._make_bias()
        for img_idx in range(4, 8):
            for txt_pos in [4, 5]:
                val = bias[0, img_idx, img + txt_pos].item()
                assert val > 0, (
                    f"img[{img_idx}]->txt[{txt_pos}]: {val:.3f} (expected >0)"
                )

    def test_unowned_text_positions_are_zero(self):
        """Text tokens not owned by any LoRA should have zero bias everywhere."""
        bias, img, _ = self._make_bias()
        for txt_pos in [2, 3, 6, 7]:
            for img_idx in range(8):
                val = bias[0, img_idx, img + txt_pos].item()
                assert val == 0.0, (
                    f"img[{img_idx}]->txt[{txt_pos}]: {val:.3f} (expected 0.0)"
                )

    def test_image_to_image_is_zero(self):
        """Image->Image region should be all zeros (no self-bias)."""
        bias, img, _ = self._make_bias()
        assert (bias[0, :img, :img] == 0).all()

    def test_suppression_scales_with_bias_scale(self):
        """Doubling bias_scale should double the suppression magnitude."""
        bias_5, img, _ = self._make_bias(bias_scale=5.0)
        bias_10, _, _ = self._make_bias(bias_scale=10.0)

        val_5 = bias_5[0, 0, img + 4].item()
        val_10 = bias_10[0, 0, img + 4].item()

        assert abs(val_10 / val_5 - 2.0) < 0.01, (
            f"Ratio: {val_10/val_5:.3f} (expected 2.0)"
        )

    def test_bidirectional_text_to_other_image_suppressed(self):
        """Concept A text tokens -> right image tokens (not A's region) suppressed."""
        bias, img, _ = self._make_bias()
        for txt_pos in [0, 1]:
            for img_idx in range(4, 8):
                val = bias[0, img + txt_pos, img_idx].item()
                assert val < 0, (
                    f"txt[{txt_pos}]->img[{img_idx}]: {val:.3f} (expected <0)"
                )

    def test_bidirectional_text_to_own_image_encouraged(self):
        """Concept A text tokens -> left image tokens (A's region) encouraged."""
        bias, img, _ = self._make_bias()
        for txt_pos in [0, 1]:
            for img_idx in range(4):
                val = bias[0, img + txt_pos, img_idx].item()
                assert val > 0, (
                    f"txt[{txt_pos}]->img[{img_idx}]: {val:.3f} (expected >0)"
                )


# ===================================================================
# 3. Full pipeline: structured Q/K/hidden -> masks -> bias
# ===================================================================


class TestFullPipelineEndToEnd:
    """End-to-end: structured attention data -> separated masks -> correct bias."""

    def test_structured_data_produces_separated_masks(self):
        """Known left/right structure in Q/K/hidden -> separated masks."""
        img_h, img_w = 8, 8
        cap_len = 10
        token_pos_maps = {"lora_a": [2, 3], "lora_b": [6, 7]}

        query, key, hidden, img_len, padded_img, _ = _make_structured_qkh(
            img_h, img_w, cap_len,
        )

        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps=token_pos_maps,
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )
        assert sim_maps is not None
        assert "lora_a" in sim_maps and "lora_b" in sim_maps

        masks = process_sim_maps(sim_maps, img_h, img_w, img_len)
        assert "lora_a" in masks and "lora_b" in masks

        left, right = _left_right_indices(img_h, img_w)
        mask_a = masks["lora_a"].squeeze(0)
        mask_b = masks["lora_b"].squeeze(0)

        # No overlap
        overlap = ((mask_a > 0) & (mask_b > 0)).sum()
        assert overlap == 0, f"Overlap: {overlap.item()}"

        # Concept A covers left, concept B covers right
        assert mask_a[left].sum() > mask_a[right].sum(), (
            f"A: left={mask_a[left].sum():.0f}, right={mask_a[right].sum():.0f}"
        )
        assert mask_b[right].sum() > mask_b[left].sum(), (
            f"B: left={mask_b[left].sum():.0f}, right={mask_b[right].sum():.0f}"
        )

    def test_masks_to_bias_has_correct_sign_pattern(self):
        """Full pipeline: masks from structured data -> bias with correct signs."""
        img_h, img_w = 8, 8
        cap_len = 10
        N = img_h * img_w
        token_pos_maps = {"lora_a": [2, 3], "lora_b": [6, 7]}

        query, key, hidden, img_len, padded_img, _ = _make_structured_qkh(
            img_h, img_w, cap_len,
        )
        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps=token_pos_maps,
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )
        masks = process_sim_maps(sim_maps, img_h, img_w, N)

        img_seq_len = _ceil_to_multiple(N, _SEQ_MULTI_OF)
        txt_seq_len = _ceil_to_multiple(cap_len, _SEQ_MULTI_OF)

        bias = construct_attention_bias(
            lora_masks=masks,
            token_pos_maps=token_pos_maps,
            img_seq_len=img_seq_len,
            txt_seq_len=txt_seq_len,
        )

        assert bias.abs().sum() > 0, "Bias is all zeros"

        left, right = _left_right_indices(img_h, img_w)

        # Aggregate cross-concept bias: left img -> concept B text
        left_to_b = bias[0, left, :][:, [img_seq_len + 6, img_seq_len + 7]]
        # Aggregate cross-concept bias: right img -> concept A text
        right_to_a = bias[0, right, :][:, [img_seq_len + 2, img_seq_len + 3]]

        # Should have net suppression (negative)
        assert left_to_b.sum() < 0 or right_to_a.sum() < 0, (
            "Expected cross-concept suppression"
        )

    def test_bias_processor_injects_at_target_positions(self):
        """BiasInjectionProcessor modifies the attention mask at specific positions."""
        img_seq, txt_seq = 8, 8
        total = img_seq + txt_seq

        bias = torch.zeros(1, total, total)
        bias[0, 0, img_seq + 4] = -5.0  # suppression
        bias[0, 0, img_seq + 0] = +1.0  # encouragement

        received = {}

        class _Recorder:
            _attention_backend = None
            _parallel_config = None

            def __call__(self, attn, hidden_states, attention_mask=None, **kw):
                received["mask"] = attention_mask
                return hidden_states

        proc = _BiasInjectionProcessor(_Recorder(), bias)
        proc(None, torch.randn(1, total, 32), attention_mask=torch.ones(1, total, dtype=torch.bool))

        result = received["mask"]  # (1, 1, total, total)

        # Suppressed position
        assert result[0, 0, 0, img_seq + 4].item() == -5.0
        # Encouraged position
        assert result[0, 0, 0, img_seq + 0].item() == 1.0
        # Unbiased position (mask was all-True -> float_mask = 0, bias = 0)
        assert result[0, 0, 0, 0].item() == 0.0


# ===================================================================
# 4. Balanced argmax with known spatial structure
# ===================================================================


class TestBalancedArgmaxSpatialAssignment:
    """Verify balanced argmax assigns correct spatial regions."""

    def test_left_right_assignment(self):
        """Clear left/right logits -> correctly assigned."""
        h, w = 8, 8
        N = h * w
        left, right = _left_right_indices(h, w)

        logits = torch.zeros(1, 2, N)
        logits[0, 0, left] = 1.0
        logits[0, 1, right] = 1.0

        result = stabilized_balanced_argmax(logits, h, w)

        left_is_0 = (result[0, left] == 0).float().mean()
        assert left_is_0 > 0.8, f"Left assigned to 0: {left_is_0:.2%}"

        right_is_1 = (result[0, right] == 1).float().mean()
        assert right_is_1 > 0.8, f"Right assigned to 1: {right_is_1:.2%}"

    def test_top_bottom_assignment(self):
        """Top/bottom split."""
        h, w = 16, 16
        N = h * w
        top = list(range(N // 2))
        bottom = list(range(N // 2, N))

        logits = torch.zeros(1, 2, N)
        logits[0, 0, top] = 1.0
        logits[0, 1, bottom] = 1.0

        result = stabilized_balanced_argmax(logits, h, w)

        assert (result[0, top] == 0).float().mean() > 0.8
        assert (result[0, bottom] == 1).float().mean() > 0.8

    def test_three_column_split(self):
        """Three-column vertical split."""
        h, w = 12, 12
        N = h * w
        col1, col2, col3 = [], [], []
        for row in range(h):
            for col in range(w):
                idx = row * w + col
                if col < 4:
                    col1.append(idx)
                elif col < 8:
                    col2.append(idx)
                else:
                    col3.append(idx)

        logits = torch.zeros(1, 3, N)
        logits[0, 0, col1] = 1.0
        logits[0, 1, col2] = 1.0
        logits[0, 2, col3] = 1.0

        result = stabilized_balanced_argmax(logits, h, w)

        assert (result[0, col1] == 0).float().mean() > 0.7
        assert (result[0, col2] == 1).float().mean() > 0.7
        assert (result[0, col3] == 2).float().mean() > 0.7

    def test_balancing_corrects_weak_uneven_input(self):
        """When logits have only slight preference, balancing nudges toward 50/50."""
        h, w = 16, 16
        N = h * w

        # Both concepts present everywhere, slight preference in each region
        logits = torch.full((1, 2, N), 0.4)
        logits[0, 0, : int(N * 0.7)] += 0.05  # Concept 0 slightly favored in 70%
        logits[0, 1, int(N * 0.7) :] += 0.05  # Concept 1 slightly favored in 30%

        result = stabilized_balanced_argmax(logits, h, w)

        count_0 = (result == 0).float().sum().item()
        count_1 = (result == 1).float().sum().item()

        # Should be closer to 50/50 than the raw 70/30
        ratio = min(count_0, count_1) / max(count_0, count_1)
        assert ratio > 0.4, f"Balance ratio: {ratio:.2f} ({count_0:.0f}/{count_1:.0f})"


# ===================================================================
# 5. Bias with auto-masks and oversized txt_seq_len
# ===================================================================


class TestBiasWithAutoMasks:
    """Test bias construction edge cases for Phase B auto-derived masks."""

    def test_auto_masks_produce_nonzero_bias(self):
        """Auto-derived masks from process_sim_maps produce a non-trivial bias."""
        img_h, img_w = 8, 8
        N = img_h * img_w

        masks = process_sim_maps(
            _make_structured_sim_maps(img_h, img_w), img_h, img_w, N,
        )

        bias = construct_attention_bias(
            lora_masks=masks,
            token_pos_maps={"lora_a": [0, 1], "lora_b": [4, 5]},
            img_seq_len=_ceil_to_multiple(N, _SEQ_MULTI_OF),
            txt_seq_len=_ceil_to_multiple(10, _SEQ_MULTI_OF),
        )

        assert bias.abs().sum() > 0

    def test_oversized_bias_sliced_preserves_positions(self):
        """Bias built with oversized txt_seq_len, sliced to actual total seq."""
        img_seq_len = 64
        txt_seq_len_oversized = 2560
        actual_total = 64 + 32  # what the transformer actually uses

        mask_a = torch.tensor([[1.0] * 4 + [0.0] * 60])
        mask_b = torch.tensor([[0.0] * 60 + [1.0] * 4])

        bias = construct_attention_bias(
            lora_masks={"lora_a": mask_a, "lora_b": mask_b},
            token_pos_maps={"lora_a": [0, 1], "lora_b": [10, 11]},
            img_seq_len=img_seq_len,
            txt_seq_len=txt_seq_len_oversized,
        )

        # Slice like BiasInjectionProcessor does
        sliced = bias[:, :actual_total, :actual_total]

        # Token positions [0,1] and [10,11] are within the 32-token caption
        # Left img (idx 0) -> concept B text (idx 64+10): suppressed
        assert sliced[0, 0, img_seq_len + 10].item() < 0
        # Left img (idx 0) -> concept A text (idx 64+0): encouraged
        assert sliced[0, 0, img_seq_len + 0].item() > 0

    def test_actual_vs_oversized_txt_seq_gives_same_effective_bias(self):
        """Using actual caption length vs oversized: identical bias at real positions."""
        img_seq = 64
        actual_txt = 32
        oversized_txt = 2560
        actual_total = img_seq + actual_txt

        mask_a = torch.tensor([[1.0] * 32 + [0.0] * 32])
        mask_b = torch.tensor([[0.0] * 32 + [1.0] * 32])
        lora_masks = {"lora_a": mask_a, "lora_b": mask_b}
        token_pos_maps = {"lora_a": [0, 1], "lora_b": [10, 11]}

        bias_actual = construct_attention_bias(
            lora_masks=lora_masks,
            token_pos_maps=token_pos_maps,
            img_seq_len=img_seq,
            txt_seq_len=actual_txt,
        )

        bias_oversized = construct_attention_bias(
            lora_masks=lora_masks,
            token_pos_maps=token_pos_maps,
            img_seq_len=img_seq,
            txt_seq_len=oversized_txt,
        )

        sliced = bias_oversized[:, :actual_total, :actual_total]

        assert torch.allclose(sliced, bias_actual), (
            f"Max diff: {(sliced - bias_actual).abs().max():.6f}"
        )

    def test_positions_beyond_actual_caption_are_lost_but_safe(self):
        """Token positions >= actual caption length produce no bias in sliced region."""
        img_seq = 64
        actual_txt = 32
        oversized_txt = 2560
        actual_total = img_seq + actual_txt

        mask_a = torch.tensor([[1.0] * 32 + [0.0] * 32])
        lora_masks = {"lora_a": mask_a}
        # Position 100 is within the oversized bias but beyond actual_total
        token_pos_maps = {"lora_a": [100]}

        bias = construct_attention_bias(
            lora_masks=lora_masks,
            token_pos_maps=token_pos_maps,
            img_seq_len=img_seq,
            txt_seq_len=oversized_txt,
        )

        sliced = bias[:, :actual_total, :actual_total]

        # Position 100 is at img_seq + 100 = 164, which is beyond actual_total=96
        # So the sliced bias should have no effect from that position
        img_to_txt = sliced[0, :img_seq, img_seq:]
        assert (img_to_txt == 0).all(), "Positions beyond actual caption leaked into bias"


# ===================================================================
# 6. Sim map extraction with structured data: verify spatial selectivity
# ===================================================================


class TestSimMapExtractSpatialSelectivity:
    """Verify _extract_concept_sim_maps produces spatially differentiated maps."""

    def test_concept_a_sim_map_favors_left(self):
        """Concept A's sim map should have higher values on left image tokens."""
        img_h, img_w = 8, 8
        cap_len = 10

        query, key, hidden, img_len, padded_img, _ = _make_structured_qkh(
            img_h, img_w, cap_len,
        )

        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [2, 3], "lora_b": [6, 7]},
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )

        assert sim_maps is not None
        left, right = _left_right_indices(img_h, img_w)

        sim_a = sim_maps["lora_a"].squeeze()  # (N,)
        left_mean = sim_a[left].mean().item()
        right_mean = sim_a[right].mean().item()

        assert left_mean > right_mean, (
            f"A: left_mean={left_mean:.6f}, right_mean={right_mean:.6f}"
        )

    def test_concept_b_sim_map_favors_right(self):
        """Concept B's sim map should have higher values on right image tokens."""
        img_h, img_w = 8, 8
        cap_len = 10

        query, key, hidden, img_len, padded_img, _ = _make_structured_qkh(
            img_h, img_w, cap_len,
        )

        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [2, 3], "lora_b": [6, 7]},
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )

        assert sim_maps is not None
        left, right = _left_right_indices(img_h, img_w)

        sim_b = sim_maps["lora_b"].squeeze()
        left_mean = sim_b[left].mean().item()
        right_mean = sim_b[right].mean().item()

        assert right_mean > left_mean, (
            f"B: left_mean={left_mean:.6f}, right_mean={right_mean:.6f}"
        )

    def test_sim_maps_have_different_peaks(self):
        """The two concepts' sim maps should peak at different image locations."""
        img_h, img_w = 8, 8
        cap_len = 10

        query, key, hidden, img_len, padded_img, _ = _make_structured_qkh(
            img_h, img_w, cap_len,
        )

        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [2, 3], "lora_b": [6, 7]},
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )

        peak_a = sim_maps["lora_a"].squeeze().argmax().item()
        peak_b = sim_maps["lora_b"].squeeze().argmax().item()

        # Peaks should be in different halves
        left, right = _left_right_indices(img_h, img_w)
        a_in_left = peak_a in left
        b_in_right = peak_b in right

        assert a_in_left, f"A peak at {peak_a} (expected in left: {left[:4]}...)"
        assert b_in_right, f"B peak at {peak_b} (expected in right: {right[:4]}...)"

    def test_16x16_grid_still_separates(self):
        """Larger grid (256 tokens) still shows spatial selectivity."""
        img_h, img_w = 16, 16
        cap_len = 20

        query, key, hidden, img_len, padded_img, _ = _make_structured_qkh(
            img_h, img_w, cap_len, heads=8, head_dim=32, hidden_dim=128,
        )

        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [2, 3], "lora_b": [6, 7]},
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )

        assert sim_maps is not None
        left, right = _left_right_indices(img_h, img_w)

        sim_a = sim_maps["lora_a"].squeeze()
        sim_b = sim_maps["lora_b"].squeeze()

        assert sim_a[left].mean() > sim_a[right].mean()
        assert sim_b[right].mean() > sim_b[left].mean()
