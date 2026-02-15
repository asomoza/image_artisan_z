"""Tests for the FreeFuse Phase B auto-mask extraction pipeline.

Exercises ``_extract_concept_sim_maps`` -> ``process_sim_maps`` end-to-end
on CPU to catch index-out-of-bounds errors that would manifest as CUDA
assertions on GPU.
"""

import torch
import torch.nn.functional as F

from iartisanz.modules.generation.graph.nodes.freefuse_auto_mask import (
    _extract_concept_sim_maps,
    morphological_clean_mask,
    process_sim_maps,
    stabilized_balanced_argmax,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sim_map_inputs(
    img_h: int,
    img_w: int,
    cap_len: int,
    heads: int = 8,
    head_dim: int = 16,
    hidden_dim: int = 128,
    seq_multi_of: int = 32,
):
    """Create realistic query/key/hidden_states for sim map extraction."""
    img_len = img_h * img_w
    padded_img = img_len + (-img_len) % seq_multi_of
    padded_cap = cap_len + (-cap_len) % seq_multi_of
    total_seq = padded_img + padded_cap

    query = torch.randn(1, total_seq, heads, head_dim)
    key = torch.randn(1, total_seq, heads, head_dim)
    hidden = torch.randn(1, total_seq, hidden_dim)

    return query, key, hidden, img_len, padded_img, total_seq


# ===================================================================
# End-to-end pipeline tests
# ===================================================================

class TestExtractAndProcessSimMaps:
    """End-to-end tests for _extract_concept_sim_maps + process_sim_maps."""

    def _run_pipeline(self, img_h, img_w, cap_len, token_pos_maps):
        """Run full extract + process pipeline and return masks."""
        query, key, hidden, img_len, padded_img, _ = _make_sim_map_inputs(
            img_h, img_w, cap_len,
        )

        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps=token_pos_maps,
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )

        if sim_maps is None:
            return None

        masks = process_sim_maps(
            sim_maps, img_h, img_w, img_len,
        )
        return masks

    def test_basic_two_concepts_1024x1024(self):
        """Standard 1024x1024: img_h=64, img_w=64, 4096 tokens (mult of 32)."""
        masks = self._run_pipeline(
            img_h=64, img_w=64, cap_len=77,
            token_pos_maps={"lora_a": [5, 6], "lora_b": [10, 11]},
        )
        assert masks is not None
        assert "lora_a" in masks
        assert "lora_b" in masks
        for name, mask in masks.items():
            assert mask.shape == (1, 4096), f"{name}: {mask.shape}"
            assert not torch.isnan(mask).any(), f"{name} contains NaN"

    def test_basic_two_concepts_512x512(self):
        """512x512: img_h=32, img_w=32, 1024 tokens."""
        masks = self._run_pipeline(
            img_h=32, img_w=32, cap_len=50,
            token_pos_maps={"lora_a": [2], "lora_b": [8]},
        )
        assert masks is not None
        for mask in masks.values():
            assert mask.shape == (1, 1024)

    def test_non_multiple_of_32_image_tokens(self):
        """320x320: img_h=20, img_w=20, 400 tokens (NOT a multiple of 32).

        This is the key regression case — text_start_idx must be 416
        (padded), not 400 (raw).
        """
        masks = self._run_pipeline(
            img_h=20, img_w=20, cap_len=77,
            token_pos_maps={"lora_a": [3, 4], "lora_b": [15, 16]},
        )
        assert masks is not None
        for mask in masks.values():
            assert mask.shape == (1, 400)

    def test_non_multiple_of_32_448x448(self):
        """448x448: img_h=28, img_w=28, 784 tokens (784 % 32 = 16)."""
        masks = self._run_pipeline(
            img_h=28, img_w=28, cap_len=100,
            token_pos_maps={"lora_a": [0, 1, 2], "lora_b": [50, 51]},
        )
        assert masks is not None
        for mask in masks.values():
            assert mask.shape == (1, 784)

    def test_single_concept(self):
        """Only one LoRA — still produces a mask (no contrastive scoring)."""
        masks = self._run_pipeline(
            img_h=32, img_w=32, cap_len=50,
            token_pos_maps={"lora_a": [5]},
        )
        assert masks is not None
        assert "lora_a" in masks
        assert masks["lora_a"].shape == (1, 1024)

    def test_small_image(self):
        """Very small image: 8x8 = 64 tokens."""
        masks = self._run_pipeline(
            img_h=8, img_w=8, cap_len=20,
            token_pos_maps={"lora_a": [0], "lora_b": [5]},
        )
        assert masks is not None
        for mask in masks.values():
            assert mask.shape == (1, 64)

    def test_large_caption(self):
        """Long prompt: 512 tokens."""
        masks = self._run_pipeline(
            img_h=32, img_w=32, cap_len=512,
            token_pos_maps={"lora_a": [100, 101], "lora_b": [400, 401]},
        )
        assert masks is not None
        for mask in masks.values():
            assert mask.shape == (1, 1024)

    def test_non_square_image(self):
        """768x512: img_h=48, img_w=32, 1536 tokens."""
        masks = self._run_pipeline(
            img_h=48, img_w=32, cap_len=77,
            token_pos_maps={"lora_a": [2, 3], "lora_b": [10]},
        )
        assert masks is not None
        for mask in masks.values():
            assert mask.shape == (1, 1536)

    def test_three_concepts(self):
        """Three LoRAs — tests balanced argmax with C=3."""
        masks = self._run_pipeline(
            img_h=32, img_w=32, cap_len=77,
            token_pos_maps={
                "lora_a": [2], "lora_b": [10], "lora_c": [20],
            },
        )
        assert masks is not None
        assert len(masks) == 3

    def test_caption_truncated_by_transformer(self):
        """Transformer truncates caption to much shorter than prompt_embeds.

        Reproduces the real-world scenario where prompt_embeds has 2560
        tokens but the transformer's unified sequence only has 32 caption
        tokens (total_seq=4128, padded_img=4096).
        """
        img_h, img_w = 64, 64
        img_len = img_h * img_w  # 4096
        padded_img = 4096  # already multiple of 32

        # Transformer only uses 32 caption tokens (padded from ~1-32 actual)
        actual_cap_in_seq = 32
        total_seq = padded_img + actual_cap_in_seq  # 4128

        # But prompt_embeds says 2560 tokens
        cap_len_from_embeds = 2560

        heads, head_dim, hidden_dim = 8, 16, 128
        query = torch.randn(1, total_seq, heads, head_dim)
        key = torch.randn(1, total_seq, heads, head_dim)
        hidden = torch.randn(1, total_seq, hidden_dim)

        # Token positions that are valid in the actual 32-token caption
        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [2, 3], "lora_b": [10, 11]},
            img_seq_len=img_len,
            cap_seq_len=cap_len_from_embeds,  # 2560 — much larger than reality
            text_start_idx=padded_img,
        )

        # Should succeed: cap_len clamped to actual_cap_in_seq=32
        assert sim_maps is not None
        assert "lora_a" in sim_maps
        assert "lora_b" in sim_maps
        assert sim_maps["lora_a"].shape == (1, img_len, 1)

        # Process into masks
        masks = process_sim_maps(sim_maps, img_h, img_w, img_len)
        assert masks is not None
        for mask in masks.values():
            assert mask.shape == (1, img_len)

    def test_caption_truncated_positions_beyond_actual_cap_dropped(self):
        """Trigger word positions beyond the actual caption are dropped."""
        img_len = 1024
        padded_img = 1024
        actual_cap = 32
        total_seq = padded_img + actual_cap

        query = torch.randn(1, total_seq, 4, 16)
        key = torch.randn(1, total_seq, 4, 16)
        hidden = torch.randn(1, total_seq, 64)

        # Positions 100, 200 are beyond actual cap of 32
        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [2], "lora_b": [100, 200]},
            img_seq_len=img_len,
            cap_seq_len=2560,
            text_start_idx=padded_img,
        )

        assert sim_maps is not None
        # lora_a valid (pos 2 < 32), lora_b dropped (pos 100, 200 >= 32)
        assert "lora_a" in sim_maps
        assert "lora_b" not in sim_maps


# ===================================================================
# Position clamping and bounds
# ===================================================================

class TestPositionBounds:
    """Verify out-of-range token positions are handled safely."""

    def test_positions_beyond_cap_len_are_skipped(self):
        """Token positions >= cap_len should be silently dropped."""
        query, key, hidden, img_len, padded_img, _ = _make_sim_map_inputs(
            img_h=16, img_w=16, cap_len=10,
        )

        # Position 50 is way beyond cap_len=10
        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [50, 100]},
            img_seq_len=img_len,
            cap_seq_len=10,
            text_start_idx=padded_img,
        )
        # All positions invalid → no concept maps, only EOS
        # (or None if implementation skips EOS too)
        if sim_maps is not None:
            assert "lora_a" not in sim_maps

    def test_mixed_valid_and_invalid_positions(self):
        """Some positions valid, some beyond cap_len."""
        query, key, hidden, img_len, padded_img, _ = _make_sim_map_inputs(
            img_h=16, img_w=16, cap_len=20,
        )

        sim_maps = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [5, 100]},  # 5 valid, 100 invalid
            img_seq_len=img_len,
            cap_seq_len=20,
            text_start_idx=padded_img,
        )
        assert sim_maps is not None
        assert "lora_a" in sim_maps

    def test_cap_len_clamped_when_embeds_larger_than_sequence(self):
        """cap_seq_len > available → clamped, positions still valid work."""
        query = torch.randn(1, 100, 4, 16)
        key = torch.randn(1, 100, 4, 16)
        hidden = torch.randn(1, 100, 64)

        # text_start=80, cap_from_embeds=50, but seq=100 → actual cap=20
        result = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [0]},  # pos 0 < 20 → valid
            img_seq_len=64,
            cap_seq_len=50,
            text_start_idx=80,
        )
        # Succeeds with clamped caption length
        assert result is not None
        assert "lora_a" in result

    def test_text_start_at_sequence_end_returns_none(self):
        """text_start_idx == total_seq → no caption at all → None."""
        query = torch.randn(1, 100, 4, 16)
        key = torch.randn(1, 100, 4, 16)
        hidden = torch.randn(1, 100, 64)

        result = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [0]},
            img_seq_len=64,
            cap_seq_len=50,
            text_start_idx=100,  # no room for caption
        )
        assert result is None


# ===================================================================
# NaN handling in process_sim_maps
# ===================================================================

class TestNaNHandling:
    """Verify NaN values in sim maps are handled gracefully."""

    def test_nan_in_one_concept_drops_it(self):
        """If one concept's sim map has NaN, it's dropped."""
        good = torch.randn(1, 64, 1).abs()  # ensure positive
        bad = torch.full((1, 64, 1), float("nan"))
        eos = torch.randn(1, 64, 1).abs()

        sim_maps = {"lora_a": good, "lora_b": bad, "__eos__": eos}

        masks = process_sim_maps(sim_maps, h_tokens=8, w_tokens=8, img_token_count=64)

        # lora_b dropped due to NaN, lora_a should still produce a mask
        assert "lora_a" in masks
        assert "lora_b" not in masks

    def test_nan_in_all_concepts_returns_empty(self):
        """If all sim maps have NaN, return empty dict."""
        bad_a = torch.full((1, 64, 1), float("nan"))
        bad_b = torch.full((1, 64, 1), float("nan"))
        eos = torch.randn(1, 64, 1).abs()

        sim_maps = {"lora_a": bad_a, "lora_b": bad_b, "__eos__": eos}

        masks = process_sim_maps(sim_maps, h_tokens=8, w_tokens=8, img_token_count=64)
        assert masks == {}

    def test_nan_in_eos_only_still_works(self):
        """NaN in EOS is dropped; concepts use fallback background."""
        good_a = torch.randn(1, 64, 1).abs()
        good_b = torch.randn(1, 64, 1).abs()
        bad_eos = torch.full((1, 64, 1), float("nan"))

        sim_maps = {"lora_a": good_a, "lora_b": good_b, "__eos__": bad_eos}

        masks = process_sim_maps(sim_maps, h_tokens=8, w_tokens=8, img_token_count=64)
        assert "lora_a" in masks
        assert "lora_b" in masks


# ===================================================================
# stabilized_balanced_argmax robustness
# ===================================================================

class TestBalancedArgmaxRobustness:
    """Test stabilized_balanced_argmax with edge cases."""

    def test_normal_two_channels(self):
        """Normal case: two channels, distinct spatial regions."""
        logits = torch.zeros(1, 2, 64)
        # Left half favors channel 0, right half favors channel 1
        logits[0, 0, :32] = 1.0
        logits[0, 1, 32:] = 1.0

        result = stabilized_balanced_argmax(logits, h=8, w=8)
        assert result.shape == (1, 64)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_single_channel(self):
        """C=1: all indices should be 0."""
        logits = torch.randn(1, 1, 64)
        result = stabilized_balanced_argmax(logits, h=8, w=8)
        assert result.shape == (1, 64)
        assert (result == 0).all()

    def test_uniform_logits(self):
        """Equal logits across channels — should not crash."""
        logits = torch.ones(1, 2, 256)
        result = stabilized_balanced_argmax(logits, h=16, w=16)
        assert result.shape == (1, 256)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_extreme_values(self):
        """Very large logits — tests numerical stability."""
        logits = torch.randn(1, 2, 64) * 1e6
        result = stabilized_balanced_argmax(logits, h=8, w=8)
        assert result.shape == (1, 64)
        assert result.min() >= 0
        assert result.max() <= 1

    def test_three_channels(self):
        """C=3 channels."""
        logits = torch.randn(1, 3, 144)
        result = stabilized_balanced_argmax(logits, h=12, w=12)
        assert result.shape == (1, 144)
        assert result.min() >= 0
        assert result.max() <= 2


# ===================================================================
# morphological_clean_mask robustness
# ===================================================================

class TestMorphologicalCleanMask:
    """Test morphological operations with various dimensions."""

    def test_square(self):
        mask = torch.ones(1, 64)
        result = morphological_clean_mask(mask, h=8, w=8)
        assert result.shape == (1, 64)

    def test_non_square(self):
        mask = torch.ones(1, 48 * 32)
        result = morphological_clean_mask(mask, h=48, w=32)
        assert result.shape == (1, 48 * 32)

    def test_small(self):
        """Very small spatial dims (4x4 = 16 tokens)."""
        mask = torch.ones(1, 16)
        result = morphological_clean_mask(mask, h=4, w=4)
        assert result.shape == (1, 16)

    def test_all_zeros(self):
        mask = torch.zeros(1, 64)
        result = morphological_clean_mask(mask, h=8, w=8)
        assert result.shape == (1, 64)
        assert (result == 0).all()
