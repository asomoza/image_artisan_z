"""Regression test: LoKr unmerge must fully restore transformer weights.

Scenario: A normal PEFT LoRA is loaded, then a LoKr LoRA is merged into the
transformer's base weights. When the LoKr LoRA is removed (unmerged), the
transformer weights must return to their pre-LoKr state exactly. If they don't,
the normal LoRA's effective output is corrupted.

This test does NOT require GPU or real model files.
"""

import torch
import torch.nn as nn

from iartisanz.modules.generation.graph.nodes.lora_node import (
    _apply_lokr_merge,
    _get_module_weight,
)


# ---------------------------------------------------------------------------
# Minimal model stubs
# ---------------------------------------------------------------------------

class FakeTransformer(nn.Module):
    """Minimal transformer with a couple of Linear layers for testing."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([FakeLayer()])


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FakeAttn()


class FakeAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(16, 16, bias=False)
        self.to_k = nn.Linear(16, 16, bias=False)


class FakePEFTWrapped(nn.Module):
    """Simulates what PEFT does: wraps a Linear, exposing base_layer."""

    def __init__(self, linear: nn.Linear):
        super().__init__()
        self.base_layer = linear


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lokr_entries_for_layer(module_path: str):
    """Create a single LoKr entry targeting the given module path.

    Uses small but non-trivial w1/w2 so kron product is non-zero.
    kron(4x4, 4x4) = 16x16, matching the FakeTransformer layer size.
    """
    torch.manual_seed(42)
    w1 = torch.randn(4, 4)  # already float32 (as _parse_lokr_entries produces)
    w2 = torch.randn(4, 4)
    return [{"w1": w1, "w2": w2, "targets": [(module_path, None)]}]


# ===========================================================================
# Tests
# ===========================================================================


class TestLokrUnmergeRestoresWeights:
    """After merge then unmerge, weights must be exactly restored."""

    def test_float32_roundtrip(self):
        """float32 weights: merge + unmerge = exact identity."""
        model = FakeTransformer()
        model.float()
        original = model.layers[0].attn.to_q.weight.data.clone()

        entries = _make_lokr_entries_for_layer("layers.0.attn.to_q")
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)

        # Weight should have changed
        assert not torch.equal(model.layers[0].attn.to_q.weight.data, original)

        # Unmerge (scale → 0)
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        # Must be exactly restored
        assert torch.equal(model.layers[0].attn.to_q.weight.data, original), (
            "float32 LoKr unmerge did not restore weights exactly"
        )

    def test_bfloat16_roundtrip(self):
        """bfloat16 weights: merge + unmerge must restore exactly."""
        model = FakeTransformer()
        model.bfloat16()
        original = model.layers[0].attn.to_q.weight.data.clone()

        entries = _make_lokr_entries_for_layer("layers.0.attn.to_q")
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)

        assert not torch.equal(model.layers[0].attn.to_q.weight.data, original)

        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        torch.testing.assert_close(
            model.layers[0].attn.to_q.weight.data,
            original,
            atol=0, rtol=0,
            msg="bfloat16 LoKr unmerge did not restore weights exactly",
        )

    def test_half_roundtrip(self):
        """float16 weights: same exact restoration requirement."""
        model = FakeTransformer()
        model.half()
        original = model.layers[0].attn.to_q.weight.data.clone()

        entries = _make_lokr_entries_for_layer("layers.0.attn.to_q")
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        torch.testing.assert_close(
            model.layers[0].attn.to_q.weight.data,
            original,
            atol=0, rtol=0,
            msg="float16 LoKr unmerge did not restore weights exactly",
        )

    def test_fractional_scale_roundtrip(self):
        """Merge at scale 0.7 then unmerge must restore weights."""
        model = FakeTransformer()
        model.bfloat16()
        original = model.layers[0].attn.to_q.weight.data.clone()

        entries = _make_lokr_entries_for_layer("layers.0.attn.to_q")
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=0.7, original_weights=saved)
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        torch.testing.assert_close(
            model.layers[0].attn.to_q.weight.data,
            original,
            atol=0, rtol=0,
            msg="Fractional scale LoKr unmerge did not restore weights",
        )

    def test_incremental_scale_changes(self):
        """Simulate LoraNode._update_lokr_merge: scale 0→1→0.5→0.

        Each step restores originals then re-applies at new scale.
        Final state must match original exactly.
        """
        model = FakeTransformer()
        model.bfloat16()
        original = model.layers[0].attn.to_q.weight.data.clone()

        entries = _make_lokr_entries_for_layer("layers.0.attn.to_q")
        saved = {}

        # 0 → 1.0
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        # 1.0 → 0.5
        _apply_lokr_merge(model, entries, new_scale=0.5, original_weights=saved)
        # 0.5 → 0
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        torch.testing.assert_close(
            model.layers[0].attn.to_q.weight.data,
            original,
            atol=0, rtol=0,
            msg="Incremental scale changes did not restore weights",
        )

    def test_peft_wrapped_module(self):
        """When a normal LoRA is loaded, PEFT wraps Linear with base_layer.

        LoKr merge targets base_layer.weight. Unmerge must hit the same tensor.
        """
        model = FakeTransformer()
        model.bfloat16()

        # Simulate PEFT wrapping the to_q module
        original_linear = model.layers[0].attn.to_q
        wrapped = FakePEFTWrapped(original_linear)
        model.layers[0].attn.to_q = wrapped

        original_weight = original_linear.weight.data.clone()

        entries = _make_lokr_entries_for_layer("layers.0.attn.to_q")

        # _get_module_weight should find base_layer.weight
        w = _get_module_weight(model, "layers.0.attn.to_q")
        assert w is not None, "_get_module_weight failed to find PEFT-wrapped weight"
        assert w.data_ptr() == original_linear.weight.data_ptr()

        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        assert not torch.equal(original_linear.weight.data, original_weight)

        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)
        torch.testing.assert_close(
            original_linear.weight.data,
            original_weight,
            atol=0, rtol=0,
            msg="PEFT-wrapped LoKr unmerge did not restore base_layer weights",
        )

    def test_multiple_layers_all_restored(self):
        """LoKr entries targeting multiple layers must all restore cleanly."""
        model = FakeTransformer()
        model.bfloat16()

        orig_q = model.layers[0].attn.to_q.weight.data.clone()
        orig_k = model.layers[0].attn.to_k.weight.data.clone()

        torch.manual_seed(42)
        entries = [
            {"w1": torch.randn(4, 4), "w2": torch.randn(4, 4),
             "targets": [("layers.0.attn.to_q", None)]},
            {"w1": torch.randn(4, 4), "w2": torch.randn(4, 4),
             "targets": [("layers.0.attn.to_k", None)]},
        ]

        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        torch.testing.assert_close(
            model.layers[0].attn.to_q.weight.data, orig_q, atol=0, rtol=0,
            msg="to_q not restored after multi-layer LoKr unmerge",
        )
        torch.testing.assert_close(
            model.layers[0].attn.to_k.weight.data, orig_k, atol=0, rtol=0,
            msg="to_k not restored after multi-layer LoKr unmerge",
        )

    def test_qkv_split_roundtrip(self):
        """LoKr entry with QKV split_idx must restore all 3 targets."""
        model = FakeTransformer()
        model.layers[0].attn.to_v = nn.Linear(16, 16, bias=False)
        model.bfloat16()

        orig_q = model.layers[0].attn.to_q.weight.data.clone()
        orig_k = model.layers[0].attn.to_k.weight.data.clone()
        orig_v = model.layers[0].attn.to_v.weight.data.clone()

        # kron product must be (48, 16) so it can be chunked 3 ways into (16, 16)
        torch.manual_seed(99)
        entries = [{
            "w1": torch.randn(12, 4),
            "w2": torch.randn(4, 4),
            "targets": [
                ("layers.0.attn.to_q", 0),
                ("layers.0.attn.to_k", 1),
                ("layers.0.attn.to_v", 2),
            ],
        }]

        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        torch.testing.assert_close(
            model.layers[0].attn.to_q.weight.data, orig_q, atol=0, rtol=0,
        )
        torch.testing.assert_close(
            model.layers[0].attn.to_k.weight.data, orig_k, atol=0, rtol=0,
        )
        torch.testing.assert_close(
            model.layers[0].attn.to_v.weight.data, orig_v, atol=0, rtol=0,
        )
