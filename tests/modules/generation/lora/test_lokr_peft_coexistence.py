"""Regression test: LoKr + PEFT adapter coexistence.

Scenario from user report: a distillation LoRA (PEFT adapter) is always loaded.
User swaps a second slot between a normal LoRA (PEFT) and a LoKr LoRA (weight merge).
After loading and unloading the LoKr, the normal LoRA's output changes.

Tests verify that LoKr merge/unmerge does not corrupt the transformer's base
weights when PEFT adapters are active on the same modules.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model

from iartisanz.modules.generation.graph.nodes.lora_node import (
    _apply_lokr_merge,
    _get_module_weight,
)


# ---------------------------------------------------------------------------
# Model stub
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    """Minimal transformer with structure matching real Z-Image/Flux2 models."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention()
        self.ff = FeedForward()

    def forward(self, x):
        return self.ff(self.attn(x))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = nn.Linear(16, 16, bias=False)
        self.to_k = nn.Linear(16, 16, bias=False)
        self.to_v = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.to_q(x) + self.to_k(x) + self.to_v(x)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 16, bias=False)

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_lokr_entries():
    """LoKr entries targeting to_q and to_k (kron(4x4, 4x4) = 16x16)."""
    torch.manual_seed(42)
    return [
        {"w1": torch.randn(4, 4), "w2": torch.randn(4, 4),
         "targets": [("layers.0.attn.to_q", None)]},
        {"w1": torch.randn(4, 4), "w2": torch.randn(4, 4),
         "targets": [("layers.0.attn.to_k", None)]},
    ]


def _forward(model, x):
    """Deterministic forward pass."""
    model.eval()
    with torch.no_grad():
        return model(x).clone()


# ===========================================================================
# Tests
# ===========================================================================

class TestLokrPeftCoexistence:
    """LoKr merge/unmerge must not corrupt PEFT adapter base weights."""

    def test_full_swap_cycle_distill_plus_lokr(self):
        """Distillation LoRA always active. LoKr loaded then unloaded.
        Forward output must be identical before and after the LoKr cycle.
        """
        model = TinyTransformer()
        model.bfloat16()
        x = torch.randn(1, 16).bfloat16()

        # Load distillation LoRA (PEFT, always present)
        config = LoraConfig(
            target_modules=["to_q", "to_k", "to_v", "linear"], r=4, lora_alpha=4
        )
        model = inject_adapter_in_model(config, model, adapter_name="distill")

        out_before = _forward(model, x)

        # LoKr merge
        entries = _make_lokr_entries()
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        out_with_lokr = _forward(model, x)
        assert not torch.equal(out_before, out_with_lokr), "LoKr should change output"

        # LoKr unmerge
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)
        out_after = _forward(model, x)

        torch.testing.assert_close(
            out_after, out_before, atol=0, rtol=0,
            msg="Output changed after LoKr load/unload cycle with distillation PEFT active",
        )

    def test_full_swap_cycle_distill_plus_normal_plus_lokr(self):
        """Full user scenario: distillation always active, swap normal↔LoKr.

        1. Distill + Normal → output A
        2. Remove Normal
        3. Add LoKr
        4. Remove LoKr
        5. Add Normal again → output must match A
        """
        torch.manual_seed(0)
        model = TinyTransformer()
        model.bfloat16()
        x = torch.randn(1, 16).bfloat16()

        # Distillation LoRA
        config_distill = LoraConfig(
            target_modules=["to_q", "to_k", "to_v"], r=4, lora_alpha=4
        )
        model = inject_adapter_in_model(config_distill, model, adapter_name="distill")

        # Normal LoRA (second adapter)
        torch.manual_seed(123)
        config_normal = LoraConfig(
            target_modules=["to_q", "to_k"], r=4, lora_alpha=4
        )
        inject_adapter_in_model(config_normal, model, adapter_name="normal")

        out_with_normal = _forward(model, x)

        # Capture base weights before LoKr cycle
        base_q_before = model.layers[0].attn.to_q.base_layer.weight.data.clone()
        base_k_before = model.layers[0].attn.to_k.base_layer.weight.data.clone()

        # Remove Normal LoRA
        for mod in [model.layers[0].attn.to_q, model.layers[0].attn.to_k]:
            mod.delete_adapter("normal")

        # Add LoKr
        entries = _make_lokr_entries()
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)

        # Remove LoKr
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        # Verify base weights restored
        torch.testing.assert_close(
            model.layers[0].attn.to_q.base_layer.weight.data,
            base_q_before,
            atol=0, rtol=0,
            msg="to_q base weight changed after LoKr cycle",
        )
        torch.testing.assert_close(
            model.layers[0].attn.to_k.base_layer.weight.data,
            base_k_before,
            atol=0, rtol=0,
            msg="to_k base weight changed after LoKr cycle",
        )

        # Re-add Normal LoRA with same weights
        torch.manual_seed(123)
        config_normal2 = LoraConfig(
            target_modules=["to_q", "to_k"], r=4, lora_alpha=4
        )
        inject_adapter_in_model(config_normal2, model, adapter_name="normal")

        out_with_normal_again = _forward(model, x)

        torch.testing.assert_close(
            out_with_normal_again, out_with_normal, atol=0, rtol=0,
            msg="Normal LoRA output changed after LoKr swap cycle",
        )

    def test_lokr_does_not_corrupt_peft_adapter_weights(self):
        """LoKr merge/unmerge must not touch PEFT's lora_A/lora_B tensors."""
        model = TinyTransformer()
        model.bfloat16()

        config = LoraConfig(target_modules=["to_q"], r=4, lora_alpha=4)
        model = inject_adapter_in_model(config, model, adapter_name="distill")

        # Capture PEFT adapter weights
        peft_module = model.layers[0].attn.to_q
        lora_a_before = peft_module.lora_A["distill"].weight.data.clone()
        lora_b_before = peft_module.lora_B["distill"].weight.data.clone()

        # LoKr cycle
        entries = _make_lokr_entries()
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        # PEFT adapter weights must be untouched
        torch.testing.assert_close(
            peft_module.lora_A["distill"].weight.data, lora_a_before, atol=0, rtol=0,
            msg="PEFT lora_A corrupted by LoKr cycle",
        )
        torch.testing.assert_close(
            peft_module.lora_B["distill"].weight.data, lora_b_before, atol=0, rtol=0,
            msg="PEFT lora_B corrupted by LoKr cycle",
        )

    def test_multiple_lokr_cycles(self):
        """Repeated LoKr load/unload cycles must not accumulate drift."""
        model = TinyTransformer()
        model.bfloat16()
        x = torch.randn(1, 16).bfloat16()

        config = LoraConfig(target_modules=["to_q", "to_k"], r=4, lora_alpha=4)
        model = inject_adapter_in_model(config, model, adapter_name="distill")

        out_baseline = _forward(model, x)

        for cycle in range(5):
            entries = _make_lokr_entries()
            saved = {}
            _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
            _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        out_after_cycles = _forward(model, x)
        torch.testing.assert_close(
            out_after_cycles, out_baseline, atol=0, rtol=0,
            msg=f"Output drifted after 5 LoKr load/unload cycles",
        )

    def test_lokr_scale_changes_with_peft(self):
        """Scale changes (0→1→0.5→0) with active PEFT adapter."""
        model = TinyTransformer()
        model.bfloat16()
        x = torch.randn(1, 16).bfloat16()

        config = LoraConfig(target_modules=["to_q", "to_k"], r=4, lora_alpha=4)
        model = inject_adapter_in_model(config, model, adapter_name="distill")

        out_baseline = _forward(model, x)

        entries = _make_lokr_entries()
        saved = {}
        _apply_lokr_merge(model, entries, new_scale=1.0, original_weights=saved)
        _apply_lokr_merge(model, entries, new_scale=0.5, original_weights=saved)
        _apply_lokr_merge(model, entries, new_scale=0.0, original_weights=saved)

        out_after = _forward(model, x)
        torch.testing.assert_close(
            out_after, out_baseline, atol=0, rtol=0,
            msg="Scale changes with PEFT active corrupted output",
        )

    def test_get_module_weight_peft_wrapped(self):
        """_get_module_weight must consistently return base_layer.weight for PEFT modules."""
        model = TinyTransformer()
        config = LoraConfig(target_modules=["to_q"], r=4, lora_alpha=4)
        model = inject_adapter_in_model(config, model, adapter_name="distill")

        w = _get_module_weight(model, "layers.0.attn.to_q")
        base_w = model.layers[0].attn.to_q.base_layer.weight
        assert w.data_ptr() == base_w.data_ptr(), (
            "_get_module_weight should return base_layer.weight for PEFT-wrapped module"
        )

    def test_get_module_weight_bare(self):
        """_get_module_weight returns weight directly for non-PEFT modules."""
        model = TinyTransformer()
        w = _get_module_weight(model, "layers.0.attn.to_q")
        assert w.data_ptr() == model.layers[0].attn.to_q.weight.data_ptr()
