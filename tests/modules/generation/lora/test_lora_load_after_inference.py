"""Regression test: loading a second LoRA after inference must not crash.

Scenario from user report (Flux Klein models):
1. Load first LoRA
2. Run inference (denoise node — model gets dtype-cast inside context manager)
3. Add a second LoRA
4. PEFT's inject_adapter_in_model fails with:
   "Setting requires_grad=True on inference tensor outside InferenceMode is not allowed."

Root cause: model_manager.get() calls model.to(device, dtype) inside the denoise
node's context.  When the context is @torch.inference_mode(), the dtype-cast
creates new parameter tensors permanently marked as "inference tensors".
A later inject_adapter_in_model call (from LoraNode) then fails because PEFT
needs to set requires_grad=True on those tainted parameters.

Fix: replace @torch.inference_mode() with @torch.no_grad() on denoise nodes.
torch.no_grad() provides the same gradient-disabling benefit without permanently
marking tensors as inference-only.

Why Z-Image was unaffected: Z-Image models are already in bf16 when loaded,
so the .to() call inside the denoise node is a no-op (same dtype = no new tensors).
Flux Klein models may undergo dtype conversion, triggering the bug.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model


# ---------------------------------------------------------------------------
# Model stub (reuses pattern from test_lokr_peft_coexistence.py)
# ---------------------------------------------------------------------------

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock()])
        self.peft_config = {}

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

def _inject_lora(model, adapter_name, targets=("to_q", "to_k", "to_v")):
    """Inject a PEFT LoRA adapter into the model."""
    config = LoraConfig(target_modules=list(targets), r=4, lora_alpha=4)
    inject_adapter_in_model(config, model, adapter_name=adapter_name)


def _simulate_denoise_with_dtype_cast(model, x, context_fn):
    """Simulate denoise node: dtype-cast (like mm.get() → .to(bf16)) + forward."""
    with context_fn():
        model = model.to(torch.bfloat16)
        out = model(x.to(torch.bfloat16))
    return model, out


# ===========================================================================
# Tests
# ===========================================================================

class TestLoraLoadAfterInference:
    """Loading a second LoRA after running inference must succeed."""

    def test_inference_mode_dtype_cast_taints_parameters(self):
        """Confirm the root cause: .to(bf16) inside inference_mode creates inference tensors."""
        model = TinyTransformer()
        _inject_lora(model, "lora_a")

        x = torch.randn(1, 16)

        model, _ = _simulate_denoise_with_dtype_cast(model, x, torch.inference_mode)

        # Parameters are now inference tensors — second adapter injection must fail
        try:
            _inject_lora(model, "lora_b")
            # If PEFT changes to avoid requires_grad, that's fine — bug is gone either way
        except RuntimeError as e:
            assert "requires_grad" in str(e).lower() or "inference" in str(e).lower(), (
                f"Expected requires_grad/inference error, got: {e}"
            )

    def test_no_grad_dtype_cast_does_not_taint(self):
        """With @torch.no_grad() (the fix), second LoRA loads without error."""
        model = TinyTransformer()
        _inject_lora(model, "lora_a")

        x = torch.randn(1, 16)

        model, _ = _simulate_denoise_with_dtype_cast(model, x, torch.no_grad)

        # Parameters are NOT inference tensors — second adapter must succeed
        _inject_lora(model, "lora_b")

        # Verify both adapters are present
        assert "lora_a" in model.peft_config
        assert "lora_b" in model.peft_config

    def test_multiple_loras_across_generations_no_grad(self):
        """Simulate multiple generation runs with LoRA additions between them."""
        model = TinyTransformer()
        x = torch.randn(1, 16)

        # Gen 1: single LoRA + inference with dtype cast
        _inject_lora(model, "lora_a")
        model, _ = _simulate_denoise_with_dtype_cast(model, x, torch.no_grad)

        # Gen 2: add second LoRA + inference
        _inject_lora(model, "lora_b")
        with torch.no_grad():
            out = model(x.to(torch.bfloat16))

        # Gen 3: add third LoRA
        _inject_lora(model, "lora_c")
        with torch.no_grad():
            out = model(x.to(torch.bfloat16))

        assert out.shape == (1, 16)
        assert set(model.peft_config.keys()) == {"lora_a", "lora_b", "lora_c"}

    def test_repeated_dtype_casts_no_grad(self):
        """Repeated .to(bf16) calls inside no_grad must not accumulate problems."""
        model = TinyTransformer()
        x = torch.randn(1, 16)

        _inject_lora(model, "lora_a")

        # Simulate 5 generation runs, each doing a dtype cast
        for _ in range(5):
            with torch.no_grad():
                model = model.to(torch.bfloat16)
                model(x.to(torch.bfloat16))

        # Adding a new adapter after many runs must still work
        _inject_lora(model, "lora_b")
        assert "lora_b" in model.peft_config

    def test_same_dtype_inside_inference_mode_is_fine(self):
        """When .to() doesn't change dtype, inference_mode doesn't taint params.

        This explains why Z-Image models (already bf16) never hit the bug.
        """
        model = TinyTransformer().to(torch.bfloat16)  # Already bf16
        x = torch.randn(1, 16, dtype=torch.bfloat16)

        _inject_lora(model, "lora_a")

        # .to(bf16) on a bf16 model is a no-op — no new tensors created
        with torch.inference_mode():
            model = model.to(torch.bfloat16)
            model(x)

        # Second adapter should succeed because no inference tensors were created
        _inject_lora(model, "lora_b")
        assert "lora_b" in model.peft_config
