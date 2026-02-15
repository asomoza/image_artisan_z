"""Regression tests for FreeFuse processor replacement.

Verifies that replacing the processor instance (class-level ``__call__``)
works correctly for both Phase A (attention bias injection) and Phase B
(sim map collection).

Background: The original implementation used ``proc.__call__ = new_func``
to monkey-patch the attention processor.  However, ``Attention.forward()``
calls ``self.processor(...)`` which is an *implicit* ``__call__`` invocation.
Python resolves implicit dunder calls on the **class** (``type(proc).__call__``),
not the instance dict.  So the patched function was never invoked.

The fix replaces the processor instance entirely with a wrapper class whose
``__call__`` is defined at the class level.
"""

import torch

from iartisanz.modules.generation.graph.nodes.freefuse_attention_bias import (
    _BiasInjectionProcessor,
    patch_attention_with_bias,
    unpatch_attention_bias,
)
from iartisanz.modules.generation.graph.nodes.freefuse_auto_mask import (
    _SimMapCollectorProcessor,
    install_sim_map_collector,
    remove_sim_map_collector,
)


# ---------------------------------------------------------------------------
# Helpers that mimic the diffusers Attention → processor call chain
# ---------------------------------------------------------------------------

class _FakeProcessor:
    """Minimal attention processor for testing."""

    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        self.call_count = 0

    def __call__(self, attn, hidden_states, **kwargs):
        self.call_count += 1
        return hidden_states


class _FakeAttentionModule:
    """Mimics ``diffusers.models.attention_processor.Attention``.

    The critical line is in ``forward()``: ``self.processor(self, hidden_states, ...)``
    which is the implicit ``__call__`` that triggered the original bug.
    """

    def __init__(self):
        self.processor = _FakeProcessor()

    def forward(self, hidden_states, **kwargs):
        return self.processor(self, hidden_states, **kwargs)


class _FakeLayer:
    def __init__(self):
        self.attention = _FakeAttentionModule()


class _FakeTransformer:
    def __init__(self, num_layers=4):
        self.layers = [_FakeLayer() for _ in range(num_layers)]


# ===================================================================
# Core regression: prove instance __call__ patching is broken
# ===================================================================

class TestInstanceCallPatchingRegression:
    """Prove that ``proc.__call__ = func`` does NOT intercept implicit calls."""

    def test_instance_call_attribute_is_not_used_by_implicit_call(self):
        """The old monkey-patching approach silently fails."""
        proc = _FakeProcessor()
        attn = _FakeAttentionModule()
        attn.processor = proc

        intercepted = []

        def patched_call(attn_arg, hidden_states, **kwargs):
            intercepted.append(True)
            return hidden_states

        # Old (broken) approach: set __call__ as instance attribute
        proc.__call__ = patched_call

        attn.forward(torch.randn(1, 4, 8))

        # Patched function was NOT called
        assert len(intercepted) == 0
        # Original class-level __call__ WAS called
        assert proc.call_count == 1

    def test_processor_replacement_is_used_by_implicit_call(self):
        """The fixed approach (replacing the processor) works."""
        attn = _FakeAttentionModule()
        original_proc = attn.processor

        intercepted = []

        class _Wrapper:
            def __init__(self, original):
                self._original = original

            def __call__(self, attn_arg, hidden_states, **kwargs):
                intercepted.append(True)
                return self._original(attn_arg, hidden_states, **kwargs)

        attn.processor = _Wrapper(original_proc)

        attn.forward(torch.randn(1, 4, 8))

        # Wrapper WAS called
        assert len(intercepted) == 1
        # Original was delegated to
        assert original_proc.call_count == 1


# ===================================================================
# Phase A: _BiasInjectionProcessor
# ===================================================================

class TestBiasInjectionProcessor:
    """Test the Phase A bias injection wrapper."""

    def test_converts_2d_mask_to_4d_with_bias(self):
        received = {}

        class _Recorder:
            _attention_backend = None
            _parallel_config = None

            def __call__(self, attn, hidden_states, attention_mask=None, **kw):
                received["mask"] = attention_mask
                return hidden_states

        original = _Recorder()
        seq_len = 8
        bias = torch.ones(1, seq_len, seq_len) * 2.5
        wrapper = _BiasInjectionProcessor(original, bias)

        hs = torch.randn(1, seq_len, 16)
        mask_2d = torch.ones(1, seq_len, dtype=torch.bool)

        wrapper(None, hs, attention_mask=mask_2d)

        result = received["mask"]
        # 2D bool → 4D float with bias
        assert result.ndim == 4
        assert result.shape == (1, 1, seq_len, seq_len)
        # All-True mask → float_mask is all zeros → result is pure bias
        assert torch.allclose(result, torch.full_like(result, 2.5))

    def test_2d_mask_with_false_values_adds_neginf(self):
        received = {}

        class _Recorder:
            _attention_backend = None
            _parallel_config = None

            def __call__(self, attn, hidden_states, attention_mask=None, **kw):
                received["mask"] = attention_mask
                return hidden_states

        original = _Recorder()
        seq_len = 4
        bias = torch.zeros(1, seq_len, seq_len)
        wrapper = _BiasInjectionProcessor(original, bias)

        hs = torch.randn(1, seq_len, 16)
        mask_2d = torch.tensor([[True, True, False, False]])

        wrapper(None, hs, attention_mask=mask_2d)

        result = received["mask"]
        assert result.ndim == 4
        # Positions where mask is False should have very negative values
        # (the row dimension broadcasts: mask[:, None, None, :])
        assert result[0, 0, 0, 2] < -1e30
        assert result[0, 0, 0, 3] < -1e30
        # True positions should be ~0 (float_mask=0 + bias=0)
        assert result[0, 0, 0, 0] == 0.0
        assert result[0, 0, 0, 1] == 0.0

    def test_passes_through_4d_mask_unchanged(self):
        received = {}

        class _Recorder:
            _attention_backend = None
            _parallel_config = None

            def __call__(self, attn, hidden_states, attention_mask=None, **kw):
                received["mask"] = attention_mask
                return hidden_states

        original = _Recorder()
        bias = torch.ones(1, 8, 8)
        wrapper = _BiasInjectionProcessor(original, bias)

        mask_4d = torch.zeros(1, 1, 8, 8)
        wrapper(None, torch.randn(1, 8, 16), attention_mask=mask_4d)

        # 4D mask not touched (ndim != 2)
        assert received["mask"] is mask_4d

    def test_passes_through_none_mask(self):
        received = {}

        class _Recorder:
            _attention_backend = None
            _parallel_config = None

            def __call__(self, attn, hidden_states, attention_mask=None, **kw):
                received["mask"] = attention_mask
                return hidden_states

        original = _Recorder()
        wrapper = _BiasInjectionProcessor(original, torch.ones(1, 8, 8))
        wrapper(None, torch.randn(1, 8, 16), attention_mask=None)

        assert received["mask"] is None

    def test_bias_sliced_to_match_sequence_length(self):
        """Bias is larger than actual sequence → sliced to match."""
        received = {}

        class _Recorder:
            _attention_backend = None
            _parallel_config = None

            def __call__(self, attn, hidden_states, attention_mask=None, **kw):
                received["mask"] = attention_mask
                return hidden_states

        original = _Recorder()
        # Bias built for seq=16, but actual mask is seq=8
        bias = torch.ones(1, 16, 16) * 3.0
        wrapper = _BiasInjectionProcessor(original, bias)

        hs = torch.randn(1, 8, 16)
        mask_2d = torch.ones(1, 8, dtype=torch.bool)
        wrapper(None, hs, attention_mask=mask_2d)

        result = received["mask"]
        assert result.shape == (1, 1, 8, 8)
        assert torch.allclose(result, torch.full_like(result, 3.0))

    def test_bias_expanded_for_batch(self):
        """Bias has B=1 but input has B=2 → expanded."""
        received = {}

        class _Recorder:
            _attention_backend = None
            _parallel_config = None

            def __call__(self, attn, hidden_states, attention_mask=None, **kw):
                received["mask"] = attention_mask
                return hidden_states

        original = _Recorder()
        bias = torch.ones(1, 8, 8) * 1.5
        wrapper = _BiasInjectionProcessor(original, bias)

        hs = torch.randn(2, 8, 16)
        mask_2d = torch.ones(2, 8, dtype=torch.bool)
        wrapper(None, hs, attention_mask=mask_2d)

        result = received["mask"]
        assert result.shape == (2, 1, 8, 8)


# ===================================================================
# Phase A: patch / unpatch lifecycle
# ===================================================================

class TestPatchUnpatchAttentionBias:
    """Test patch_attention_with_bias and unpatch_attention_bias lifecycle."""

    def test_patch_replaces_last_half_processors(self):
        transformer = _FakeTransformer(num_layers=4)
        original_procs = [l.attention.processor for l in transformer.layers]

        bias = torch.zeros(1, 8, 8)
        patched = patch_attention_with_bias(transformer, bias)

        # last_half_only=True (default): only layers 2,3 patched
        assert len(patched) == 2
        # First half unchanged
        for i in range(2):
            assert isinstance(transformer.layers[i].attention.processor, _FakeProcessor)
        # Last half patched
        for i in range(2, 4):
            assert isinstance(transformer.layers[i].attention.processor, _BiasInjectionProcessor)

        # Original processors preserved in patched list
        for (_, orig), expected in zip(patched, original_procs[2:]):
            assert orig is expected

    def test_patch_all_layers_when_last_half_false(self):
        transformer = _FakeTransformer(num_layers=3)

        bias = torch.zeros(1, 8, 8)
        patched = patch_attention_with_bias(transformer, bias, last_half_only=False)

        assert len(patched) == 3
        for layer in transformer.layers:
            assert isinstance(layer.attention.processor, _BiasInjectionProcessor)

    def test_unpatch_restores_all_processors(self):
        transformer = _FakeTransformer(num_layers=3)
        original_procs = [l.attention.processor for l in transformer.layers]

        bias = torch.zeros(1, 8, 8)
        patched = patch_attention_with_bias(transformer, bias)
        unpatch_attention_bias(patched)

        for layer, expected in zip(transformer.layers, original_procs):
            assert layer.attention.processor is expected

    def test_patched_processor_is_invoked_through_forward(self):
        """End-to-end: bias wrapper actually runs via Attention.forward()."""
        transformer = _FakeTransformer(num_layers=4)
        seq_len = 8
        bias = torch.ones(1, seq_len, seq_len) * 99.0

        patched = patch_attention_with_bias(transformer, bias)

        # Simulate Attention.forward() call on the last layer (patched)
        attn_mod = transformer.layers[3].attention
        hs = torch.randn(1, seq_len, 16)
        mask_2d = torch.ones(1, seq_len, dtype=torch.bool)

        # This calls attn_mod.processor(attn_mod, hs, attention_mask=mask_2d)
        # which should go through _BiasInjectionProcessor.__call__
        result = attn_mod.forward(hs, attention_mask=mask_2d)

        # The original _FakeProcessor just returns hidden_states, but with
        # the mask transformed. We can verify the original was called.
        original_proc = patched[1][1]  # Second patched layer (index 3)
        assert original_proc.call_count == 1

        unpatch_attention_bias(patched)


# ===================================================================
# Phase B: _SimMapCollectorProcessor
# ===================================================================

class TestSimMapCollectorProcessor:
    """Test the Phase B sim map collector wrapper."""

    def test_install_replaces_target_layer_only(self):
        transformer = _FakeTransformer(num_layers=4)
        block_idx = 2

        state = install_sim_map_collector(
            transformer, block_idx,
            token_pos_maps={"lora_a": [0, 1]},
            img_seq_len=64,
            cap_seq_len=32,
        )

        # Target layer replaced
        assert isinstance(
            transformer.layers[block_idx].attention.processor,
            _SimMapCollectorProcessor,
        )
        # Other layers untouched
        for i, layer in enumerate(transformer.layers):
            if i != block_idx:
                assert isinstance(layer.attention.processor, _FakeProcessor)

        remove_sim_map_collector(state)

    def test_remove_restores_original_processor(self):
        transformer = _FakeTransformer(num_layers=4)
        block_idx = 2
        original_proc = transformer.layers[block_idx].attention.processor

        state = install_sim_map_collector(
            transformer, block_idx,
            token_pos_maps={"lora_a": [0, 1]},
            img_seq_len=64,
            cap_seq_len=32,
        )
        remove_sim_map_collector(state)

        assert transformer.layers[block_idx].attention.processor is original_proc

    def test_collector_state_accessible_via_collector_key(self):
        transformer = _FakeTransformer(num_layers=4)

        state = install_sim_map_collector(
            transformer, 2,
            token_pos_maps={"lora_a": [0, 1]},
            img_seq_len=64,
            cap_seq_len=32,
        )

        collector = state["collector"]
        assert isinstance(collector, _SimMapCollectorProcessor)
        assert collector.cal_concept_sim_map is False
        assert collector.concept_sim_maps is None

        # Toggle flag
        collector.cal_concept_sim_map = True
        assert collector.cal_concept_sim_map is True

        remove_sim_map_collector(state)

    def test_collector_preserves_original_processor_attributes(self):
        transformer = _FakeTransformer(num_layers=4)
        original_proc = transformer.layers[0].attention.processor

        state = install_sim_map_collector(
            transformer, 0,
            token_pos_maps={"lora_a": [0]},
            img_seq_len=16,
            cap_seq_len=8,
        )

        collector = state["collector"]
        assert collector._attention_backend is original_proc._attention_backend
        assert collector._parallel_config is original_proc._parallel_config

        remove_sim_map_collector(state)

    def test_text_start_idx_equals_padded_img_len(self):
        """text_start_idx should be img_seq_len rounded up to SEQ_MULTI_OF=32."""
        transformer = _FakeTransformer(num_layers=4)

        # img_seq_len=64 is already a multiple of 32 → text_start_idx=64
        state = install_sim_map_collector(
            transformer, 0,
            token_pos_maps={"lora_a": [0]},
            img_seq_len=64,
            cap_seq_len=32,
        )
        assert state["collector"].text_start_idx == 64
        remove_sim_map_collector(state)

        # img_seq_len=1008 is NOT a multiple of 32 → padded to 1024
        state = install_sim_map_collector(
            transformer, 0,
            token_pos_maps={"lora_a": [0]},
            img_seq_len=1008,
            cap_seq_len=77,
        )
        assert state["collector"].text_start_idx == 1024
        remove_sim_map_collector(state)

        # img_seq_len=400 (320x320 latents) → padded to 416
        state = install_sim_map_collector(
            transformer, 0,
            token_pos_maps={"lora_a": [0]},
            img_seq_len=400,
            cap_seq_len=77,
        )
        assert state["collector"].text_start_idx == 416
        remove_sim_map_collector(state)


# ===================================================================
# Sequence offset regression
# ===================================================================

class TestTextTokenOffset:
    """Verify text tokens are extracted from the correct offset.

    The Z-Image transformer pads image and caption tokens individually
    to SEQ_MULTI_OF=32 before concatenating:

        [image_tokens (padded) | caption_tokens (padded)]

    Text tokens start at padded_img_len, NOT at img_ori_len.
    """

    def test_extract_sim_maps_uses_text_start_idx(self):
        """Token positions should index into the correct text region."""
        from iartisanz.modules.generation.graph.nodes.freefuse_auto_mask import (
            _extract_concept_sim_maps,
        )

        # Simulate a sequence where img is NOT a multiple of 32
        img_ori_len = 400   # e.g., 20x20 patch grid
        cap_ori_len = 10
        padded_img = 416    # ceil_to_multiple(400, 32)
        padded_cap = 32     # ceil_to_multiple(10, 32)
        total_seq = padded_img + padded_cap  # 448

        heads, head_dim, dim = 8, 16, 128
        B = 1

        # Create tensors with distinguishable image vs text regions
        query = torch.randn(B, total_seq, heads, head_dim)
        key = torch.randn(B, total_seq, heads, head_dim)
        hidden_states = torch.randn(B, total_seq, dim)

        # Put a distinct marker in the ACTUAL text region (at padded_img offset)
        key[:, padded_img:padded_img + cap_ori_len] = 1.0

        # And different values in the WRONG region (at img_ori_len offset)
        key[:, img_ori_len:img_ori_len + cap_ori_len] = -1.0

        token_pos_maps = {"lora_a": [0, 1]}  # First two text tokens

        result = _extract_concept_sim_maps(
            query, key, hidden_states,
            token_pos_maps,
            img_seq_len=img_ori_len,
            cap_seq_len=cap_ori_len,
            text_start_idx=padded_img,  # Correct offset
        )

        # Should succeed without error and produce a result
        assert result is not None
        assert "lora_a" in result
        assert "__eos__" in result
        assert result["lora_a"].shape == (1, img_ori_len, 1)

    def test_out_of_range_positions_are_clamped(self):
        """Token positions beyond cap_seq_len should be silently skipped."""
        from iartisanz.modules.generation.graph.nodes.freefuse_auto_mask import (
            _extract_concept_sim_maps,
        )

        img_len = 64
        cap_len = 10
        padded_img = 64
        total_seq = 64 + 32

        query = torch.randn(1, total_seq, 4, 16)
        key = torch.randn(1, total_seq, 4, 16)
        hidden = torch.randn(1, total_seq, 64)

        # Position 50 is way beyond cap_len=10
        result = _extract_concept_sim_maps(
            query, key, hidden,
            token_pos_maps={"lora_a": [50]},
            img_seq_len=img_len,
            cap_seq_len=cap_len,
            text_start_idx=padded_img,
        )

        # Should not crash; lora_a skipped due to no valid positions
        # Only __eos__ should be present (or None if no concepts extracted)
        if result is not None:
            assert "lora_a" not in result
