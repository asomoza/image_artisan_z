"""Regression tests for LoRA format conversion.

All tests use synthetic in-memory state dicts — no files, no models.
"""

import pytest
import torch

from iartisanz.modules.generation.graph.nodes.lora_conversion import (
    _convert_flux2_lora,
    _convert_flux2_lora_to_diffusers,
    _convert_non_diffusers_z_image_lora_to_diffusers,
    _convert_zimage_lora,
    _is_diffusers_format,
    _is_lokr_format,
    _map_flux2_lokr_layer_to_targets,
    _map_zimage_lokr_layer_to_targets,
    _normalize_flux2_lora_keys,
    _strip_lora_unet_prefix,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _w(shape=(4, 4)):
    return torch.randn(*shape)


def _alpha(value, rank=4):
    return torch.tensor(float(value))


# ===========================================================================
# _is_diffusers_format
# ===========================================================================

class TestIsDiffusersFormat:
    def test_peft_keys_true(self):
        sd = {
            "transformer.layers.0.attn.to_q.lora_A.weight": _w(),
            "transformer.layers.0.attn.to_q.lora_B.weight": _w(),
        }
        assert _is_diffusers_format(sd) is True

    def test_kohya_keys_false(self):
        sd = {
            "layers.0.attn.to_q.lora_down.weight": _w(),
            "layers.0.attn.to_q.lora_up.weight": _w(),
        }
        assert _is_diffusers_format(sd) is False

    def test_transformer_prefix_with_lora_down_false(self):
        sd = {
            "transformer.layers.0.attn.to_q.lora_down.weight": _w(),
            "transformer.layers.0.attn.to_q.lora_up.weight": _w(),
        }
        assert _is_diffusers_format(sd) is False


# ===========================================================================
# _is_lokr_format
# ===========================================================================

class TestIsLokrFormat:
    def test_lokr_w_detected(self):
        sd = {"layers.0.lokr_w1": _w(), "layers.0.lokr_w2": _w()}
        assert _is_lokr_format(sd) is True

    def test_standard_keys_not_detected(self):
        sd = {"layers.0.lora_A.weight": _w(), "layers.0.lora_B.weight": _w()}
        assert _is_lokr_format(sd) is False


# ===========================================================================
# _convert_non_diffusers_z_image_lora_to_diffusers
# ===========================================================================

class TestConvertNonDiffusersZImage:
    def test_diffusion_model_prefix_stripped(self):
        sd = {
            "diffusion_model.layers.0.attn.to_q.lora_down.weight": _w(),
            "diffusion_model.layers.0.attn.to_q.lora_up.weight": _w(),
            "diffusion_model.layers.0.attn.to_q.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert all(k.startswith("transformer.") for k in result)
        assert not any("diffusion_model" in k for k in result)

    def test_lora_unet_single_underscore(self):
        """lora_unet_ prefix with protected n-grams: to_q/to_k/to_v/to_out/feed_forward."""
        sd = {
            "lora_unet_layers_0_attention_to_q.lora_down.weight": _w(),
            "lora_unet_layers_0_attention_to_q.lora_up.weight": _w(),
            "lora_unet_layers_0_attention_to_q.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert "transformer.layers.0.attention.to_q.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_q.lora_B.weight" in result

    def test_lora_unet_double_underscore(self):
        """Fun-distilled format uses lora_unet__ (double underscore)."""
        sd = {
            "lora_unet__layers_0_attention_to_q.lora_down.weight": _w(),
            "lora_unet__layers_0_attention_to_q.lora_up.weight": _w(),
            "lora_unet__layers_0_attention_to_q.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert "transformer.layers.0.attention.to_q.lora_A.weight" in result

    def test_context_refiner_protected(self):
        """context_refiner must not be split into context.refiner."""
        sd = {
            "lora_unet_context_refiner_0_attention_to_q.lora_down.weight": _w(),
            "lora_unet_context_refiner_0_attention_to_q.lora_up.weight": _w(),
            "lora_unet_context_refiner_0_attention_to_q.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        key = "transformer.context_refiner.0.attention.to_q.lora_A.weight"
        assert key in result, f"Expected {key}, got {list(result.keys())}"

    def test_noise_refiner_protected(self):
        """noise_refiner must not be split into noise.refiner."""
        sd = {
            "lora_unet_noise_refiner_0_attention_to_q.lora_down.weight": _w(),
            "lora_unet_noise_refiner_0_attention_to_q.lora_up.weight": _w(),
            "lora_unet_noise_refiner_0_attention_to_q.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        key = "transformer.noise_refiner.0.attention.to_q.lora_A.weight"
        assert key in result, f"Expected {key}, got {list(result.keys())}"

    def test_out_normalized_to_to_out_0(self):
        """.out → .to_out.0 for attention output."""
        sd = {
            "layers.0.attention.out.lora_down.weight": _w(),
            "layers.0.attention.out.lora_up.weight": _w(),
            "layers.0.attention.out.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert "transformer.layers.0.attention.to_out.0.lora_A.weight" in result

    def test_existing_to_out_not_doubled(self):
        """.to_out should not become .to_out.0.to_out.0."""
        sd = {
            "layers.0.attention.to_out.lora_down.weight": _w(),
            "layers.0.attention.to_out.lora_up.weight": _w(),
            "layers.0.attention.to_out.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        # Should keep .to_out as-is (normalize_out_key skips keys with .to_out)
        assert any("to_out" in k for k in result)
        assert not any("to_out.0.to_out" in k for k in result)

    def test_default_prefix_stripped(self):
        sd = {
            "default.layers.0.attn.to_q.lora_down.weight": _w(),
            "default.layers.0.attn.to_q.lora_up.weight": _w(),
            "default.layers.0.attn.to_q.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert not any("default" in k for k in result)

    def test_alpha_scaling(self):
        """lora_A * lora_B product = lora_down * lora_up * (alpha/rank)."""
        rank = 4
        alpha_val = 2.0
        down = torch.ones(rank, 8)
        up = torch.ones(16, rank)

        sd = {
            "layers.0.to_q.lora_down.weight": down.clone(),
            "layers.0.to_q.lora_up.weight": up.clone(),
            "layers.0.to_q.alpha": _alpha(alpha_val),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)

        a_key = "transformer.layers.0.to_q.lora_A.weight"
        b_key = "transformer.layers.0.to_q.lora_B.weight"
        product = result[b_key] @ result[a_key]
        expected = up @ (down * (alpha_val / rank))
        assert torch.allclose(product, expected, atol=1e-6)

    def test_already_diffusers_passthrough(self):
        """Keys already in lora_A/lora_B format pass through."""
        sd = {
            "layers.0.to_q.lora_A.weight": _w(),
            "layers.0.to_q.lora_B.weight": _w(),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert "transformer.layers.0.to_q.lora_A.weight" in result
        assert "transformer.layers.0.to_q.lora_B.weight" in result

    def test_all_output_keys_prefixed_with_transformer(self):
        sd = {
            "layers.0.to_q.lora_down.weight": _w(),
            "layers.0.to_q.lora_up.weight": _w(),
            "layers.0.to_q.alpha": _alpha(4),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert all(k.startswith("transformer.") for k in result)

    def test_unexpected_remaining_keys_raise(self):
        sd = {
            "layers.0.to_q.lora_down.weight": _w(),
            "layers.0.to_q.lora_up.weight": _w(),
            "layers.0.to_q.alpha": _alpha(4),
            "unexpected_orphan_key": _w(),
        }
        with pytest.raises(ValueError, match="state_dict.*should be empty"):
            _convert_non_diffusers_z_image_lora_to_diffusers(sd)

    def test_out_normalized_with_lora_a_b_suffix(self):
        """.out → .to_out.0 must also work for lora_A/lora_B keys, not just lora_down/lora_up."""
        sd = {
            "diffusion_model.layers.0.attention.out.lora_A.weight": _w((32, 3840)),
            "diffusion_model.layers.0.attention.out.lora_B.weight": _w((3840, 32)),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)
        assert "transformer.layers.0.attention.to_out.0.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_out.0.lora_B.weight" in result

    def test_qkv_split_into_to_q_to_k_to_v(self):
        """Fused attention.qkv must be split into separate to_q/to_k/to_v keys."""
        hidden = 12  # small for testing, must be divisible by 3
        rank = 4
        sd = {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": _w((rank, hidden)),
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": _w((hidden * 3, rank)),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)

        # Must have separate Q, K, V keys
        assert "transformer.layers.0.attention.to_q.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_k.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_v.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_q.lora_B.weight" in result
        assert "transformer.layers.0.attention.to_k.lora_B.weight" in result
        assert "transformer.layers.0.attention.to_v.lora_B.weight" in result

        # No fused qkv keys should remain
        assert not any(".qkv." in k for k in result)

        # lora_B must be split (each 1/3 of original)
        assert result["transformer.layers.0.attention.to_q.lora_B.weight"].shape == (hidden, rank)
        assert result["transformer.layers.0.attention.to_k.lora_B.weight"].shape == (hidden, rank)
        assert result["transformer.layers.0.attention.to_v.lora_B.weight"].shape == (hidden, rank)

        # lora_A is shared (same shape as original)
        assert result["transformer.layers.0.attention.to_q.lora_A.weight"].shape == (rank, hidden)

    def test_qkv_split_lora_b_chunks_are_correct(self):
        """QKV split must produce the correct chunk values from the fused tensor."""
        rank = 2
        hidden = 6
        fused_b = torch.arange(hidden * 3 * rank, dtype=torch.float32).reshape(hidden * 3, rank)
        sd = {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": _w((rank, hidden)),
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": fused_b,
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)

        q_b = result["transformer.layers.0.attention.to_q.lora_B.weight"]
        k_b = result["transformer.layers.0.attention.to_k.lora_B.weight"]
        v_b = result["transformer.layers.0.attention.to_v.lora_B.weight"]

        expected_q, expected_k, expected_v = fused_b.chunk(3, dim=0)
        torch.testing.assert_close(q_b, expected_q)
        torch.testing.assert_close(k_b, expected_k)
        torch.testing.assert_close(v_b, expected_v)

    def test_diffusion_model_prefix_with_qkv_and_out_combined(self):
        """Full scenario: diffusion_model. prefix + fused qkv + .out + feed_forward."""
        rank = 4
        hidden = 12
        sd = {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": _w((rank, hidden)),
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": _w((hidden * 3, rank)),
            "diffusion_model.layers.0.attention.out.lora_A.weight": _w((rank, hidden)),
            "diffusion_model.layers.0.attention.out.lora_B.weight": _w((hidden, rank)),
            "diffusion_model.layers.0.feed_forward.w1.lora_A.weight": _w((rank, hidden)),
            "diffusion_model.layers.0.feed_forward.w1.lora_B.weight": _w((hidden, rank)),
            "diffusion_model.context_refiner.0.attention.qkv.lora_A.weight": _w((rank, hidden)),
            "diffusion_model.context_refiner.0.attention.qkv.lora_B.weight": _w((hidden * 3, rank)),
            "diffusion_model.noise_refiner.0.attention.out.lora_A.weight": _w((rank, hidden)),
            "diffusion_model.noise_refiner.0.attention.out.lora_B.weight": _w((hidden, rank)),
        }
        result = _convert_non_diffusers_z_image_lora_to_diffusers(sd)

        # QKV split
        assert "transformer.layers.0.attention.to_q.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_k.lora_B.weight" in result
        assert "transformer.layers.0.attention.to_v.lora_A.weight" in result
        # .out → .to_out.0
        assert "transformer.layers.0.attention.to_out.0.lora_A.weight" in result
        # feed_forward preserved
        assert "transformer.layers.0.feed_forward.w1.lora_A.weight" in result
        # context_refiner QKV split
        assert "transformer.context_refiner.0.attention.to_q.lora_A.weight" in result
        assert "transformer.context_refiner.0.attention.to_k.lora_A.weight" in result
        # noise_refiner .out → .to_out.0
        assert "transformer.noise_refiner.0.attention.to_out.0.lora_A.weight" in result
        # No original names remain
        assert not any(".qkv." in k for k in result)
        assert not any(k.endswith(".attention.out.lora_A.weight") for k in result)


# ===========================================================================
# _convert_zimage_lora
# ===========================================================================

class TestTransformerPrefixedKohyaZImage:
    """Regression: Z-Image LoRA with transformer. prefix but kohya lora_down/lora_up naming.

    Keys like transformer.layers.N.attention.to_q.lora_down.weight — the transformer.
    prefix must be stripped before conversion to avoid double-prefixing
    (transformer.transformer.layers...).
    """

    def _build_sd(self, num_layers=2):
        sd = {}
        rank = 32
        hidden = 3840
        ffn = 10240
        for i in range(num_layers):
            for module, down_in, up_out in [
                ("attention.to_q", hidden, hidden),
                ("attention.to_k", hidden, hidden),
                ("attention.to_v", hidden, hidden),
                ("attention.to_out.0", hidden, hidden),
                ("feed_forward.w1", hidden, ffn),
                ("feed_forward.w2", ffn, hidden),
                ("feed_forward.w3", hidden, ffn),
            ]:
                prefix = f"transformer.layers.{i}.{module}"
                sd[f"{prefix}.lora_down.weight"] = torch.randn(rank, down_in)
                sd[f"{prefix}.lora_up.weight"] = torch.randn(up_out, rank)
                sd[f"{prefix}.alpha"] = torch.tensor(float(rank))
        return sd

    def test_no_double_transformer_prefix(self):
        result = _convert_zimage_lora(self._build_sd())
        assert all(k.startswith("transformer.") for k in result)
        assert not any(k.startswith("transformer.transformer.") for k in result)

    def test_all_keys_converted(self):
        result = _convert_zimage_lora(self._build_sd())
        assert all("lora_A" in k or "lora_B" in k for k in result)
        assert not any("lora_down" in k for k in result)

    def test_expected_keys_present(self):
        result = _convert_zimage_lora(self._build_sd(num_layers=1))
        assert "transformer.layers.0.attention.to_q.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_q.lora_B.weight" in result
        assert "transformer.layers.0.feed_forward.w1.lora_A.weight" in result
        assert "transformer.layers.0.attention.to_out.0.lora_A.weight" in result

    def test_layer_count(self):
        result = _convert_zimage_lora(self._build_sd(num_layers=30))
        layer_indices = {
            int(k.split("layers.")[1].split(".")[0])
            for k in result if "layers." in k
        }
        assert layer_indices == set(range(30))


class TestConvertZImageLora:
    def test_passthrough_diffusers_format(self):
        sd = {
            "transformer.layers.0.to_q.lora_A.weight": _w(),
            "transformer.layers.0.to_q.lora_B.weight": _w(),
        }
        result = _convert_zimage_lora(sd)
        assert result is sd  # Same object (no copy)

    def test_converts_non_diffusers(self):
        sd = {
            "layers.0.to_q.lora_down.weight": _w(),
            "layers.0.to_q.lora_up.weight": _w(),
            "layers.0.to_q.alpha": _alpha(4),
        }
        result = _convert_zimage_lora(sd)
        assert "transformer.layers.0.to_q.lora_A.weight" in result


# ===========================================================================
# _normalize_flux2_lora_keys
# ===========================================================================

class TestNormalizeFlux2LoraKeys:
    def test_lora_down_to_lora_a(self):
        sd = {
            "block.0.lora_down.weight": _w(),
            "block.0.lora_up.weight": _w(),
        }
        result = _normalize_flux2_lora_keys(sd)
        assert "block.0.lora_A.weight" in result
        assert "block.0.lora_B.weight" in result

    def test_alpha_baking(self):
        rank = 4
        alpha_val = 2.0
        down = torch.ones(rank, 8)
        up = torch.ones(16, rank)
        sd = {
            "block.0.lora_down.weight": down.clone(),
            "block.0.lora_up.weight": up.clone(),
            "block.0.alpha": _alpha(alpha_val),
        }
        result = _normalize_flux2_lora_keys(sd)
        expected_scale = alpha_val / rank
        assert torch.allclose(result["block.0.lora_A.weight"], down * expected_scale)
        assert torch.allclose(result["block.0.lora_B.weight"], up)

    def test_no_alpha_scale_1(self):
        down = torch.ones(4, 8)
        up = torch.ones(16, 4)
        sd = {
            "block.0.lora_down.weight": down.clone(),
            "block.0.lora_up.weight": up.clone(),
        }
        result = _normalize_flux2_lora_keys(sd)
        assert torch.allclose(result["block.0.lora_A.weight"], down)

    def test_mixed_format_preserves_existing(self):
        existing_a = _w()
        sd = {
            "block.0.lora_A.weight": existing_a,
            "block.0.lora_B.weight": _w(),
            "block.1.lora_down.weight": _w(),
            "block.1.lora_up.weight": _w(),
        }
        result = _normalize_flux2_lora_keys(sd)
        assert result["block.0.lora_A.weight"] is existing_a
        assert "block.1.lora_A.weight" in result


# ===========================================================================
# _strip_lora_unet_prefix
# ===========================================================================

class TestStripLoraUnetPrefix:
    def test_single_blocks(self):
        sd = {"lora_unet_single_blocks_8_linear1.lora_A.weight": _w()}
        result = _strip_lora_unet_prefix(sd)
        assert "single_blocks.8.linear1.lora_A.weight" in result

    def test_double_blocks_compound_names(self):
        sd = {"lora_unet_double_blocks_5_img_attn_qkv.lora_A.weight": _w()}
        result = _strip_lora_unet_prefix(sd)
        assert "double_blocks.5.img_attn.qkv.lora_A.weight" in result

    def test_non_matching_passthrough(self):
        sd = {"some_other_key.weight": _w()}
        result = _strip_lora_unet_prefix(sd)
        assert "some_other_key.weight" in result


# ===========================================================================
# _convert_flux2_lora_to_diffusers
# ===========================================================================

class TestConvertFlux2LoraToDiffusers:
    def _make_single_block_sd(self, idx=0):
        return {
            f"single_blocks.{idx}.linear1.lora_A.weight": _w(),
            f"single_blocks.{idx}.linear1.lora_B.weight": _w((12, 4)),
            f"single_blocks.{idx}.linear2.lora_A.weight": _w(),
            f"single_blocks.{idx}.linear2.lora_B.weight": _w(),
        }

    def _make_double_block_sd(self, idx=0):
        sd = {}
        for attn in ("img_attn", "txt_attn"):
            sd[f"double_blocks.{idx}.{attn}.qkv.lora_A.weight"] = _w()
            sd[f"double_blocks.{idx}.{attn}.qkv.lora_B.weight"] = _w((12, 4))
            sd[f"double_blocks.{idx}.{attn}.proj.lora_A.weight"] = _w()
            sd[f"double_blocks.{idx}.{attn}.proj.lora_B.weight"] = _w()
        for mlp_pre in ("img_mlp", "txt_mlp"):
            for sub in ("0", "2"):
                sd[f"double_blocks.{idx}.{mlp_pre}.{sub}.lora_A.weight"] = _w()
                sd[f"double_blocks.{idx}.{mlp_pre}.{sub}.lora_B.weight"] = _w()
        return sd

    def test_single_block_mapping(self):
        sd = self._make_single_block_sd(3)
        result = _convert_flux2_lora_to_diffusers(sd)
        assert "transformer.single_transformer_blocks.3.attn.to_qkv_mlp_proj.lora_A.weight" in result
        assert "transformer.single_transformer_blocks.3.attn.to_out.lora_B.weight" in result

    def test_double_block_qkv_split(self):
        sd = self._make_double_block_sd(1)
        result = _convert_flux2_lora_to_diffusers(sd)
        # img_attn QKV split (lora_B gets chunked into 3)
        assert "transformer.transformer_blocks.1.attn.to_q.lora_B.weight" in result
        assert "transformer.transformer_blocks.1.attn.to_k.lora_B.weight" in result
        assert "transformer.transformer_blocks.1.attn.to_v.lora_B.weight" in result
        # txt_attn QKV split
        assert "transformer.transformer_blocks.1.attn.add_q_proj.lora_B.weight" in result

    def test_proj_and_mlp_mapping(self):
        sd = self._make_double_block_sd(0)
        result = _convert_flux2_lora_to_diffusers(sd)
        assert "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight" in result
        assert "transformer.transformer_blocks.0.attn.to_add_out.lora_A.weight" in result
        assert "transformer.transformer_blocks.0.ff.linear_in.lora_A.weight" in result
        assert "transformer.transformer_blocks.0.ff_context.linear_out.lora_B.weight" in result

    def test_root_mappings(self):
        sd = {
            "txt_in.lora_A.weight": _w(),
            "txt_in.lora_B.weight": _w(),
        }
        result = _convert_flux2_lora_to_diffusers(sd)
        assert "transformer.context_embedder.lora_A.weight" in result

    def test_transformer_prefix_on_all(self):
        sd = self._make_single_block_sd(0)
        result = _convert_flux2_lora_to_diffusers(sd)
        assert all(k.startswith("transformer.") for k in result)

    def test_unexpected_keys_raise(self):
        sd = self._make_single_block_sd(0)
        sd["orphan_key.lora_A.weight"] = _w()
        with pytest.raises(ValueError, match="Unexpected keys"):
            _convert_flux2_lora_to_diffusers(sd)

    def test_attention_only_double_block(self):
        """LoRAs that only target attention (no MLP) must convert without error."""
        sd = {}
        for idx in range(5):
            for attn in ("img_attn", "txt_attn"):
                sd[f"double_blocks.{idx}.{attn}.qkv.lora_A.weight"] = _w()
                sd[f"double_blocks.{idx}.{attn}.qkv.lora_B.weight"] = _w((12, 4))
                sd[f"double_blocks.{idx}.{attn}.proj.lora_A.weight"] = _w()
                sd[f"double_blocks.{idx}.{attn}.proj.lora_B.weight"] = _w()
        result = _convert_flux2_lora_to_diffusers(sd)
        assert all(k.startswith("transformer.") for k in result)
        # Attention keys present
        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in result
        assert "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight" in result
        # MLP keys absent
        assert not any("ff.linear_in" in k for k in result)
        assert not any("ff_context" in k for k in result)

    def test_partial_single_block_linear1_only(self):
        """LoRAs that only target linear1 (not linear2) in single blocks."""
        sd = {
            "single_blocks.0.linear1.lora_A.weight": _w(),
            "single_blocks.0.linear1.lora_B.weight": _w((12, 4)),
        }
        result = _convert_flux2_lora_to_diffusers(sd)
        assert "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight" in result
        assert not any("to_out" in k for k in result)

    def test_base_model_model_prefix_flux2_klein_4b(self):
        """PEFT-format LoRA with base_model.model. prefix — Flux2 Klein 4B outpaint style.

        5 double blocks (attn only + proj + modulation), 20 single blocks,
        root mappings (txt_in, img_in, time_in, final_layer, modulations).
        """
        sd = {}
        # 5 double blocks — attention only (no MLP)
        for idx in range(5):
            for attn in ("img_attn", "txt_attn"):
                sd[f"base_model.model.double_blocks.{idx}.{attn}.qkv.lora_A.weight"] = _w()
                sd[f"base_model.model.double_blocks.{idx}.{attn}.qkv.lora_B.weight"] = _w((12, 4))
                sd[f"base_model.model.double_blocks.{idx}.{attn}.proj.lora_A.weight"] = _w()
                sd[f"base_model.model.double_blocks.{idx}.{attn}.proj.lora_B.weight"] = _w()
        # 20 single blocks
        for idx in range(20):
            sd[f"base_model.model.single_blocks.{idx}.linear1.lora_A.weight"] = _w()
            sd[f"base_model.model.single_blocks.{idx}.linear1.lora_B.weight"] = _w((12, 4))
            sd[f"base_model.model.single_blocks.{idx}.linear2.lora_A.weight"] = _w()
            sd[f"base_model.model.single_blocks.{idx}.linear2.lora_B.weight"] = _w()
        # Root mappings
        root_keys = [
            "double_stream_modulation_txt.lin",
            "double_stream_modulation_img.lin",
            "single_stream_modulation.lin",
            "txt_in",
            "img_in",
            "final_layer.linear",
            "time_in.in_layer",
            "time_in.out_layer",
        ]
        for rk in root_keys:
            sd[f"base_model.model.{rk}.lora_A.weight"] = _w()
            sd[f"base_model.model.{rk}.lora_B.weight"] = _w()

        result = _convert_flux2_lora_to_diffusers(sd)
        assert all(k.startswith("transformer.") for k in result)
        # Double block attention converted
        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in result
        assert "transformer.transformer_blocks.4.attn.to_out.0.lora_B.weight" in result
        # Single block converted
        assert "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight" in result
        assert "transformer.single_transformer_blocks.19.attn.to_out.lora_B.weight" in result
        # Root mappings converted
        assert "transformer.context_embedder.lora_A.weight" in result
        assert "transformer.x_embedder.lora_A.weight" in result
        assert "transformer.proj_out.lora_A.weight" in result
        assert "transformer.time_guidance_embed.timestep_embedder.linear_1.lora_A.weight" in result
        assert "transformer.single_stream_modulation.linear.lora_B.weight" in result
        # No MLP keys
        assert not any("ff.linear_in" in k for k in result)

    def test_diffusion_model_prefix_attention_only(self):
        """ComfyUI-format LoRA with diffusion_model. prefix and attention-only layers."""
        sd = {
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_A.weight": _w(),
            "diffusion_model.double_blocks.0.img_attn.qkv.lora_B.weight": _w((12, 4)),
            "diffusion_model.double_blocks.0.txt_attn.qkv.lora_A.weight": _w(),
            "diffusion_model.double_blocks.0.txt_attn.qkv.lora_B.weight": _w((12, 4)),
            "diffusion_model.double_blocks.0.img_attn.proj.lora_A.weight": _w(),
            "diffusion_model.double_blocks.0.img_attn.proj.lora_B.weight": _w(),
            "diffusion_model.double_blocks.0.txt_attn.proj.lora_A.weight": _w(),
            "diffusion_model.double_blocks.0.txt_attn.proj.lora_B.weight": _w(),
            "diffusion_model.single_blocks.0.linear1.lora_A.weight": _w(),
            "diffusion_model.single_blocks.0.linear1.lora_B.weight": _w((12, 4)),
            "diffusion_model.single_blocks.0.linear2.lora_A.weight": _w(),
            "diffusion_model.single_blocks.0.linear2.lora_B.weight": _w(),
        }
        result = _convert_flux2_lora_to_diffusers(sd)
        assert all(k.startswith("transformer.") for k in result)
        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in result
        assert not any("ff.linear_in" in k for k in result)


# ===========================================================================
# _convert_flux2_lora
# ===========================================================================

class TestConvertFlux2Lora:
    def test_passthrough_diffusers(self):
        sd = {
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight": _w(),
            "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight": _w(),
        }
        result = _convert_flux2_lora(sd)
        assert result is sd

    def test_kohya_plus_original_format(self):
        """Full pipeline: kohya naming + original block format."""
        sd = {
            "single_blocks.0.linear1.lora_down.weight": _w(),
            "single_blocks.0.linear1.lora_up.weight": _w(),
            "single_blocks.0.linear1.alpha": _alpha(4),
            "single_blocks.0.linear2.lora_down.weight": _w(),
            "single_blocks.0.linear2.lora_up.weight": _w(),
            "single_blocks.0.linear2.alpha": _alpha(4),
        }
        result = _convert_flux2_lora(sd)
        assert "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight" in result
        assert all(k.startswith("transformer.") for k in result)

    def test_peft_base_model_prefix(self):
        """Full pipeline: PEFT base_model.model. prefix with lora_A/lora_B naming."""
        sd = {
            "base_model.model.single_blocks.0.linear1.lora_A.weight": _w(),
            "base_model.model.single_blocks.0.linear1.lora_B.weight": _w((12, 4)),
            "base_model.model.single_blocks.0.linear2.lora_A.weight": _w(),
            "base_model.model.single_blocks.0.linear2.lora_B.weight": _w(),
            "base_model.model.double_blocks.0.img_attn.qkv.lora_A.weight": _w(),
            "base_model.model.double_blocks.0.img_attn.qkv.lora_B.weight": _w((12, 4)),
            "base_model.model.double_blocks.0.img_attn.proj.lora_A.weight": _w(),
            "base_model.model.double_blocks.0.img_attn.proj.lora_B.weight": _w(),
            "base_model.model.double_blocks.0.txt_attn.qkv.lora_A.weight": _w(),
            "base_model.model.double_blocks.0.txt_attn.qkv.lora_B.weight": _w((12, 4)),
            "base_model.model.double_blocks.0.txt_attn.proj.lora_A.weight": _w(),
            "base_model.model.double_blocks.0.txt_attn.proj.lora_B.weight": _w(),
        }
        result = _convert_flux2_lora(sd)
        assert all(k.startswith("transformer.") for k in result)
        assert "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight" in result
        assert "transformer.transformer_blocks.0.attn.to_q.lora_A.weight" in result


# ===========================================================================
# _map_zimage_lokr_layer_to_targets
# ===========================================================================

class TestMapZImageLokrLayerToTargets:
    def test_layers_identity(self):
        targets = _map_zimage_lokr_layer_to_targets("layers.5.attn.to_q")
        assert targets == [("layers.5.attn.to_q", None)]

    def test_context_refiner_identity(self):
        targets = _map_zimage_lokr_layer_to_targets("context_refiner.2.attn.to_q")
        assert targets == [("context_refiner.2.attn.to_q", None)]

    def test_noise_refiner_identity(self):
        targets = _map_zimage_lokr_layer_to_targets("noise_refiner.0.ff.linear")
        assert targets == [("noise_refiner.0.ff.linear", None)]

    def test_unknown_empty(self):
        assert _map_zimage_lokr_layer_to_targets("unknown.block") == []


# ===========================================================================
# _map_flux2_lokr_layer_to_targets
# ===========================================================================

class TestMapFlux2LokrLayerToTargets:
    def test_single_block_linear1(self):
        targets = _map_flux2_lokr_layer_to_targets("single_blocks.3.linear1")
        assert targets == [("single_transformer_blocks.3.attn.to_qkv_mlp_proj", None)]

    def test_double_block_img_qkv_split(self):
        targets = _map_flux2_lokr_layer_to_targets("double_blocks.1.img_attn.qkv")
        assert len(targets) == 3
        assert targets[0] == ("transformer_blocks.1.attn.to_q", 0)
        assert targets[1] == ("transformer_blocks.1.attn.to_k", 1)
        assert targets[2] == ("transformer_blocks.1.attn.to_v", 2)

    def test_double_block_txt_qkv_split(self):
        targets = _map_flux2_lokr_layer_to_targets("double_blocks.0.txt_attn.qkv")
        assert targets[0] == ("transformer_blocks.0.attn.add_q_proj", 0)

    def test_proj_mapping(self):
        targets = _map_flux2_lokr_layer_to_targets("double_blocks.2.img_attn.proj")
        assert targets == [("transformer_blocks.2.attn.to_out.0", None)]

    def test_root_mapping(self):
        targets = _map_flux2_lokr_layer_to_targets("txt_in")
        assert targets == [("context_embedder", None)]

    def test_unknown_empty(self):
        assert _map_flux2_lokr_layer_to_targets("totally_unknown") == []


# ===========================================================================
# Regression: kohya lora_unet_ prefix with lora_down/lora_up (Klein 9B)
# ===========================================================================

class TestKohyaLoraUnetFlux2Klein9B:
    """Regression test for kohya-format Flux2 Klein 9B LoRA.

    Key format: lora_unet_{block_type}_{idx}_{layer}.lora_down.weight / .lora_up.weight / .alpha
    8 double blocks (attn+mlp for img+txt), 24 single blocks (linear1+linear2).
    Rank 64, alpha 32.0.
    """

    RANK = 64
    HIDDEN = 4096
    ALPHA = 32.0
    NUM_DOUBLE = 8
    NUM_SINGLE = 24

    def _build_sd(self):
        sd = {}
        r, h, a = self.RANK, self.HIDDEN, self.ALPHA
        for idx in range(self.NUM_DOUBLE):
            for layer, up_dim in [
                ("img_attn_proj", h),
                ("img_attn_qkv", h * 3),
                ("img_mlp_0", h * 6),
                ("img_mlp_2", h),
                ("txt_attn_proj", h),
                ("txt_attn_qkv", h * 3),
                ("txt_mlp_0", h * 6),
                ("txt_mlp_2", h),
            ]:
                down_in = h if "mlp_2" not in layer else h * 3
                prefix = f"lora_unet_double_blocks_{idx}_{layer}"
                sd[f"{prefix}.lora_down.weight"] = torch.randn(r, down_in)
                sd[f"{prefix}.lora_up.weight"] = torch.randn(up_dim, r)
                sd[f"{prefix}.alpha"] = torch.tensor(a)

        for idx_s in range(self.NUM_SINGLE):
            for layer, down_in, up_dim in [
                ("linear1", h, h * 9),
                ("linear2", h * 4, h),
            ]:
                prefix = f"lora_unet_single_blocks_{idx_s}_{layer}"
                sd[f"{prefix}.lora_down.weight"] = torch.randn(r, down_in)
                sd[f"{prefix}.lora_up.weight"] = torch.randn(up_dim, r)
                sd[f"{prefix}.alpha"] = torch.tensor(a)

        return sd

    def test_all_keys_converted_to_diffusers(self):
        result = _convert_flux2_lora(self._build_sd())
        assert all(k.startswith("transformer.") for k in result)
        assert all("lora_A" in k or "lora_B" in k for k in result)
        assert not any("lora_unet" in k for k in result)

    def test_double_block_count(self):
        result = _convert_flux2_lora(self._build_sd())
        indices = {
            int(k.split("transformer_blocks.")[1].split(".")[0])
            for k in result
            if "transformer_blocks." in k and "single_transformer_blocks." not in k
        }
        assert indices == set(range(self.NUM_DOUBLE))

    def test_single_block_count(self):
        result = _convert_flux2_lora(self._build_sd())
        indices = {
            int(k.split("single_transformer_blocks.")[1].split(".")[0])
            for k in result if "single_transformer_blocks." in k
        }
        assert indices == set(range(self.NUM_SINGLE))

    def test_qkv_split_in_double_blocks(self):
        result = _convert_flux2_lora(self._build_sd())
        for idx in range(self.NUM_DOUBLE):
            tb = f"transformer.transformer_blocks.{idx}"
            # img_attn QKV split
            assert f"{tb}.attn.to_q.lora_A.weight" in result
            assert f"{tb}.attn.to_k.lora_A.weight" in result
            assert f"{tb}.attn.to_v.lora_A.weight" in result
            assert f"{tb}.attn.to_q.lora_B.weight" in result
            # txt_attn QKV split
            assert f"{tb}.attn.add_q_proj.lora_A.weight" in result
            assert f"{tb}.attn.add_k_proj.lora_B.weight" in result
            assert f"{tb}.attn.add_v_proj.lora_B.weight" in result

    def test_proj_and_mlp_in_double_blocks(self):
        result = _convert_flux2_lora(self._build_sd())
        for idx in range(self.NUM_DOUBLE):
            tb = f"transformer.transformer_blocks.{idx}"
            assert f"{tb}.attn.to_out.0.lora_A.weight" in result
            assert f"{tb}.attn.to_add_out.lora_A.weight" in result
            assert f"{tb}.ff.linear_in.lora_A.weight" in result
            assert f"{tb}.ff.linear_out.lora_B.weight" in result
            assert f"{tb}.ff_context.linear_in.lora_A.weight" in result
            assert f"{tb}.ff_context.linear_out.lora_B.weight" in result

    def test_single_block_mappings(self):
        result = _convert_flux2_lora(self._build_sd())
        for idx in range(self.NUM_SINGLE):
            stb = f"transformer.single_transformer_blocks.{idx}"
            assert f"{stb}.attn.to_qkv_mlp_proj.lora_A.weight" in result
            assert f"{stb}.attn.to_qkv_mlp_proj.lora_B.weight" in result
            assert f"{stb}.attn.to_out.lora_A.weight" in result
            assert f"{stb}.attn.to_out.lora_B.weight" in result

    def test_no_fused_qkv_keys_remain(self):
        result = _convert_flux2_lora(self._build_sd())
        assert not any(".qkv." in k for k in result)

    def test_alpha_scaling_applied(self):
        """Alpha/rank scaling must be baked into lora_A weights."""
        r = self.RANK
        scale = self.ALPHA / r  # 32/64 = 0.5
        down = torch.ones(r, self.HIDDEN)
        sd = {
            "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight": down.clone(),
            "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight": torch.ones(self.HIDDEN, r),
            "lora_unet_double_blocks_0_img_attn_proj.alpha": torch.tensor(self.ALPHA),
        }
        result = _convert_flux2_lora(sd)
        a_key = "transformer.transformer_blocks.0.attn.to_out.0.lora_A.weight"
        assert torch.allclose(result[a_key], down * scale)
