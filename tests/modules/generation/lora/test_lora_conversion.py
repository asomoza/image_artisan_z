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
