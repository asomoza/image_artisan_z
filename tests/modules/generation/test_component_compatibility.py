"""Tests for component compatibility mapping and architecture detection."""
from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from iartisanz.modules.generation.component_compatibility import (
    ARCHITECTURE_COMPATIBILITY,
    _ARCHITECTURE_SIGNATURES,
    detect_transformer_architecture,
)


class TestArchitectureCompatibility:
    def test_zimage_transformer_has_text_encoder_compat(self):
        compat = ARCHITECTURE_COMPATIBILITY["ZImageTransformer2DModel"]
        assert "Qwen3Model" in compat["text_encoder"]
        assert "Qwen3ForCausalLM" in compat["text_encoder"]

    def test_zimage_transformer_has_vae_compat(self):
        compat = ARCHITECTURE_COMPATIBILITY["ZImageTransformer2DModel"]
        assert "AutoencoderKL" in compat["vae"]

    def test_zimage_transformer_has_tokenizer_compat(self):
        compat = ARCHITECTURE_COMPATIBILITY["ZImageTransformer2DModel"]
        assert "Qwen2Tokenizer" in compat["tokenizer"]


class TestDetectTransformerArchitecture:
    def test_detects_zimage_transformer(self, tmp_path):
        """Safetensors with ZImage signature keys should be detected."""
        # Create a safetensors file with the signature keys
        tensors = {
            "context_refiner.weight": torch.randn(4, 4),
            "all_final_layer.adaLN_modulation.1.weight": torch.randn(4, 4),
            "some.other.weight": torch.randn(2, 2),
        }
        path = str(tmp_path / "transformer.safetensors")
        save_file(tensors, path)

        arch = detect_transformer_architecture(path)
        assert arch == "ZImageTransformer2DModel"

    def test_returns_none_for_non_transformer(self, tmp_path):
        """Safetensors without signature keys should return None."""
        tensors = {"decoder.weight": torch.randn(4, 4), "encoder.weight": torch.randn(4, 4)}
        path = str(tmp_path / "vae.safetensors")
        save_file(tensors, path)

        arch = detect_transformer_architecture(path)
        assert arch is None

    def test_returns_none_for_partial_signature(self, tmp_path):
        """Only one of the two required keys — should NOT match."""
        tensors = {
            "context_refiner.weight": torch.randn(4, 4),
            "other.weight": torch.randn(2, 2),
        }
        path = str(tmp_path / "partial.safetensors")
        save_file(tensors, path)

        arch = detect_transformer_architecture(path)
        assert arch is None

    def test_returns_none_for_nonexistent_file(self, tmp_path):
        arch = detect_transformer_architecture(str(tmp_path / "missing.safetensors"))
        assert arch is None

    def test_returns_none_for_invalid_file(self, tmp_path):
        path = str(tmp_path / "invalid.safetensors")
        with open(path, "w") as f:
            f.write("not a safetensors file")

        arch = detect_transformer_architecture(path)
        assert arch is None


class TestArchitectureSignatures:
    def test_signature_keys_are_frozensets(self):
        for keys, name in _ARCHITECTURE_SIGNATURES:
            assert isinstance(keys, frozenset)
            assert isinstance(name, str)
            assert len(keys) > 0
