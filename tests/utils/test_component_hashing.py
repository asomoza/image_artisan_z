"""Tests for component hashing functions in model_utils."""
from __future__ import annotations

import os

import pytest
import torch
from safetensors.torch import save_file

from iartisanz.utils.model_utils import (
    _hash_directory_contents,
    _hash_safetensors_dir,
    _normalize_tensor_key,
    _tensor_to_bytes,
    calculate_component_hash,
    calculate_structural_hash,
)


# ---------------------------------------------------------------------------
# _normalize_tensor_key
# ---------------------------------------------------------------------------


class TestNormalizeTensorKey:
    def test_strips_model_prefix(self):
        assert _normalize_tensor_key("model.layers.0.weight") == "layers.0.weight"

    def test_strips_encoder_model_prefix(self):
        assert _normalize_tensor_key("encoder.model.layers.0.bias") == "layers.0.bias"

    def test_strips_text_model_prefix(self):
        assert _normalize_tensor_key("text_model.embeddings.weight") == "embeddings.weight"

    def test_no_prefix_unchanged(self):
        assert _normalize_tensor_key("layers.0.weight") == "layers.0.weight"

    def test_strips_first_matching_prefix_only(self):
        # "model.model.foo" should strip "model." once -> "model.foo"
        assert _normalize_tensor_key("model.model.foo") == "model.foo"


# ---------------------------------------------------------------------------
# _tensor_to_bytes
# ---------------------------------------------------------------------------


class TestTensorToBytes:
    def test_float32_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        raw = _tensor_to_bytes(t)
        assert isinstance(raw, bytes)
        assert len(raw) == 3 * 4  # 3 elements * 4 bytes each

    def test_bfloat16_tensor(self):
        t = torch.tensor([1.0, 2.0, 3.0], dtype=torch.bfloat16)
        raw = _tensor_to_bytes(t)
        assert isinstance(raw, bytes)
        assert len(raw) == 3 * 2  # 3 elements * 2 bytes each

    def test_non_contiguous_tensor(self):
        t = torch.randn(4, 4, dtype=torch.float32)
        sliced = t[:, ::2]  # non-contiguous
        assert not sliced.is_contiguous()
        raw = _tensor_to_bytes(sliced)
        assert isinstance(raw, bytes)

    def test_same_data_same_bytes(self):
        t1 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        t2 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        assert _tensor_to_bytes(t1) == _tensor_to_bytes(t2)

    def test_different_data_different_bytes(self):
        t1 = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
        t2 = torch.tensor([3.0, 4.0], dtype=torch.bfloat16)
        assert _tensor_to_bytes(t1) != _tensor_to_bytes(t2)


# ---------------------------------------------------------------------------
# Helpers to create synthetic safetensors files
# ---------------------------------------------------------------------------


def _create_safetensors_component(
    comp_dir: str,
    tensors: dict[str, torch.Tensor],
    config: dict | None = None,
) -> None:
    """Create a synthetic component directory with safetensors + config.json."""
    os.makedirs(comp_dir, exist_ok=True)
    save_file(tensors, os.path.join(comp_dir, "model.safetensors"))
    if config is not None:
        import json

        with open(os.path.join(comp_dir, "config.json"), "w") as f:
            json.dump(config, f)


def _create_text_files(comp_dir: str, files: dict[str, str]) -> None:
    """Create a directory with plain text files (like tokenizer)."""
    os.makedirs(comp_dir, exist_ok=True)
    for name, content in files.items():
        with open(os.path.join(comp_dir, name), "w") as f:
            f.write(content)


# ---------------------------------------------------------------------------
# _hash_safetensors_dir
# ---------------------------------------------------------------------------


class TestHashSafetensorsDir:
    def test_identical_tensors_same_hash(self, tmp_path):
        tensors = {"layer.weight": torch.randn(4, 4)}
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        _create_safetensors_component(dir_a, tensors)
        _create_safetensors_component(dir_b, tensors)

        assert _hash_safetensors_dir(dir_a) == _hash_safetensors_dir(dir_b)

    def test_different_tensors_different_hash(self, tmp_path):
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        _create_safetensors_component(dir_a, {"w": torch.randn(4, 4)})
        _create_safetensors_component(dir_b, {"w": torch.randn(4, 4)})

        assert _hash_safetensors_dir(dir_a) != _hash_safetensors_dir(dir_b)

    def test_key_normalization_same_hash(self, tmp_path):
        """model.layers.0.weight and layers.0.weight should produce the same hash."""
        t = torch.randn(8, 8)
        dir_a = str(tmp_path / "with_prefix")
        dir_b = str(tmp_path / "without_prefix")
        _create_safetensors_component(dir_a, {"model.layers.0.weight": t})
        _create_safetensors_component(dir_b, {"layers.0.weight": t})

        assert _hash_safetensors_dir(dir_a) == _hash_safetensors_dir(dir_b)

    def test_bfloat16_tensors(self, tmp_path):
        t = torch.randn(4, 4, dtype=torch.bfloat16)
        comp_dir = str(tmp_path / "bf16")
        _create_safetensors_component(comp_dir, {"w": t})
        h = _hash_safetensors_dir(comp_dir)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_multiple_shards(self, tmp_path):
        comp_dir = str(tmp_path / "multi")
        os.makedirs(comp_dir)
        save_file({"a.weight": torch.randn(2, 2)}, os.path.join(comp_dir, "model-00001.safetensors"))
        save_file({"b.weight": torch.randn(2, 2)}, os.path.join(comp_dir, "model-00002.safetensors"))

        h = _hash_safetensors_dir(comp_dir)
        assert isinstance(h, str)

    def test_no_safetensors_falls_back(self, tmp_path):
        """If no .safetensors files, falls back to directory content hash."""
        comp_dir = str(tmp_path / "plain")
        _create_text_files(comp_dir, {"vocab.json": '{"hello": 0}'})

        h = _hash_safetensors_dir(comp_dir)
        assert isinstance(h, str)


# ---------------------------------------------------------------------------
# _hash_directory_contents
# ---------------------------------------------------------------------------


class TestHashDirectoryContents:
    def test_same_files_same_hash(self, tmp_path):
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        files = {"vocab.json": '{"a": 0}', "merges.txt": "a b"}
        _create_text_files(dir_a, files)
        _create_text_files(dir_b, files)

        assert _hash_directory_contents(dir_a) == _hash_directory_contents(dir_b)

    def test_different_content_different_hash(self, tmp_path):
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        _create_text_files(dir_a, {"vocab.json": '{"a": 0}'})
        _create_text_files(dir_b, {"vocab.json": '{"b": 1}'})

        assert _hash_directory_contents(dir_a) != _hash_directory_contents(dir_b)

    def test_ignores_subdirectories(self, tmp_path):
        comp_dir = str(tmp_path / "tok")
        _create_text_files(comp_dir, {"vocab.json": '{"a": 0}'})
        os.makedirs(os.path.join(comp_dir, "subdir"))
        with open(os.path.join(comp_dir, "subdir", "extra.txt"), "w") as f:
            f.write("extra")

        # Should only hash top-level files
        h = _hash_directory_contents(comp_dir)
        assert isinstance(h, str)


# ---------------------------------------------------------------------------
# calculate_component_hash
# ---------------------------------------------------------------------------


class TestCalculateComponentHash:
    def test_safetensors_dir(self, tmp_path):
        comp_dir = str(tmp_path / "comp")
        _create_safetensors_component(comp_dir, {"w": torch.randn(4, 4)})

        h = calculate_component_hash(comp_dir)
        assert isinstance(h, str)
        assert len(h) > 0

    def test_non_safetensors_dir(self, tmp_path):
        comp_dir = str(tmp_path / "tok")
        _create_text_files(comp_dir, {"vocab.json": '{"a": 0}'})

        h = calculate_component_hash(comp_dir)
        assert isinstance(h, str)

    def test_nonexistent_dir_raises(self, tmp_path):
        with pytest.raises(ValueError, match="does not exist"):
            calculate_component_hash(str(tmp_path / "nonexistent"))

    def test_deterministic(self, tmp_path):
        comp_dir = str(tmp_path / "comp")
        _create_safetensors_component(comp_dir, {"w": torch.tensor([1.0, 2.0])})

        h1 = calculate_component_hash(comp_dir)
        h2 = calculate_component_hash(comp_dir)
        assert h1 == h2

    def test_normalized_keys_same_hash(self, tmp_path):
        """Validates the core Z-Image / Z-Image-Turbo text encoder dedup scenario."""
        t1 = torch.randn(16, 16, dtype=torch.bfloat16)
        t2 = torch.randn(8, 8, dtype=torch.bfloat16)
        dir_a = str(tmp_path / "qwen3_for_causal")
        dir_b = str(tmp_path / "qwen3_model")
        _create_safetensors_component(dir_a, {"model.embed_tokens.weight": t1, "model.layers.0.weight": t2})
        _create_safetensors_component(dir_b, {"embed_tokens.weight": t1, "layers.0.weight": t2})

        assert calculate_component_hash(dir_a) == calculate_component_hash(dir_b)


# ---------------------------------------------------------------------------
# calculate_structural_hash
# ---------------------------------------------------------------------------


class TestCalculateStructuralHash:
    def test_same_structure_same_hash(self, tmp_path):
        """Same key names, shapes, dtypes → same structural hash (even if data differs)."""
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        _create_safetensors_component(dir_a, {"w": torch.randn(4, 4)})
        _create_safetensors_component(dir_b, {"w": torch.randn(4, 4)})

        # Structural hash ignores data — but shapes/dtypes/keys are the same
        h_a = calculate_structural_hash(dir_a)
        h_b = calculate_structural_hash(dir_b)
        assert h_a == h_b

    def test_different_shape_different_hash(self, tmp_path):
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        _create_safetensors_component(dir_a, {"w": torch.randn(4, 4)})
        _create_safetensors_component(dir_b, {"w": torch.randn(8, 8)})

        assert calculate_structural_hash(dir_a) != calculate_structural_hash(dir_b)

    def test_normalized_keys(self, tmp_path):
        """model.w and w should produce the same structural hash."""
        t = torch.randn(4, 4)
        dir_a = str(tmp_path / "a")
        dir_b = str(tmp_path / "b")
        _create_safetensors_component(dir_a, {"model.w": t})
        _create_safetensors_component(dir_b, {"w": t})

        assert calculate_structural_hash(dir_a) == calculate_structural_hash(dir_b)

    def test_non_safetensors_falls_back_to_content_hash(self, tmp_path):
        comp_dir = str(tmp_path / "tok")
        _create_text_files(comp_dir, {"vocab.json": '{"a": 0}'})

        h = calculate_structural_hash(comp_dir)
        assert isinstance(h, str)
