"""Tests for ZImageModelNode smart model switching via component hashes."""
from __future__ import annotations

import types

import pytest

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.zimage_model_node import ZImageModelNode


class DummyTransformer:
    def __init__(self):
        self.in_channels = 16


class DummyVae:
    def __init__(self):
        self.config = types.SimpleNamespace(block_out_channels=[1, 2, 3, 4])


def _make_loaders(monkeypatch, calls: dict):
    """Patch from_pretrained methods to count calls."""
    import iartisanz.modules.generation.graph.nodes.zimage_model_node as mod

    def _tok(*a, **kw):
        calls["tokenizer"] = calls.get("tokenizer", 0) + 1
        return object()

    def _te(*a, **kw):
        calls["text_encoder"] = calls.get("text_encoder", 0) + 1
        return object()

    def _tx(*a, **kw):
        calls["transformer"] = calls.get("transformer", 0) + 1
        return DummyTransformer()

    def _vae(*a, **kw):
        calls["vae"] = calls.get("vae", 0) + 1
        return DummyVae()

    monkeypatch.setattr(mod.AutoTokenizer, "from_pretrained", _tok)
    monkeypatch.setattr(mod.Qwen2Tokenizer, "from_pretrained", _tok)
    monkeypatch.setattr(mod.Qwen3Model, "from_pretrained", _te)
    monkeypatch.setattr(mod.ZImageTransformer2DModel, "from_pretrained", _tx)
    monkeypatch.setattr(mod.AutoencoderKL, "from_pretrained", _vae)


def _make_node(
    model_name="test-model",
    db_model_id=None,
    target_hashes=None,
    registry_paths=None,
):
    """Create a ZImageModelNode with optional registry override."""
    node = ZImageModelNode(
        path="/tmp/ignored",
        model_name=model_name,
        version="v1",
        model_type="1",
        db_model_id=db_model_id,
    )
    node.dtype = None

    # Override registry methods so we don't need a real database
    if target_hashes is not None:
        node._get_target_component_hashes = lambda: target_hashes
    if registry_paths is not None:
        node._resolve_component_paths = lambda: registry_paths

    return node


class TestSmartLoad:
    def test_all_hashes_match_skips_all_loads(self, monkeypatch):
        """When all component hashes match, no from_pretrained should be called."""
        mm = get_model_manager()
        mm.clear()

        # Pre-load all components
        mm.register_component("tokenizer", object())
        mm.register_component("text_encoder", object())
        mm.register_component("transformer", DummyTransformer())
        mm.register_component("vae", DummyVae())
        mm.set_component_hash("tokenizer", "hash_tok")
        mm.set_component_hash("text_encoder", "hash_te")
        mm.set_component_hash("transformer", "hash_tx")
        mm.set_component_hash("vae", "hash_vae")

        calls = {}
        _make_loaders(monkeypatch, calls)

        target_hashes = {
            "tokenizer": "hash_tok",
            "text_encoder": "hash_te",
            "transformer": "hash_tx",
            "vae": "hash_vae",
        }
        node = _make_node(target_hashes=target_hashes, registry_paths={})
        out = node()

        assert calls == {}  # Nothing loaded
        assert out["num_channels_latents"] == 16
        assert out["vae_scale_factor"] == 8

        mm.clear()

    def test_only_transformer_changed_reloads_transformer_only(self, monkeypatch):
        """When only the transformer hash differs, only transformer should reload."""
        mm = get_model_manager()
        mm.clear()

        mm.register_component("tokenizer", object())
        mm.register_component("text_encoder", object())
        mm.register_component("transformer", DummyTransformer())
        mm.register_component("vae", DummyVae())
        mm.set_component_hash("tokenizer", "hash_tok")
        mm.set_component_hash("text_encoder", "hash_te")
        mm.set_component_hash("transformer", "hash_tx_OLD")
        mm.set_component_hash("vae", "hash_vae")

        calls = {}
        _make_loaders(monkeypatch, calls)

        target_hashes = {
            "tokenizer": "hash_tok",
            "text_encoder": "hash_te",
            "transformer": "hash_tx_NEW",
            "vae": "hash_vae",
        }
        node = _make_node(target_hashes=target_hashes, registry_paths={})
        out = node()

        assert calls.get("tokenizer", 0) == 0
        assert calls.get("text_encoder", 0) == 0
        assert calls["transformer"] == 1
        assert calls.get("vae", 0) == 0
        assert out["num_channels_latents"] == 16

        mm.clear()

    def test_transformer_change_clears_lora_and_compiled(self, monkeypatch):
        """When transformer changes, LoRA sources and compiled cache should be cleared."""
        mm = get_model_manager()
        mm.clear()

        mm.register_component("tokenizer", object())
        mm.register_component("text_encoder", object())
        mm.register_component("transformer", DummyTransformer())
        mm.register_component("vae", DummyVae())
        mm.set_component_hash("tokenizer", "hash_tok")
        mm.set_component_hash("text_encoder", "hash_te")
        mm.set_component_hash("transformer", "hash_tx_OLD")
        mm.set_component_hash("vae", "hash_vae")

        # Simulate existing LoRA and compiled state
        mm._lora_sources["lora1"] = "/path/to/lora"
        mm._compiled_components[("model", "transformer", "cuda:0", "float16", "")] = object()

        calls = {}
        _make_loaders(monkeypatch, calls)

        target_hashes = {
            "tokenizer": "hash_tok",
            "text_encoder": "hash_te",
            "transformer": "hash_tx_NEW",
            "vae": "hash_vae",
        }
        node = _make_node(target_hashes=target_hashes, registry_paths={})
        node()

        assert mm._lora_sources == {}
        assert mm._compiled_components == {}

        mm.clear()

    def test_tokenizer_and_transformer_changed(self, monkeypatch):
        """Multiple components changed — only those should reload."""
        mm = get_model_manager()
        mm.clear()

        mm.register_component("tokenizer", object())
        mm.register_component("text_encoder", object())
        mm.register_component("transformer", DummyTransformer())
        mm.register_component("vae", DummyVae())
        mm.set_component_hash("tokenizer", "hash_tok_OLD")
        mm.set_component_hash("text_encoder", "hash_te")
        mm.set_component_hash("transformer", "hash_tx_OLD")
        mm.set_component_hash("vae", "hash_vae")

        calls = {}
        _make_loaders(monkeypatch, calls)

        target_hashes = {
            "tokenizer": "hash_tok_NEW",
            "text_encoder": "hash_te",
            "transformer": "hash_tx_NEW",
            "vae": "hash_vae",
        }
        node = _make_node(target_hashes=target_hashes, registry_paths={})
        out = node()

        assert calls["tokenizer"] == 1
        assert calls.get("text_encoder", 0) == 0
        assert calls["transformer"] == 1
        assert calls.get("vae", 0) == 0

        mm.clear()

    def test_updates_component_hashes_after_load(self, monkeypatch):
        """After loading, the new hashes should be set on the model manager."""
        mm = get_model_manager()
        mm.clear()

        mm.register_component("tokenizer", object())
        mm.register_component("text_encoder", object())
        mm.register_component("transformer", DummyTransformer())
        mm.register_component("vae", DummyVae())
        mm.set_component_hash("tokenizer", "old_tok")
        mm.set_component_hash("text_encoder", "old_te")
        mm.set_component_hash("transformer", "old_tx")
        mm.set_component_hash("vae", "old_vae")

        calls = {}
        _make_loaders(monkeypatch, calls)

        target_hashes = {
            "tokenizer": "old_tok",
            "text_encoder": "old_te",
            "transformer": "new_tx",
            "vae": "old_vae",
        }
        node = _make_node(target_hashes=target_hashes, registry_paths={})
        node()

        # Unchanged components keep their hash
        assert mm.get_component_hash("tokenizer") == "old_tok"
        assert mm.get_component_hash("text_encoder") == "old_te"
        assert mm.get_component_hash("vae") == "old_vae"
        # Changed component gets new hash
        assert mm.get_component_hash("transformer") == "new_tx"

        mm.clear()


class TestLegacyLoad:
    def test_falls_back_to_legacy_without_hashes(self, monkeypatch):
        """When no target_hashes available, falls back to legacy full load."""
        mm = get_model_manager()
        mm.clear()

        calls = {}
        _make_loaders(monkeypatch, calls)

        node = _make_node(target_hashes=None, registry_paths=None)
        # Override _get_target_component_hashes to return None (no registry)
        node._get_target_component_hashes = lambda: None

        out = node()

        # All 4 components should be loaded
        assert calls["tokenizer"] == 1
        assert calls["text_encoder"] == 1
        assert calls["transformer"] == 1
        assert calls["vae"] == 1

        mm.clear()

    def test_legacy_reuses_active_model(self, monkeypatch):
        """Legacy path reuses components when model_id matches."""
        mm = get_model_manager()
        mm.clear()

        mm.register_active_model(
            model_id="my-model",
            tokenizer=object(),
            text_encoder=object(),
            transformer=DummyTransformer(),
            vae=DummyVae(),
        )

        calls = {}
        _make_loaders(monkeypatch, calls)

        node = _make_node(model_name="my-model", target_hashes=None)
        node._get_target_component_hashes = lambda: None

        out = node()

        assert calls == {}  # Nothing loaded
        assert out["num_channels_latents"] == 16

        mm.clear()


class TestNodeSerialization:
    def test_db_model_id_in_to_dict(self):
        node = ZImageModelNode(
            path="/p", model_name="m", version="v", model_type="t", db_model_id=42
        )
        d = node.to_dict()
        assert d["db_model_id"] == 42

    def test_db_model_id_missing_from_dict_is_none(self):
        node = ZImageModelNode(path="/p", model_name="m", version="v", model_type="t")
        d = node.to_dict()
        # Remove db_model_id if present
        d.pop("db_model_id", None)

        restored = ZImageModelNode.from_dict(d)
        assert restored.db_model_id is None

    def test_db_model_id_roundtrip(self):
        node = ZImageModelNode(
            path="/p", model_name="m", version="v", model_type="t", db_model_id=99
        )
        d = node.to_dict()
        restored = ZImageModelNode.from_dict(d)
        assert restored.db_model_id == 99
