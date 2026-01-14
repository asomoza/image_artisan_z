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


def test_zimage_model_node_reuses_active_model(monkeypatch):
    mm = get_model_manager()
    mm.clear()

    # Register a pre-loaded model in the global manager.
    mm.register_active_model(
        model_id="my-model",
        tokenizer=object(),
        text_encoder=object(),
        transformer=DummyTransformer(),
        vae=DummyVae(),
    )

    # If reuse is working, we should never call into from_pretrained.
    def _boom(*_a, **_kw):
        raise AssertionError("from_pretrained should not be called when reusing the active model")

    import iartisanz.modules.generation.graph.nodes.zimage_model_node as mod

    monkeypatch.setattr(mod.Qwen2Tokenizer, "from_pretrained", _boom)
    monkeypatch.setattr(mod.Qwen3Model, "from_pretrained", _boom)
    monkeypatch.setattr(mod.ZImageTransformer2DModel, "from_pretrained", _boom)
    monkeypatch.setattr(mod.AutoencoderKL, "from_pretrained", _boom)

    node = ZImageModelNode(path="/tmp/ignored", model_name="my-model", version="v", model_type="t")
    node.dtype = None

    out = node()

    assert out["num_channels_latents"] == 16
    assert out["vae_scale_factor"] == 8

    # Clean up global state for isolation.
    mm.clear()


def test_zimage_model_node_loads_when_model_differs(monkeypatch):
    mm = get_model_manager()
    mm.clear()

    # Different active model id should force load path.
    mm.register_active_model(
        model_id="other", tokenizer=object(), text_encoder=object(), transformer=DummyTransformer(), vae=DummyVae()
    )

    calls = {"n": 0}

    def _count(*_a, **_kw):
        calls["n"] += 1
        return object()

    class _Transformer:
        in_channels = 4

    class _Vae:
        config = types.SimpleNamespace(block_out_channels=[1, 2])

    def _transformer(*_a, **_kw):
        calls["n"] += 1
        return _Transformer()

    def _vae(*_a, **_kw):
        calls["n"] += 1
        return _Vae()

    import iartisanz.modules.generation.graph.nodes.zimage_model_node as mod

    monkeypatch.setattr(mod.Qwen2Tokenizer, "from_pretrained", _count)
    monkeypatch.setattr(mod.Qwen3Model, "from_pretrained", _count)
    monkeypatch.setattr(mod.ZImageTransformer2DModel, "from_pretrained", _transformer)
    monkeypatch.setattr(mod.AutoencoderKL, "from_pretrained", _vae)

    node = ZImageModelNode(path="/tmp/ignored", model_name="my-model", version="v", model_type="t")
    node.dtype = None

    out = node()

    assert calls["n"] == 4
    assert out["num_channels_latents"] == 4
    assert out["vae_scale_factor"] == 2

    mm.clear()
