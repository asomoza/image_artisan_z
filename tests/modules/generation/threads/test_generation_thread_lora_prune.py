import json

import torch

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread


class FakeGraph:
    def set_abort_function(self, _fn):
        pass

    def to_json(self):
        return "{}"


class FakeTransformer:
    def __init__(self, adapters):
        # mimic diffusers/peft structure: a dict-like peft_config mapping adapter_name -> config
        self.peft_config = {name: object() for name in adapters}
        self.deleted: list[list[str]] = []

    def delete_adapters(self, names):
        self.deleted.append(list(names))
        for n in names:
            self.peft_config.pop(n, None)


def _json_with_loras(loras):
    # loras: list of dict(adapter_name, path, enabled=True, lora_enabled=True)
    nodes = []
    for i, spec in enumerate(loras):
        nodes.append(
            {
                "class": "LoraNode",
                "id": i,
                "name": f"lora_{i}",
                "enabled": spec.get("enabled", True),
                "state": {
                    "adapter_name": spec.get("adapter_name"),
                    "path": spec.get("path"),
                    "lora_enabled": spec.get("lora_enabled", True),
                },
            }
        )
    return json.dumps({"format_version": 1, "nodes": nodes, "connections": [], "additional_generation_data": {}})


def test_prune_removes_unused_adapters():
    mm = get_model_manager()
    mm.clear()

    transformer = FakeTransformer(adapters=["a", "b", "c"])
    mm.register_active_model(
        model_id="m", transformer=transformer, tokenizer=object(), text_encoder=object(), vae=object()
    )
    mm.set_lora_source("a", "/a.safetensors")
    mm.set_lora_source("b", "/b.safetensors")
    mm.set_lora_source("c", "/c.safetensors")

    thread = NodeGraphThread(None, FakeGraph(), torch.float32, torch.device("cpu"))

    required_json = _json_with_loras(
        [
            {"adapter_name": "a", "path": "/a.safetensors"},
            # b and c not present => should be pruned
        ]
    )

    required = thread._extract_required_loras(required_json)
    thread._prune_transformer_loras(required)

    assert "a" in transformer.peft_config
    assert "b" not in transformer.peft_config
    assert "c" not in transformer.peft_config

    mm.clear()


def test_prune_forces_reload_when_source_changes():
    mm = get_model_manager()
    mm.clear()

    transformer = FakeTransformer(adapters=["a"])
    mm.register_active_model(
        model_id="m", transformer=transformer, tokenizer=object(), text_encoder=object(), vae=object()
    )

    # Previously loaded from /old
    mm.set_lora_source("a", "/old.safetensors")

    thread = NodeGraphThread(None, FakeGraph(), torch.float32, torch.device("cpu"))

    required_json = _json_with_loras(
        [
            {"adapter_name": "a", "path": "/new.safetensors"},
        ]
    )

    required = thread._extract_required_loras(required_json)
    thread._prune_transformer_loras(required)

    # Adapter should have been deleted so the node can reload it.
    assert "a" not in transformer.peft_config
    assert mm.get_lora_source("a") is None

    mm.clear()
