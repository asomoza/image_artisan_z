import pytest

from iartisanz.utils.json_utils import extract_dict_from_json_graph


def test_extract_dict_from_json_graph_includes_lora_is_slider_and_enabled_from_lora_enabled():
    graph = {
        "nodes": [
            {
                "class": "LoraNode",
                "id": 3,
                "name": "my_lora",
                "enabled": True,
                "state": {
                    "adapter_name": "foo",
                    "lora_name": "foo",
                    "path": "/tmp/foo.safetensors",
                    "transformer_weight": 0.7,
                    "version": "v1",
                    "lora_enabled": False,
                    "is_slider": True,
                    "database_id": 42,
                    "granular_transformer_weights_enabled": True,
                    "transformer_granular_weights": {"layers.0": 0.5},
                },
            }
        ]
    }

    subset = extract_dict_from_json_graph(graph, ["loras"])

    assert "loras" in subset
    assert len(subset["loras"]) == 1

    lora = subset["loras"][0]
    assert lora["enabled"] is False
    assert lora["is_slider"] is True
    assert lora["transformer_weight"] == pytest.approx(0.7)
