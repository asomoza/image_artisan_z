from __future__ import annotations

import json
from typing import Any, Iterable


def extract_dict_from_json_graph(json_graph: Any, wanted: Iterable[Any], *, include_missing: bool = False) -> dict:
    data = _coerce_to_dict(json_graph)
    if not data:
        return {}

    nodes = (data.get("nodes") or []) if isinstance(data, dict) else []

    by_name: dict[str, dict] = {}
    loras: list[dict[str, Any]] = []

    for node in nodes:
        if not isinstance(node, dict):
            continue

        name = node.get("name") or (node.get("state") or {}).get("name")
        if name:
            by_name[name] = node

        if node.get("class") == "LoraNode":
            state = node.get("state") or {}
            loras.append(
                {
                    "id": state.get("id", node.get("id")),
                    "name": state.get("name", node.get("name")),
                    "adapter_name": state.get("adapter_name"),
                    "lora_name": state.get("lora_name"),
                    "path": state.get("path"),
                    "transformer_weight": state.get("transformer_weight"),
                    "version": state.get("version"),
                    "enabled": state.get("enabled", node.get("enabled")),
                    "database_id": state.get("database_id", node.get("database_id")),
                }
            )

    def auto_value(node_state: dict) -> Any:
        if "text" in node_state:
            return node_state.get("text")
        if "number" in node_state:
            return node_state.get("number")
        if "model_name" in node_state:
            return node_state.get("model_name")
        if "scheduler_data_object" in node_state:
            return node_state.get("scheduler_data_object")
        if "value" in node_state:
            return node_state.get("value")
        return None

    out: dict[str, Any] = {}
    for spec in wanted or []:
        out_key: str | None
        node_name: str | None
        state_key: str | None
        default: Any

        if isinstance(spec, str):
            out_key = spec
            node_name = spec
            state_key = None
            default = None
        elif isinstance(spec, dict):
            out_key = spec.get("out") or spec.get("name")
            node_name = spec.get("name")
            state_key = spec.get("key")
            default = spec.get("default")
        else:
            continue

        if not out_key or not node_name:
            continue

        # Special case: "loras" returns a list
        if node_name == "loras":
            value = loras
            if (value is None or value == []) and include_missing:
                value = default if default is not None else []
            if value is None and not include_missing:
                continue
            out[out_key] = value
            continue

        node = by_name.get(node_name)
        if not node:
            if include_missing:
                out[out_key] = default
            continue

        state = node.get("state") or {}
        value = state.get(state_key) if state_key else auto_value(state)
        if value is None:
            value = default

        if value is None and not include_missing:
            continue

        out[out_key] = value

    return out


def _coerce_to_dict(json_graph: Any) -> dict:
    if isinstance(json_graph, dict):
        return json_graph
    if isinstance(json_graph, str):
        try:
            parsed = json.loads(json_graph)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def cast_number_range(value) -> list[float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError("the number range must be a list of 2 numbers")

    a, b = value[0], value[1]
    if isinstance(a, bool) or isinstance(b, bool):
        raise ValueError("the number range values must be int|float (not bool)")

    return [float(a), float(b)]
