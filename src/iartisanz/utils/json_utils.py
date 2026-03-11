from __future__ import annotations

import json
from typing import Any, Iterable

from iartisanz.modules.generation.data_objects.model_data_object import ModelDataObject
from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject


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
                    "adapter_name": state.get("adapter_name", None),
                    "lora_name": state.get("lora_name", None),
                    "path": state.get("path", None),
                    "transformer_weight": state.get("transformer_weight", 1.0),
                    "version": state.get("version", None),
                    "enabled": state.get("lora_enabled", state.get("enabled", node.get("enabled"))),
                    "is_slider": state.get("is_slider", False),
                    "database_id": state.get("database_id", 0),
                    "granular_transformer_weights_enabled": state.get("granular_transformer_weights_enabled", False),
                    "granular_transformer_weights": state.get("transformer_granular_weights", {}),
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
        if "path" in node_state:
            return node_state.get("path")
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

        # Special case: "model" is stored at the top-level of the node dict
        if node_name == "model":
            node = by_name.get(node_name)
            if not node:
                if include_missing:
                    out[out_key] = default
                continue

            model_name = node.get("model_name")
            path = node.get("path")
            version = node.get("version")
            model_type = node.get("model_type")
            model_id = node.get("db_model_id", 0)

            if model_name is None and path is None and version is None and model_type is None:
                if include_missing:
                    out[out_key] = default
                continue

            out[out_key] = {
                "name": model_name or "",
                "version": version or "",
                "filepath": path or "",
                "model_type": model_type or 0,
                "id": model_id or 0,
            }
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


def cast_scheduler(value) -> SchedulerDataObject:
    if isinstance(value, SchedulerDataObject):
        return value

    if value is None:
        return SchedulerDataObject()

    if isinstance(value, dict):
        try:
            return SchedulerDataObject.from_dict(value)
        except Exception:
            return SchedulerDataObject()

    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return SchedulerDataObject.from_dict(data)
        except Exception:
            pass
        return SchedulerDataObject()

    return SchedulerDataObject()


def persist_image_paths_in_graph(
    json_graph: str,
    directories: Any,
    timestamp: str,
) -> str:
    """Copy ImageLoadNode files to permanent output directories and rewrite paths.

    This ensures that graph JSON stored in examples or PNG metadata references
    files in stable output directories rather than temporary paths that get
    cleaned up between sessions.

    Uses content hashing to deduplicate: if an identical file was already saved
    for the same kind (e.g. "source_image"), the existing copy is reused.

    Handles: source_image, source_image_mask, edit_image_0-3, edit_image_mask.
    """
    import hashlib
    import os
    import shutil
    from pathlib import Path

    from iartisanz.utils.database import Database

    try:
        data = json.loads(json_graph)
    except Exception:
        return json_graph

    nodes = data.get("nodes")
    if not isinstance(nodes, list):
        return json_graph

    # Map node names to (dest_dir, filename_prefix) for each ImageLoadNode we care about.
    dest_map: dict[str, tuple[str, str]] = {
        "source_image": (directories.outputs_source_images, "source_image"),
        "source_image_mask": (directories.outputs_source_masks, "source_mask"),
        "edit_image_mask": (directories.outputs_edit_masks, "edit_image_mask"),
    }
    for i in range(4):
        dest_map[f"edit_image_{i}"] = (directories.outputs_edit_source_images, f"edit_image_{i}")

    db = Database(os.path.join(directories.data_path, "app.db"))
    updated = False

    for node in nodes:
        if not isinstance(node, dict) or node.get("class") != "ImageLoadNode":
            continue

        node_name = node.get("name")
        if node_name not in dest_map:
            continue

        state = node.get("state")
        if not isinstance(state, dict):
            continue

        src_path = state.get("path")
        if not isinstance(src_path, str) or not src_path.strip():
            continue

        src = Path(src_path)
        if not src.exists() or not src.is_file():
            continue

        dest_dir_path, prefix = dest_map[node_name]
        dest_dir = Path(dest_dir_path)

        # Skip if already in the target directory.
        try:
            if src.parent.resolve() == dest_dir.resolve():
                continue
        except Exception:
            pass

        content_hash = hashlib.md5(src.read_bytes()).hexdigest()

        # Check for existing identical file.
        existing = _find_existing_source(db, node_name, content_hash)
        if existing is not None:
            state["path"] = existing
            updated = True
            continue

        ext = src.suffix if src.suffix else ".png"
        dest = dest_dir / f"{timestamp}_{prefix}{ext}"

        if dest.exists():
            for seq in range(1, 10_000):
                candidate = dest_dir / f"{timestamp}_{prefix}_{seq}{ext}"
                if not candidate.exists():
                    dest = candidate
                    break

        try:
            shutil.copy2(src, dest)
        except Exception:
            continue

        dest_str = str(dest)
        _record_source(db, node_name, content_hash, dest_str)
        state["path"] = dest_str
        updated = True

    if not updated:
        return json_graph

    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return json_graph


def _find_existing_source(db: Any, kind: str, content_hash: str) -> str | None:
    """Look up an existing source file by kind and content hash.

    Returns the filepath if found and the file still exists on disk.
    Removes stale DB entries if the file has been deleted.
    """
    row = db.fetch_one(
        "SELECT id, filepath FROM source_file WHERE kind = ? AND content_hash = ?",
        (kind, content_hash),
    )
    if row is None:
        return None

    row_id, filepath = row
    import os

    if os.path.isfile(filepath):
        return filepath

    # Stale entry — file was deleted from disk.
    db.execute("DELETE FROM source_file WHERE id = ?", (row_id,))
    return None


def _record_source(db: Any, kind: str, content_hash: str, filepath: str) -> None:
    """Record a source file in the database, upserting on conflict."""
    db.execute(
        "INSERT INTO source_file (kind, content_hash, filepath) VALUES (?, ?, ?) "
        "ON CONFLICT(kind, content_hash) DO UPDATE SET filepath = excluded.filepath",
        (kind, content_hash, filepath),
    )


def cast_model(value: Any) -> ModelDataObject | None:
    if isinstance(value, ModelDataObject):
        return value

    if value is None:
        return ModelDataObject()

    if isinstance(value, dict):
        try:
            return ModelDataObject.from_dict(value)
        except Exception:
            return ModelDataObject()

    if isinstance(value, str):
        try:
            data = json.loads(value)
            if isinstance(data, dict):
                return ModelDataObject.from_dict(data)
        except Exception:
            pass
        return ModelDataObject()

    return ModelDataObject()
