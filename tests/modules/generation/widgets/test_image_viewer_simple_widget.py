from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def image_viewer_module(monkeypatch):
    """Import ImageViewerSimpleWidget with heavy UI deps stubbed."""

    # FullScreenPreview pulls in extra UI; stub it.
    full_screen_mod = ModuleType("iartisanz.modules.generation.dialogs.full_screen_preview")
    full_screen_mod.FullScreenPreview = object
    monkeypatch.setitem(sys.modules, "iartisanz.modules.generation.dialogs.full_screen_preview", full_screen_mod)

    # EventBus isn't needed for these pure helper-method tests.
    event_bus_mod = ModuleType("iartisanz.app.event_bus")

    class _EventBus:
        def __init__(self):
            pass

    event_bus_mod.EventBus = _EventBus
    monkeypatch.setitem(sys.modules, "iartisanz.app.event_bus", event_bus_mod)

    sys.modules.pop("iartisanz.modules.generation.widgets.image_viewer_simple_widget", None)
    return importlib.import_module("iartisanz.modules.generation.widgets.image_viewer_simple_widget")


def _node_by_name(nodes: list[dict], name: str) -> dict:
    for node in nodes:
        if node.get("name") == name:
            return node
    raise AssertionError(f"Node '{name}' not found")


def test_copy_source_image_and_mask_rewrites_paths(image_viewer_module, tmp_path: Path):
    ImageViewerSimpleWidget = image_viewer_module.ImageViewerSimpleWidget

    outputs_source_images = tmp_path / "outputs" / "source_images"
    outputs_source_masks = tmp_path / "outputs" / "source_masks"
    outputs_source_images.mkdir(parents=True)
    outputs_source_masks.mkdir(parents=True)

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_image = src_dir / "init.png"
    src_mask = src_dir / "mask_final.png"
    src_image.write_bytes(b"img")
    src_mask.write_bytes(b"mask")

    json_graph = json.dumps(
        {
            "nodes": [
                {"class": "ImageLoadNode", "name": "source_image", "state": {"path": str(src_image)}},
                {"class": "ImageLoadNode", "name": "source_image_mask", "state": {"path": str(src_mask)}},
            ]
        }
    )

    viewer = ImageViewerSimpleWidget.__new__(ImageViewerSimpleWidget)
    viewer.directories = SimpleNamespace(
        outputs_source_images=str(outputs_source_images),
        outputs_source_masks=str(outputs_source_masks),
    )

    ts = "20990101010101"
    updated = viewer._copy_source_image_and_rewrite_graph(json_graph, ts)
    updated = viewer._copy_source_mask_and_rewrite_graph(updated, ts)

    payload = json.loads(updated)
    nodes = payload["nodes"]

    new_src_image_path = Path(_node_by_name(nodes, "source_image")["state"]["path"])
    new_src_mask_path = Path(_node_by_name(nodes, "source_image_mask")["state"]["path"])

    assert new_src_image_path.exists()
    assert new_src_mask_path.exists()

    assert str(new_src_image_path).startswith(str(outputs_source_images))
    assert str(new_src_mask_path).startswith(str(outputs_source_masks))

    assert new_src_image_path.name.startswith(f"{ts}_source_image")
    assert new_src_mask_path.name.startswith(f"{ts}_source_mask")

    # Original files are not deleted by the copy.
    assert src_image.exists()
    assert src_mask.exists()


def test_copy_source_mask_noop_when_missing_node(image_viewer_module, tmp_path: Path):
    ImageViewerSimpleWidget = image_viewer_module.ImageViewerSimpleWidget

    outputs_source_masks = tmp_path / "outputs" / "source_masks"
    outputs_source_masks.mkdir(parents=True)

    json_graph = json.dumps({"nodes": [{"class": "ImageLoadNode", "name": "other", "state": {"path": "x"}}]})

    viewer = ImageViewerSimpleWidget.__new__(ImageViewerSimpleWidget)
    viewer.directories = SimpleNamespace(outputs_source_masks=str(outputs_source_masks), outputs_source_images="")

    ts = "20990101010101"
    assert viewer._copy_source_mask_and_rewrite_graph(json_graph, ts) == json_graph
