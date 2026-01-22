from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def generation_module(monkeypatch, qapp):
    """Import GenerationModule with heavy ControlNet deps stubbed.

    The GenerationModule imports ControlNet UI + preprocess code, which pulls in
    optional dependencies. For this unit test we only need the source-image
    removal logic.
    """

    controlnet_dialog_mod = ModuleType("iartisanz.modules.generation.controlnet.controlnet_image_dialog")
    controlnet_dialog_mod.ControlNetImageDialog = object
    monkeypatch.setitem(
        sys.modules,
        "iartisanz.modules.generation.controlnet.controlnet_image_dialog",
        controlnet_dialog_mod,
    )

    # Force reload so the torch stub is used.
    sys.modules.pop("iartisanz.modules.generation.generation_module", None)
    mod = importlib.import_module("iartisanz.modules.generation.generation_module")
    return mod


class _FakeThread:
    def __init__(self):
        self.calls: list[str] = []

    def remove_source_image_mask(self):
        self.calls.append("remove_source_image_mask")

    def remove_source_image(self):
        self.calls.append("remove_source_image")


def test_remove_source_image_also_clears_mask_and_deletes_temp_file(generation_module, tmp_path: Path):
    GenerationModule = generation_module.GenerationModule

    # Create a bare instance without running __init__ (avoids heavy UI setup).
    gm = GenerationModule.__new__(GenerationModule)

    temp_dir = tmp_path / "temp"
    temp_dir.mkdir()
    mask_path = temp_dir / "mask.png"
    mask_path.write_bytes(b"mask")

    gm.generation_thread = _FakeThread()
    gm.directories = SimpleNamespace(temp_path=str(temp_dir))

    gm.source_image_mask_path = str(mask_path)
    gm.source_image_mask_thumb_path = "thumb"  # will be cleared

    # Also set these to confirm they're cleared without errors.
    gm.source_image_path = "some.png"
    gm.source_image_thumb_path = "some_thumb"
    gm.source_image_layers = [object()]

    gm.on_source_image_event({"action": "remove"})

    assert "remove_source_image_mask" in gm.generation_thread.calls
    assert "remove_source_image" in gm.generation_thread.calls

    assert gm.source_image_mask_path is None
    assert gm.source_image_mask_thumb_path is None
    assert gm.source_image_path is None
    assert gm.source_image_thumb_path is None
    assert gm.source_image_layers is None

    assert not mask_path.exists()
