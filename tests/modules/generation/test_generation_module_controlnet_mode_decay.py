from __future__ import annotations

import importlib
import sys
from types import ModuleType

import pytest


@pytest.fixture
def generation_module(monkeypatch, qapp):
    """Import GenerationModule with heavy ControlNet deps stubbed."""

    controlnet_dialog_mod = ModuleType("iartisanz.modules.generation.controlnet.controlnet_image_dialog")
    controlnet_dialog_mod.ControlNetImageDialog = object
    monkeypatch.setitem(
        sys.modules,
        "iartisanz.modules.generation.controlnet.controlnet_image_dialog",
        controlnet_dialog_mod,
    )

    sys.modules.pop("iartisanz.modules.generation.generation_module", None)
    mod = importlib.import_module("iartisanz.modules.generation.generation_module")
    return mod


class _FakeThread:
    def __init__(self):
        self.calls: list[tuple[str, tuple, dict]] = []

    def add_controlnet(self, *args, **kwargs):
        self.calls.append(("add_controlnet", args, kwargs))

    def update_controlnet(self, *args, **kwargs):
        self.calls.append(("update_controlnet", args, kwargs))

    def update_controlnet_conditioning_scale(self, *args, **kwargs):
        self.calls.append(("update_controlnet_conditioning_scale", args, kwargs))

    def update_node(self, *args, **kwargs):
        self.calls.append(("update_node", args, kwargs))

    def enable_controlnet(self, *args, **kwargs):
        self.calls.append(("enable_controlnet", args, kwargs))

    def remove_controlnet(self, *args, **kwargs):
        self.calls.append(("remove_controlnet", args, kwargs))


def test_controlnet_update_mode_validates_and_updates_graph(generation_module):
    GenerationModule = generation_module.GenerationModule
    gm = GenerationModule.__new__(GenerationModule)
    gm.generation_thread = _FakeThread()

    gm.controlnet_control_mode = "balanced"
    gm.controlnet_prompt_decay = 0.825

    gm.on_controlnet_event({"action": "update_control_mode", "controlnet_control_mode": "NOT_A_MODE"})

    assert gm.controlnet_control_mode == "balanced"
    assert ("update_node", ("controlnet_control_mode", "balanced"), {}) in gm.generation_thread.calls


def test_controlnet_update_prompt_decay_clamps(generation_module):
    GenerationModule = generation_module.GenerationModule
    gm = GenerationModule.__new__(GenerationModule)
    gm.generation_thread = _FakeThread()

    gm.controlnet_control_mode = "balanced"
    gm.controlnet_prompt_decay = 0.825

    gm.on_controlnet_event({"action": "update_prompt_decay", "controlnet_prompt_decay": 2.0})

    assert gm.controlnet_prompt_decay == 1.0
    assert ("update_node", ("controlnet_prompt_decay", 1.0), {}) in gm.generation_thread.calls


def test_controlnet_add_uses_stored_mode_and_decay(generation_module):
    GenerationModule = generation_module.GenerationModule
    gm = GenerationModule.__new__(GenerationModule)
    gm.generation_thread = _FakeThread()

    gm.controlnet_control_mode = "prompt"
    gm.controlnet_prompt_decay = 0.4

    # Model path must be set for add_controlnet to be called
    gm.controlnet_model_path = "/tmp/fake_controlnet.safetensors"
    gm.controlnet_processed_image_path = None
    gm.controlnet_condition_thumb_path = None
    gm.controlnet_mask_path = None
    gm.controlnet_mask_final_path = None

    gm.on_controlnet_event(
        {
            "action": "add",
            "control_image_path": "/tmp/fake_control_image.png",
            "control_image_thumb_path": "/tmp/thumb.png",
            "conditioning_scale": 0.75,
            "control_guidance_start_end": [0.0, 1.0],
        }
    )

    add_calls = [c for c in gm.generation_thread.calls if c[0] == "add_controlnet"]
    assert add_calls, "Expected add_controlnet call"

    _name, _args, kwargs = add_calls[0]
    assert kwargs["control_mode"] == "prompt"
    assert kwargs["prompt_decay"] == 0.4


def test_controlnet_remove_resets_state_and_calls_thread(generation_module):
    GenerationModule = generation_module.GenerationModule
    gm = GenerationModule.__new__(GenerationModule)
    gm.generation_thread = _FakeThread()

    gm.controlnet_model_path = "/tmp/model"
    gm.controlnet_processed_image_path = "/tmp/img"
    gm.controlnet_processed_image_layers = [object()]
    gm.controlnet_condition_thumb_path = "/tmp/thumb"
    gm.controlnet_control_mode = "prompt"
    gm.controlnet_prompt_decay = 0.4

    gm.on_controlnet_event({"action": "remove"})

    assert ("remove_controlnet", (), {}) in gm.generation_thread.calls
    assert gm.controlnet_model_path is None
    assert gm.controlnet_processed_image_path is None
    assert gm.controlnet_processed_image_layers is None
    assert gm.controlnet_condition_thumb_path is None
    assert gm.controlnet_control_mode == "balanced"
    assert gm.controlnet_prompt_decay == 0.825
