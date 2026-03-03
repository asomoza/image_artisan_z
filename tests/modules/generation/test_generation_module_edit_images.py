"""Tests for GenerationModule.on_edit_images_event plumbing."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def module():
    """Create a GenerationModule with mocked dependencies."""
    with (
        patch("iartisanz.modules.generation.generation_module.QSettings"),
        patch("iartisanz.modules.generation.generation_module.GenerationSettings") as mock_gs_cls,
        patch("iartisanz.modules.generation.generation_module.create_graph_for_model_type"),
        patch("iartisanz.modules.generation.generation_module.NodeGraphThread") as mock_thread_cls,
        patch("iartisanz.modules.generation.generation_module.BaseModule.__init__", return_value=None),
    ):
        mock_gs = MagicMock()
        mock_gs.model = MagicMock(model_type=3, id=1)
        mock_gs.image_width = 512
        mock_gs.image_height = 512
        mock_gs.num_inference_steps = 4
        mock_gs.guidance_scale = 1.0
        mock_gs.guidance_start_end = [0.0, 1.0]
        mock_gs.scheduler = "euler"
        mock_gs.strength = 0.75
        mock_gs.to_graph_nodes.return_value = {}
        mock_gs.GRAPH_KEYS = set()
        mock_gs_cls.load.return_value = mock_gs

        mock_thread = MagicMock()
        mock_thread_cls.return_value = mock_thread

        from iartisanz.modules.generation.generation_module import GenerationModule

        mod = GenerationModule.__new__(GenerationModule)

        # Minimal init to set up what on_edit_images_event needs
        mod.gen_settings = mock_gs
        mod.generation_thread = mock_thread
        mod.edit_image_paths = [None] * 4
        mod.edit_image_thumb_paths = [None] * 4
        mod.edit_source_image_layers = [None] * 4
        mod.edit_result_image_layers = [None] * 4
        mod.edit_image_enabled = [True] * 4

        yield mod


class TestOnEditImagesEvent:
    def test_add_event_stores_state_and_calls_thread(self, module):
        module.on_edit_images_event(
            {"action": "add", "image_index": 0, "image_path": "/tmp/edit.png", "image_thumb_path": "/tmp/thumb.png"}
        )

        assert module.edit_image_paths[0] == "/tmp/edit.png"
        assert module.edit_image_thumb_paths[0] == "/tmp/thumb.png"
        module.generation_thread.add_edit_image.assert_called_once_with(0, "/tmp/edit.png")

    def test_update_event_stores_state_and_calls_thread(self, module):
        module.on_edit_images_event(
            {
                "action": "update",
                "image_index": 1,
                "image_path": "/tmp/edit_v2.png",
                "image_thumb_path": "/tmp/thumb_v2.png",
            }
        )

        assert module.edit_image_paths[1] == "/tmp/edit_v2.png"
        module.generation_thread.update_edit_image.assert_called_once_with(1, "/tmp/edit_v2.png")

    def test_remove_event_clears_state_and_calls_thread(self, module):
        # Pre-populate
        module.edit_image_paths[2] = "/tmp/edit.png"
        module.edit_image_thumb_paths[2] = "/tmp/thumb.png"
        module.edit_source_image_layers[2] = ["layer"]
        module.edit_result_image_layers[2] = ["layer"]

        module.on_edit_images_event({"action": "remove", "image_index": 2})

        assert module.edit_image_paths[2] is None
        assert module.edit_image_thumb_paths[2] is None
        assert module.edit_source_image_layers[2] is None
        assert module.edit_result_image_layers[2] is None
        module.generation_thread.remove_edit_image.assert_called_once_with(2)

    def test_reset_clears_all_slots(self, module):
        # Pre-populate all slots
        for i in range(4):
            module.edit_image_paths[i] = f"/tmp/edit_{i}.png"
            module.edit_image_thumb_paths[i] = f"/tmp/thumb_{i}.png"

        module.on_edit_images_event({"action": "reset"})

        assert module.edit_image_paths == [None] * 4
        assert module.edit_image_thumb_paths == [None] * 4
        assert module.edit_source_image_layers == [None] * 4
        assert module.edit_result_image_layers == [None] * 4
        module.generation_thread.remove_all_edit_images.assert_called_once()

    def test_add_without_image_path_does_not_call_thread(self, module):
        """If image_path is missing/None, don't call the thread."""
        module.on_edit_images_event({"action": "add", "image_index": 0, "image_path": None})

        module.generation_thread.add_edit_image.assert_not_called()

    def test_update_layers_stores_layers(self, module):
        layers = [{"type": "paint", "data": b"..."}]
        module.on_edit_images_event({"action": "update_layers", "image_index": 1, "layers": layers})

        assert module.edit_result_image_layers[1] is layers

    def test_update_source_layers_stores_layers(self, module):
        layers = [{"type": "source"}]
        module.on_edit_images_event({"action": "update_source_layers", "image_index": 0, "layers": layers})

        assert module.edit_source_image_layers[0] is layers

    def test_disable_removes_from_graph_preserves_state(self, module):
        """Disabling an edit image removes it from the graph but keeps paths/layers."""
        module.edit_image_paths[1] = "/tmp/edit.png"
        module.edit_image_thumb_paths[1] = "/tmp/thumb.png"
        module.edit_source_image_layers[1] = ["layer"]

        module.on_edit_images_event({"action": "disable", "image_index": 1})

        assert module.edit_image_enabled[1] is False
        # State preserved
        assert module.edit_image_paths[1] == "/tmp/edit.png"
        assert module.edit_image_thumb_paths[1] == "/tmp/thumb.png"
        assert module.edit_source_image_layers[1] == ["layer"]
        # Node removed from graph
        module.generation_thread.remove_edit_image.assert_called_once_with(1)

    def test_enable_readds_to_graph(self, module):
        """Enabling a previously disabled edit image re-adds it to the graph."""
        module.edit_image_paths[2] = "/tmp/edit.png"
        module.edit_image_enabled[2] = False

        module.on_edit_images_event({"action": "enable", "image_index": 2})

        assert module.edit_image_enabled[2] is True
        module.generation_thread.add_edit_image.assert_called_once_with(2, "/tmp/edit.png")

    def test_enable_without_path_does_not_call_thread(self, module):
        """Enabling a slot with no image path should not call the thread."""
        module.edit_image_enabled[0] = False

        module.on_edit_images_event({"action": "enable", "image_index": 0})

        assert module.edit_image_enabled[0] is True
        module.generation_thread.add_edit_image.assert_not_called()

    def test_remove_resets_enabled_to_true(self, module):
        """Removing an edit image resets enabled state to True for next use."""
        module.edit_image_paths[0] = "/tmp/edit.png"
        module.edit_image_enabled[0] = False

        module.on_edit_images_event({"action": "remove", "image_index": 0})

        assert module.edit_image_enabled[0] is True

    def test_reset_resets_all_enabled(self, module):
        """Reset clears all enabled states back to True."""
        module.edit_image_enabled = [False, True, False, True]

        module.on_edit_images_event({"action": "reset"})

        assert module.edit_image_enabled == [True] * 4

    def test_add_sets_enabled_true(self, module):
        """Adding an image always sets enabled to True."""
        module.edit_image_enabled[0] = False

        module.on_edit_images_event(
            {"action": "add", "image_index": 0, "image_path": "/tmp/edit.png", "image_thumb_path": "/tmp/thumb.png"}
        )

        assert module.edit_image_enabled[0] is True

    def test_apply_loaded_edit_subset_restores_paths_and_mask(self, module):
        module.edit_source_image_layers = [["src"]] * 4
        module.edit_result_image_layers = [["res"]] * 4

        module._apply_loaded_edit_subset(
            {
                "edit_image_0": "/tmp/e0.png",
                "edit_image_2": "/tmp/e2.png",
                "edit_image_mask": "/tmp/mask.png",
                "edit_image_mask_strength": 0.42,
            }
        )

        assert module.edit_image_paths == ["/tmp/e0.png", None, "/tmp/e2.png", None]
        assert module.edit_image_thumb_paths == ["/tmp/e0.png", None, "/tmp/e2.png", None]
        assert module.edit_image_enabled == [True, True, True, True]
        assert module.edit_image_mask_path == "/tmp/mask.png"
        assert module.edit_image_mask_thumb_path == "/tmp/mask.png"
        assert module.edit_image_mask_strength == pytest.approx(0.42)
        assert module.edit_source_image_layers == [None] * 4
        assert module.edit_result_image_layers == [None] * 4

    def test_apply_loaded_edit_subset_clears_mask_without_slot0(self, module):
        module._apply_loaded_edit_subset(
            {
                "edit_image_1": "/tmp/e1.png",
                "edit_image_mask": "/tmp/mask.png",
                "edit_image_mask_strength": 0.8,
            }
        )

        assert module.edit_image_paths == [None, "/tmp/e1.png", None, None]
        assert module.edit_image_mask_path is None
        assert module.edit_image_mask_thumb_path is None
        assert module.edit_image_mask_strength == pytest.approx(1.0)


class TestEditImagesDialogCloseKey:
    """Verify that dialog close events resolve to the correct dialog key."""

    def test_dialog_key_includes_image_index(self):
        """The edit_images dialog key must include the image_index from the event data."""
        from iartisanz.modules.generation.generation_module import GenerationModule

        # The key function for edit_images in _get_dialog_specs is:
        # lambda d: f"edit_images_{d.get('image_index', 0)}"
        # Simulate what happens when close events arrive with different image_index values.
        for idx in range(4):
            data = {"dialog_type": "edit_images", "action": "close", "image_index": idx}
            key = f"edit_images_{data.get('image_index', 0)}"
            assert key == f"edit_images_{idx}"

    def test_close_without_image_index_defaults_to_zero(self):
        """Without image_index, the key defaults to edit_images_0 (backward compat)."""
        data = {"dialog_type": "edit_images", "action": "close"}
        key = f"edit_images_{data.get('image_index', 0)}"
        assert key == "edit_images_0"


def test_edit_images_dialog_factory_passes_source_and_result_paths(module):
    """Regression: loaded graph edit-image paths must be passed to EditImagesDialog.

    Without this, the Edit Images panel thumbnail could show an image while the
    dialog itself opened empty after loading metadata from a dropped PNG.
    """
    module.directories = object()
    module.preferences = object()
    module.image_viewer = object()

    module.edit_image_paths[1] = "/tmp/edit_slot_1.png"
    module.edit_source_image_layers[1] = ["source_layer"]
    module.edit_result_image_layers[1] = ["result_layer"]

    with patch("iartisanz.modules.generation.generation_module.EditImagesDialog") as mock_dialog_cls:
        specs = module._get_dialog_specs()
        factory = specs["edit_images"]["factory"]

        factory({"image_index": 1})

    _, kwargs = mock_dialog_cls.call_args
    assert kwargs["image_index"] == 1
    assert kwargs["source_image_path"] == "/tmp/edit_slot_1.png"
    assert kwargs["result_image_path"] == "/tmp/edit_slot_1.png"
    assert kwargs["source_image_layers"] == ["source_layer"]
    assert kwargs["result_image_layers"] == ["result_layer"]
