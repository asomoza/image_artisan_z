from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def source_image_dialog_module(monkeypatch, qapp, fake_superqt, tmp_path: Path):
    """Import `SourceImageDialog` with minimal fakes for heavy dependencies."""

    from PyQt6.QtCore import QObject, pyqtSignal
    from PyQt6.QtGui import QColor, QPixmap
    from PyQt6.QtWidgets import QPushButton, QWidget

    # --- Fake buttons ---
    buttons_brush_mod = ModuleType("iartisanz.buttons.brush_erase_button")

    class BrushEraseButton(QWidget):
        brush_selected = pyqtSignal(bool)

        def __init__(self, *args, **kwargs):
            super().__init__()

    buttons_brush_mod.BrushEraseButton = BrushEraseButton

    buttons_color_mod = ModuleType("iartisanz.buttons.color_button")

    class ColorButton(QWidget):
        color_changed = pyqtSignal(tuple)

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._color = (0, 0, 0)

        def set_color(self, rgb):
            self._color = tuple(rgb)

        def color(self):
            return self._color

    buttons_color_mod.ColorButton = ColorButton

    buttons_eye_mod = ModuleType("iartisanz.buttons.eyedropper_button")

    class EyeDropperButton(QWidget):
        clicked = pyqtSignal(bool)

        def __init__(self, *args, **kwargs):
            super().__init__()

    buttons_eye_mod.EyeDropperButton = EyeDropperButton

    # --- Fake editors/widgets ---
    class FakeImageEditor(QObject):
        def __init__(self):
            super().__init__()
            self.brush_size = 42
            self.hardness = 0.33
            self.steps = 2.5
            self.brush_color = QColor(10, 20, 30)
            self._layers = []
            self.changed_image_path = None

        def set_brush_color(self, *args, **kwargs):
            return None

        def set_brush_size(self, *args, **kwargs):
            return None

        def set_brush_hardness(self, *args, **kwargs):
            return None

        def set_brush_steps(self, *args, **kwargs):
            return None

        def hide_brush_preview(self, *args, **kwargs):
            return None

        def change_layer_image(self, path: str):
            self.changed_image_path = path

        def get_all_layers(self):
            return list(self._layers)

        def restore_layers(self, layers):
            self._layers = list(layers)

    class FakeImageWidget(QObject):
        def __init__(self, editor: FakeImageEditor):
            super().__init__()
            self.image_editor = editor
            self.erase_mode_calls = 0

        def set_erase_mode(self, *args, **kwargs):
            self.erase_mode_calls += 1

    class FakeSectionWidget(QWidget):
        source_image_added = pyqtSignal(object)
        add_mask_clicked = pyqtSignal()
        delete_mask_clicked = pyqtSignal()

        def __init__(self, *args, layers=None, mask_image_path=None, **kwargs):
            super().__init__()
            self.image_widget = FakeImageWidget(FakeImageEditor())
            self._mask_image_path = mask_image_path
            self._existing_mask_buttons_set = 0
            if layers is not None:
                self.image_widget.image_editor.restore_layers(layers)

            # Match the real ImageSectionWidget API used by the dialog/tests.
            self.add_button = QPushButton()
            self.add_button.clicked.connect(lambda _checked=False: self.on_source_image_added())
            if layers is not None and len(layers) > 0:
                self.set_add_button_update()
            else:
                self.set_add_button_add()

        def set_add_button_add(self):
            self.add_button.setText("Add source image")

        def set_add_button_update(self):
            self.add_button.setText("Update source image")

        def on_source_image_added(self):
            self.source_image_added.emit(QPixmap(2, 2))

        def set_existing_mask_buttons(self):
            self._existing_mask_buttons_set += 1

    class FakeMaskSectionWidget(QWidget):
        save_mask_clicked = pyqtSignal()
        mask_canceled = pyqtSignal()
        mask_deleted = pyqtSignal()

        def __init__(self, *args, mask_image_path=None, **kwargs):
            super().__init__()
            self.image_widget = FakeImageWidget(FakeImageEditor())
            self._mask_image_path = mask_image_path

    image_section_mod = ModuleType("iartisanz.modules.generation.common.mask.image_section_widget")
    image_section_mod.ImageSectionWidget = FakeSectionWidget

    mask_section_mod = ModuleType("iartisanz.modules.generation.source_image.mask_section_widget")
    mask_section_mod.MaskSectionWidget = FakeMaskSectionWidget

    # --- Fake threads ---
    pixmap_thread_mod = ModuleType("iartisanz.modules.generation.threads.pixmap_save_thread")

    class PixmapSaveThread(QObject):
        save_finished = pyqtSignal(str, str)
        finished = pyqtSignal()
        error = pyqtSignal(str)

        _counter = 0

        def __init__(self, pixmap, *, prefix: str, temp_path: str, thumb_width: int, thumb_height: int):
            super().__init__()
            self._prefix = prefix
            self._temp_path = temp_path

        def start(self):
            PixmapSaveThread._counter += 1
            n = PixmapSaveThread._counter

            os.makedirs(self._temp_path, exist_ok=True)
            image_path = os.path.join(self._temp_path, f"{self._prefix}_{n}.png")
            thumb_path = os.path.join(self._temp_path, f"{self._prefix}_thumb_{n}.png")

            # Create placeholder files so deletion behavior is testable.
            Path(image_path).write_bytes(b"test")
            Path(thumb_path).write_bytes(b"test")
            self.save_finished.emit(image_path, thumb_path)
            self.finished.emit()

    pixmap_thread_mod.PixmapSaveThread = PixmapSaveThread

    layers_thread_mod = ModuleType("iartisanz.modules.generation.threads.save_layers_thread")

    class SaveLayersThread(QObject):
        finished = pyqtSignal()
        error = pyqtSignal(str)

        def __init__(self, layers, prefix: str, temp_path: str):
            super().__init__()

        def start(self):
            self.finished.emit()

    layers_thread_mod.SaveLayersThread = SaveLayersThread

    for name, mod in (
        ("iartisanz.buttons.brush_erase_button", buttons_brush_mod),
        ("iartisanz.buttons.color_button", buttons_color_mod),
        ("iartisanz.buttons.eyedropper_button", buttons_eye_mod),
        ("iartisanz.modules.generation.common.mask.image_section_widget", image_section_mod),
        ("iartisanz.modules.generation.source_image.mask_section_widget", mask_section_mod),
        ("iartisanz.modules.generation.threads.pixmap_save_thread", pixmap_thread_mod),
        ("iartisanz.modules.generation.threads.save_layers_thread", layers_thread_mod),
    ):
        monkeypatch.setitem(sys.modules, name, mod)

    sys.modules.pop("iartisanz.modules.generation.source_image.source_image_dialog", None)
    return importlib.import_module("iartisanz.modules.generation.source_image.source_image_dialog")


def _make_dirs(tmp_path: Path):
    from iartisanz.app.directories import DirectoriesObject

    base = tmp_path
    return DirectoriesObject(
        data_path=str(base / "data"),
        models_diffusers=str(base / "models_diffusers"),
        models_loras=str(base / "models_loras"),
        models_controlnets=str(base / "models_controlnets"),
        outputs_images=str(base / "outputs_images"),
        outputs_source_images=str(base / "outputs_source_images"),
        outputs_source_masks=str(base / "outputs_source_masks"),
        outputs_controlnet_source_images=str(base / "outputs_controlnet_source_images"),
        outputs_conditioning_images=str(base / "outputs_conditioning_images"),
        outputs_edit_source_images=str(base / "outputs_edit_source_images"),
        outputs_edit_images=str(base / "outputs_edit_images"),
        temp_path=str(base / "temp"),
    )


def test_init_connects_editor_and_sets_defaults(source_image_dialog_module, tmp_path: Path):
    from iartisanz.app.preferences import PreferencesObject

    SourceImageDialog = source_image_dialog_module.SourceImageDialog

    directories = _make_dirs(tmp_path)
    os.makedirs(directories.temp_path, exist_ok=True)

    dialog = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=str(tmp_path / "some.png"),
        source_image_layers=None,
    )

    editor = dialog.image_section_widget.image_widget.image_editor
    assert dialog.active_editor is editor
    # The dialog now sets the sliders to the editor's actual values via _connect_editor
    assert dialog.brush_size_slider.value() == editor.brush_size  # 42 from FakeImageEditor
    assert dialog.brush_hardness_slider.value() == pytest.approx(editor.hardness)  # 0.33
    assert dialog.brush_steps_slider.value() == pytest.approx(editor.steps)  # 2.5
    assert dialog.color_button.color() == (editor.brush_color.red(), editor.brush_color.green(), editor.brush_color.blue())  # (10, 20, 30)

    # If a source image path is provided and no layers are provided, the editor is instructed to load it.
    assert editor.changed_image_path == str(tmp_path / "some.png")

    # Mask editor defaults are applied by the dialog.
    mask_editor = dialog.mask_section_widget.image_widget.image_editor
    assert mask_editor.brush_size == 150
    assert mask_editor.hardness == 0.0
    assert mask_editor.steps == 1.25


def test_init_with_existing_mask_wires_sections(source_image_dialog_module, tmp_path: Path):
    from iartisanz.app.preferences import PreferencesObject

    SourceImageDialog = source_image_dialog_module.SourceImageDialog

    directories = _make_dirs(tmp_path)
    os.makedirs(directories.temp_path, exist_ok=True)

    mask_path = str(tmp_path / "mask.png")
    Path(mask_path).write_bytes(b"mask")

    dialog = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=str(tmp_path / "some.png"),
        source_image_layers=None,
        source_image_mask_path=mask_path,
    )

    # The dialog should pass the mask path down so it can be edited again.
    assert getattr(dialog.image_section_widget, "_mask_image_path", None) == mask_path
    assert getattr(dialog.mask_section_widget, "_mask_image_path", None) == mask_path
    # And it should flip the ImageSectionWidget UI to "existing mask" state.
    assert getattr(dialog.image_section_widget, "_existing_mask_buttons_set", 0) == 1


def test_on_source_image_added_publishes_update_and_update_layers(source_image_dialog_module, tmp_path: Path):
    from PyQt6.QtGui import QPixmap

    from iartisanz.app.preferences import PreferencesObject
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer

    SourceImageDialog = source_image_dialog_module.SourceImageDialog

    directories = _make_dirs(tmp_path)
    os.makedirs(directories.temp_path, exist_ok=True)

    old_path = Path(directories.temp_path) / "old_source.png"
    old_path.write_text("x")

    dialog = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=str(old_path),
        source_image_layers=None,
    )

    # Provide layers so `update_layers` payload can be validated.
    layer = ImageEditorLayer(layer_id=0, layer_name="L0", order=0)
    layer.pixmap_item = object()
    dialog.image_section_widget.image_widget.image_editor.restore_layers([layer])

    published: list[tuple[str, dict]] = []
    dialog.event_bus.unsubscribe_all()
    dialog.event_bus.subscribe("source_image", lambda data: published.append(("source_image", data)))

    pixmap = QPixmap(2, 2)
    dialog.on_source_image_added(pixmap)

    assert not old_path.exists(), "Existing temp source image should be removed"

    assert published, "Expected source_image events to be published"
    assert published[0][1]["action"] == "update"
    assert "source_image_path" in published[0][1]
    assert "source_thumb_path" in published[0][1]

    assert published[-1][1]["action"] == "update_layers"
    layers = published[-1][1]["layers"]
    assert len(layers) == 1
    assert layers[0].pixmap_item is None

    assert dialog.dialog_busy is False
    assert dialog.pixmap_save_thread is None
    assert dialog.save_layers_thread is None


def test_on_pixmap_saved_is_noop_when_path_unchanged(source_image_dialog_module, tmp_path: Path, monkeypatch):
    from iartisanz.app.preferences import PreferencesObject

    SourceImageDialog = source_image_dialog_module.SourceImageDialog

    directories = _make_dirs(tmp_path)
    os.makedirs(directories.temp_path, exist_ok=True)

    dialog = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=str(Path(directories.temp_path) / "same.png"),
        source_image_layers=None,
    )

    dialog.event_bus.unsubscribe_all()
    published: list[dict] = []
    dialog.event_bus.subscribe("source_image", lambda data: published.append(data))

    monkeypatch.setattr(dialog, "save_layers", lambda: (_ for _ in ()).throw(AssertionError("save_layers called")))

    dialog.on_pixmap_saved(dialog.source_image_path, str(Path(directories.temp_path) / "thumb.png"))
    assert published == []


def test_dialog_full_complex_cycle(source_image_dialog_module, tmp_path: Path):
    from iartisanz.app.preferences import PreferencesObject
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer

    SourceImageDialog = source_image_dialog_module.SourceImageDialog

    directories = _make_dirs(tmp_path)
    os.makedirs(directories.temp_path, exist_ok=True)

    layer = ImageEditorLayer(layer_id=0, layer_name="layer", order=0)
    layer.pixmap_item = object()
    one_layer = [layer]

    # 1) First open: no existing layers -> button should say "Add source image".
    dialog1 = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=None,
        source_image_layers=None,
    )
    dialog1.image_section_widget.image_widget.image_editor.restore_layers(one_layer)
    assert dialog1.image_section_widget.add_button.text() == "Add source image"

    dialog1.event_bus.unsubscribe_all()
    dialog1.image_section_widget.add_button.click()
    p1 = Path(dialog1.source_image_path)
    assert p1.exists(), "First saved source image should exist in temp"

    # 2) Second open: restore with one layer -> button should say "Update source image".
    dialog2 = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=str(p1),
        source_image_layers=one_layer,
    )
    dialog2.image_section_widget.image_widget.image_editor.restore_layers(one_layer)
    assert dialog2.image_section_widget.add_button.text() == "Update source image"

    dialog2.event_bus.unsubscribe_all()
    dialog2.image_section_widget.add_button.click()
    p2 = Path(dialog2.source_image_path)
    assert p2.exists(), "Second saved source image should exist in temp"
    assert not p1.exists(), "Previous temp source image should be removed on update"

    # 3) Third open which triggers a second update
    dialog3 = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=str(p2),
        source_image_layers=one_layer,
    )
    dialog3.image_section_widget.image_widget.image_editor.restore_layers(one_layer)
    assert dialog3.image_section_widget.add_button.text() == "Update source image"

    dialog3.event_bus.unsubscribe_all()
    dialog3.image_section_widget.add_button.click()
    p3 = Path(dialog3.source_image_path)
    t3 = Path(dialog3.source_thumb_path)
    assert p3.exists(), "Third saved source image should exist in temp"
    assert t3.exists(), "Third saved source image thumbnail should exist in temp"
    assert dialog3.dialog_busy is False

    # 4) Delete/remove the source image (simulate panel remove behavior) and open again.
    p3.unlink()
    t3.unlink()
    assert not p3.exists(), "Deleted temp source image should be removed"
    assert not t3.exists(), "Deleted temp source image thumbnail should be removed"

    dialog4 = SourceImageDialog(
        "source_image",
        directories,
        PreferencesObject(),
        None,
        512,
        512,
        source_image_path=None,
        source_image_layers=None,
    )
    dialog4.image_section_widget.image_widget.image_editor.restore_layers(one_layer)
    assert dialog4.image_section_widget.add_button.text() == "Add source image"

    # 5) Add a new source image after deletion.
    dialog4.event_bus.unsubscribe_all()
    dialog4.image_section_widget.add_button.click()
    p4 = Path(dialog4.source_image_path)
    t4 = Path(dialog4.source_thumb_path)
    assert p4.exists(), "New saved source image should exist in temp"
    assert t4.exists(), "New saved source image thumbnail should exist in temp"
    assert p4 != p3
    assert dialog4.dialog_busy is False
