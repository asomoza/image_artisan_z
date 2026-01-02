from __future__ import annotations

from pathlib import Path


def _write_test_png(path: Path, *, width: int = 16, height: int = 16):
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QPixmap

    pixmap = QPixmap(width, height)
    pixmap.fill(Qt.GlobalColor.transparent)
    assert pixmap.save(str(path)), f"Failed to save test pixmap to {path}"


def test_restore_layers_adds_all_layers_to_manager_and_scene(qapp, tmp_path: Path):
    from iartisanz.modules.generation.image.image_editor import ImageEditor
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer

    img0 = tmp_path / "layer0.png"
    img1 = tmp_path / "layer1.png"
    _write_test_png(img0)
    _write_test_png(img1)

    editor = ImageEditor(target_width=64, target_height=64, aspect_ratio=1.0)

    layers = [
        ImageEditorLayer(layer_id=1, image_path=str(img1), order=1, locked=True, visible=True),
        ImageEditorLayer(layer_id=0, image_path=str(img0), order=0, locked=True, visible=True),
    ]

    editor.restore_layers(layers)

    # LayerManager should contain both layers
    assert len(editor.layer_manager.layers) == 2

    # LayerManager stores layers sorted by order (background first)
    assert [layer.order for layer in editor.layer_manager.layers] == [0, 1]

    # QGraphicsScene should contain both pixmap items
    assert len(editor.scene.items()) == 2

    # Each restored layer should now have a pixmap item
    assert all(layer.pixmap_item is not None for layer in layers)

    # Z-values should match the effective order assigned in restore
    for layer in layers:
        assert layer.pixmap_item.zValue() == layer.order

    # The topmost layer is selected by restore_layers
    assert editor.selected_layer.order == 1


def test_clear_all_removes_scene_items_and_layers(qapp, tmp_path: Path):
    from iartisanz.modules.generation.image.image_editor import ImageEditor
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer

    img0 = tmp_path / "layer0.png"
    img1 = tmp_path / "layer1.png"
    _write_test_png(img0)
    _write_test_png(img1)

    editor = ImageEditor(target_width=64, target_height=64, aspect_ratio=1.0)
    editor.restore_layers(
        [
            ImageEditorLayer(layer_id=0, image_path=str(img0), order=0),
            ImageEditorLayer(layer_id=1, image_path=str(img1), order=1),
        ]
    )

    assert len(editor.layer_manager.layers) == 2
    assert len(editor.scene.items()) == 2

    editor.clear_all()

    assert len(editor.layer_manager.layers) == 0
    assert len(editor.scene.items()) == 0


def test_layer_manager_widget_restore_layers_populates_list(qapp):
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer
    from iartisanz.modules.generation.image.layer_manager_widget import LayerManagerWidget

    widget = LayerManagerWidget(start_expanded=True)

    layers = [
        ImageEditorLayer(layer_id=0, layer_name="Background", order=0),
        ImageEditorLayer(layer_id=1, layer_name="Layer #1", order=1),
    ]

    widget.restore_layers(layers)

    assert widget.list_widget.count() == 2
    assert widget.list_widget.currentRow() == 0

    item0 = widget.list_widget.item(0)
    item1 = widget.list_widget.item(1)

    w0 = widget.list_widget.itemWidget(item0)
    w1 = widget.list_widget.itemWidget(item1)

    # Layer manager displays topmost layer first (highest order).
    assert w0.layer is layers[1]
    assert w1.layer is layers[0]
