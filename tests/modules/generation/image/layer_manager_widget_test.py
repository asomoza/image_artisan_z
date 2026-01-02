from __future__ import annotations

import pytest


def _layer_orders_in_list(widget) -> list[int]:
    orders: list[int] = []
    for i in range(widget.list_widget.count()):
        item = widget.list_widget.item(i)
        row_widget = widget.list_widget.itemWidget(item)
        orders.append(row_widget.layer.order)
    return orders


def test_restore_layers_shows_topmost_first_and_selects_it(qapp, fake_superqt):
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer
    from iartisanz.modules.generation.image.layer_manager_widget import LayerManagerWidget

    widget = LayerManagerWidget(start_expanded=True)

    layers = [
        ImageEditorLayer(layer_id=0, layer_name="Background", order=0),
        ImageEditorLayer(layer_id=2, layer_name="Top", order=10),
        ImageEditorLayer(layer_id=1, layer_name="Middle", order=5),
    ]

    selected: list[ImageEditorLayer] = []
    widget.layer_selected.connect(lambda layer: selected.append(layer))

    widget.restore_layers(layers)

    assert widget.list_widget.count() == 3
    assert _layer_orders_in_list(widget) == [10, 5, 0]

    assert widget.list_widget.currentRow() == 0
    assert widget.selected_layer is not None
    assert widget.selected_layer.order == 10
    assert selected and selected[-1].order == 10


def test_add_layer_inserts_at_top_and_selects(qapp, fake_superqt):
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer
    from iartisanz.modules.generation.image.layer_manager_widget import LayerManagerWidget

    widget = LayerManagerWidget(start_expanded=True)

    base = ImageEditorLayer(layer_id=0, layer_name="Background", order=0)
    widget.restore_layers([base])

    new_layer = ImageEditorLayer(layer_id=1, layer_name="Layer #1", order=1)
    widget.add_layer(new_layer)

    assert widget.list_widget.count() == 2

    item0 = widget.list_widget.item(0)
    row0 = widget.list_widget.itemWidget(item0)

    assert row0.layer is new_layer
    assert widget.list_widget.currentRow() == 0
    assert widget.selected_layer is new_layer


def test_delete_layer_removes_selected_layer(qapp, fake_superqt):
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer
    from iartisanz.modules.generation.image.layer_manager_widget import LayerManagerWidget

    widget = LayerManagerWidget(start_expanded=True)

    top = ImageEditorLayer(layer_id=1, layer_name="Top", order=1)
    bottom = ImageEditorLayer(layer_id=0, layer_name="Background", order=0)

    widget.restore_layers([bottom, top])

    assert widget.selected_layer is top
    assert widget.list_widget.count() == 2

    widget.delete_layer()

    assert widget.list_widget.count() == 1
    remaining_item = widget.list_widget.item(0)
    remaining_widget = widget.list_widget.itemWidget(remaining_item)
    assert remaining_widget.layer is bottom


def test_handle_item_selected_updates_sliders(qapp, fake_superqt):
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer
    from iartisanz.modules.generation.image.layer_manager_widget import LayerManagerWidget

    widget = LayerManagerWidget(start_expanded=True)

    layer_a = ImageEditorLayer(
        layer_id=0, layer_name="A", order=0, opacity=0.25, brightness=-0.5, contrast=2.0, saturation=1.5
    )
    layer_b = ImageEditorLayer(
        layer_id=1, layer_name="B", order=1, opacity=0.9, brightness=0.1, contrast=-10.0, saturation=0.8
    )

    widget.restore_layers([layer_a, layer_b])

    widget.list_widget.setCurrentRow(1)

    assert widget.selected_layer is layer_a
    assert widget.opacity_slider.value() == pytest.approx(0.25)
    assert widget.brightness_slider.value() == pytest.approx(-0.5)
    assert widget.contrast_slider.value() == pytest.approx(2.0)
    assert widget.saturation_slider.value() == pytest.approx(1.5)
