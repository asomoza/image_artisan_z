from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QImageReader, QPixmap
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QPushButton, QSizePolicy, QSpacerItem, QVBoxLayout, QWidget

from iartisanz.layouts.aspect_ratio_layout import AspectRatioLayout
from iartisanz.modules.generation.image.image_control import ImageControl
from iartisanz.modules.generation.image.image_editor import ImageEditor
from iartisanz.modules.generation.image.layer_manager_widget import LayerManagerWidget


if TYPE_CHECKING:
    from iartisanz.modules.generation.data_objects.mask_image_data_object import MaskImageDataObject
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer
    from iartisanz.modules.generation.widgets.image_viewer_simple_widget import ImageViewerSimpleWidget


class MaskWidget(QWidget):
    image_loaded = pyqtSignal()
    image_changed = pyqtSignal()

    def __init__(
        self,
        text: str,
        prefix: str,
        image_viewer: ImageViewerSimpleWidget,
        editor_width: int,
        editor_height: int,
        temp_directory: str = "tmp/",
    ):
        super().__init__()

        self.setObjectName("image_widget")
        self.text = text
        self.image_viewer = image_viewer
        self.image_path = None
        self.temp_directory = temp_directory

        self.setAcceptDrops(True)

        self.prefix = prefix
        self.editor_width = editor_width
        self.editor_height = editor_height
        self.aspect_ratio = float(self.editor_width) / float(self.editor_height)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(5)

        top_layout = QHBoxLayout()
        fit_image_button = QPushButton("Fit")
        top_layout.addWidget(fit_image_button)
        current_image_button = QPushButton("Current")
        current_image_button.setToolTip("Copy the current generated image an set it as an image.")
        top_layout.addWidget(current_image_button)
        load_image_button = QPushButton("Load image")
        load_image_button.setToolTip("Load an image from your computer.")
        load_image_button.clicked.connect(self.on_load_image)
        top_layout.addWidget(load_image_button)
        main_layout.addLayout(top_layout)

        middle_layout = QHBoxLayout()

        layers_layout = QVBoxLayout()
        self.layer_manager_widget = LayerManagerWidget(start_expanded=True, expandable=False, use_for_mask=True)
        self.layer_manager_widget.layer_selected.connect(self.on_layer_selected)
        layers_layout.addWidget(self.layer_manager_widget)
        middle_layout.addLayout(layers_layout)

        image_widget = QWidget()
        self.image_editor = ImageEditor(self.editor_width, self.editor_height, self.aspect_ratio, self.temp_directory)
        self.image_editor.set_enable_copy(False)
        self.image_editor.set_enable_save(False)
        self.image_editor.image_changed.connect(self.on_image_changed)
        self.image_editor.image_scaled.connect(self.update_image_scale)
        self.image_editor.image_moved.connect(self.update_image_position)
        self.image_editor.image_rotated.connect(self.update_image_angle)
        self.image_editor.image_pasted.connect(self.on_image_pasted)
        editor_layout = AspectRatioLayout(image_widget, self.aspect_ratio)
        editor_layout.addWidget(self.image_editor)
        image_widget.setLayout(editor_layout)
        middle_layout.addWidget(image_widget, 4)

        main_layout.addLayout(middle_layout)

        image_bottom_layout = QHBoxLayout()
        reset_image_button = QPushButton("Reset layer")
        image_bottom_layout.addWidget(reset_image_button)
        clear_mask_button = QPushButton("Clear mask")
        clear_mask_button.clicked.connect(self.on_clear_mask)
        image_bottom_layout.addWidget(clear_mask_button)
        reset_view_button = QPushButton("Reset view")
        reset_view_button.setToolTip("Reset zoom and position of the viewport.")
        image_bottom_layout.addWidget(reset_view_button)
        main_layout.addLayout(image_bottom_layout)

        image_controls_layout = QHBoxLayout()
        self.image_scale_control = ImageControl("Scale: ", 1.0, 3)
        self.image_scale_control.value_changed.connect(self.image_editor.set_image_scale)
        image_controls_layout.addWidget(self.image_scale_control)
        self.image_x_pos_control = ImageControl("X Pos: ", 0, 0)
        self.image_x_pos_control.value_changed.connect(self.image_editor.set_image_x)
        image_controls_layout.addWidget(self.image_x_pos_control)
        self.image_y_pos_control = ImageControl("Y Pos: ", 0, 0)
        self.image_y_pos_control.value_changed.connect(self.image_editor.set_image_y)
        image_controls_layout.addWidget(self.image_y_pos_control)
        self.image_rotation_control = ImageControl("Rotation: ", 0, 0)
        self.image_rotation_control.value_changed.connect(self.image_editor.rotate_image)
        image_controls_layout.addWidget(self.image_rotation_control)
        main_layout.addLayout(image_controls_layout)

        main_layout.addSpacerItem(QSpacerItem(0, 0, QSizePolicy.Policy.Expanding))

        main_layout.setStretch(0, 0)
        main_layout.setStretch(1, 1)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 0)

        self.setLayout(main_layout)

        fit_image_button.clicked.connect(self.image_editor.fit_image)
        current_image_button.clicked.connect(self.set_current_image)
        reset_image_button.clicked.connect(self.image_editor.clear_and_restore)
        reset_view_button.clicked.connect(self.image_editor.reset_view)

    def set_layers(self, mask_image_path: str = None):
        self.image_layer = self.image_editor.add_layer()
        self.image_layer.layer_name = "Reference Image"
        self.layer_manager_widget.add_layer(self.image_layer)

        self.mask_layer = self.image_editor.add_layer(mask_image_path)
        self.mask_layer.layer_name = "Mask"
        self.mask_layer.set_opacity(1.0)
        self.layer_manager_widget.add_layer(self.mask_layer)
        self.image_editor.selected_layer = self.mask_layer

    def update_image_scale(self, scale: float):
        self.image_scale_control.set_value(scale)

    def update_image_position(self, x, y):
        self.image_x_pos_control.set_value(x)
        self.image_y_pos_control.set_value(y)

    def update_image_angle(self, angle):
        self.image_rotation_control.set_value(angle)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.image_editor.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.image_editor.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.image_editor.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()

            reader = QImageReader(path)

            if reader.canRead():
                self.image_editor.change_layer_image(path)
                self.image_loaded.emit()

    def on_load_image(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        selected_path, _ = QFileDialog.getOpenFileName(self, "Load Image", "", "Images (*.png *.jpg)", options=options)
        if selected_path:
            self.image_editor.change_layer_image(selected_path)
            self.image_loaded.emit()

    def reload_mask(self, mask_image: MaskImageDataObject):
        if mask_image.back is not None:
            self.image_editor.selected_layer = self.image_layer
            self.image_editor.change_layer_image(mask_image.background_image.image_original)

            self.set_image_parameters(
                mask_image.background_image.image_scale,
                mask_image.background_image.image_x_pos,
                mask_image.background_image.image_y_pos,
                mask_image.background_image.image_rotation,
            )

        if mask_image.mask_image is not None:
            self.image_editor.selected_layer = self.drawing_layer
            self.image_editor.change_layer_image(mask_image.mask_image.image)
            self.image_editor.selected_layer.locked = False

    def set_image_parameters(self, scale: float, x: int, y: int, angle: float):
        self.image_scale_control.set_value(scale)
        self.image_editor.set_image_scale(scale)
        self.image_x_pos_control.set_value(x)
        self.image_editor.set_image_x(x)
        self.image_y_pos_control.set_value(y)
        self.image_editor.set_image_y(y)
        self.image_rotation_control.set_value(angle)
        self.image_editor.rotate_image(angle)

    def reset_controls(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()

    def clear_image(self):
        self.image_scale_control.reset()
        self.image_x_pos_control.reset()
        self.image_y_pos_control.reset()
        self.image_rotation_control.reset()
        self.image_editor.clear_all()

    def on_image_changed(self):
        self.image_changed.emit()

    def on_reset_drawings(self):
        self.image_editor.clear_and_restore()

    def set_erase_mode(self, value: bool):
        self.image_editor.erasing = value

    def on_clear_mask(self):
        pixmap = QPixmap(self.editor_width, self.editor_height)
        pixmap.fill(Qt.GlobalColor.transparent)

        self.image_editor.selected_layer = self.mask_layer
        self.image_editor.change_layer_image(pixmap)

    def on_image_pasted(self, pixmap: QPixmap):
        self.image_editor.change_layer_image(pixmap)
        self.image_loaded.emit()

    def on_layer_selected(self, layer: ImageEditorLayer):
        self.image_editor.selected_layer = layer
        self.update_image_position(layer.pixmap_item.x(), layer.pixmap_item.y())
        self.update_image_angle(layer.pixmap_item.rotation())
        self.update_image_scale(layer.pixmap_item.scale())

    def set_current_image(self):
        self.image_editor.change_layer_image(self.image_viewer.pixmap_item.pixmap())
        self.image_loaded.emit()
        self.reset_controls()
