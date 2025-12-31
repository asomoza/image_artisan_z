from __future__ import annotations

import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from iartisanz.modules.generation.image.image_widget import ImageWidget


if TYPE_CHECKING:
    from iartisanz.modules.generation.widgets.image_viewer_simple_widget import ImageViewerSimpleWidget


class ImageSectionWidget(QWidget):
    add_mask_clicked = pyqtSignal(QPixmap)
    source_image_added = pyqtSignal(QPixmap)
    delete_mask_clicked = pyqtSignal()

    def __init__(
        self,
        image_viewer: ImageViewerSimpleWidget,
        target_width: int,
        target_height: int,
        outputh_path: str,
        temp_path: str,
        layers: list = None,
        mask_image_path: str = None,
    ):
        super().__init__()

        self.image_viewer = image_viewer
        self.target_width = target_width
        self.target_height = target_height
        self.outputh_path = outputh_path
        self.temp_path = temp_path

        if mask_image_path is not None:
            self.mask_button_text = "Edit Mask"
        else:
            self.mask_button_text = "Add Mask"

        self.original_image = None

        self.init_ui()

        if layers is not None and len(layers) > 0:
            self.image_widget.restore_layers(layers)
        else:
            self.image_widget.add_layer()

        self.image_widget.set_enabled(True)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(1, 3, 0, 0)
        main_layout.setSpacing(0)

        self.image_widget = ImageWidget(
            "Source image",
            "src_image",
            self.image_viewer,
            self.target_width,
            self.target_height,
            show_layer_manager=True,
            save_directory=self.outputh_path,
            temp_path=self.temp_path,
        )
        self.image_widget.image_loaded.connect(self.on_image_loaded)
        main_layout.addWidget(self.image_widget)

        bottom_layout = QHBoxLayout()
        bottom_layout.setContentsMargins(5, 0, 5, 0)

        self.add_mask_button = QPushButton(self.mask_button_text)
        self.add_mask_button.clicked.connect(self.on_add_mask)
        self.add_mask_button.setObjectName("blue_button")
        bottom_layout.addWidget(self.add_mask_button)
        self.delete_mask_button = QPushButton("Delete Mask")
        self.delete_mask_button.clicked.connect(self.on_delete_mask)
        self.delete_mask_button.setObjectName("red_button")
        self.delete_mask_button.setVisible(False)
        bottom_layout.addWidget(self.delete_mask_button)
        self.add_button = QPushButton("Set Source Image")
        self.add_button.setObjectName("green_button")
        self.add_button.clicked.connect(self.on_source_image_added)
        bottom_layout.addWidget(self.add_button)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def on_image_loaded(self, image_path: str):
        if self.original_image is not None and os.path.isfile(self.original_image):
            if self.original_image.startswith(self.temp_path):
                os.remove(self.original_image)
        self.original_image = image_path

    def on_add_mask(self):
        pixmap = self.image_widget.image_editor.get_scene_as_pixmap()
        self.add_mask_clicked.emit(pixmap)

    def on_source_image_added(self):
        pixmap = self.image_widget.image_editor.get_scene_as_pixmap()
        self.source_image_added.emit(pixmap)

    def disable_buttons(self, state: bool = True):
        self.add_button.setDisabled(state)
        self.add_mask_button.setDisabled(state)

    def set_existing_mask_buttons(self):
        self.add_mask_button.setText("Edit Mask")
        self.delete_mask_button.setVisible(True)

    def on_delete_mask(self):
        self.add_mask_button.setText("Add Mask")
        self.delete_mask_button.setVisible(False)
        self.delete_mask_clicked.emit()
