from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from iartisanz.modules.generation.image.mask_widget import MaskWidget


if TYPE_CHECKING:
    from iartisanz.modules.generation.lora.image_viewer_simple_widget import ImageViewerSimpleWidget


class MaskSectionWidget(QWidget):
    save_mask_clicked = pyqtSignal(QPixmap, float)
    mask_canceled = pyqtSignal()
    mask_deleted = pyqtSignal()

    def __init__(
        self,
        image_viewer: ImageViewerSimpleWidget,
        target_width: int,
        target_height: int,
        outputh_path: str,
        temp_directory: str,
        mask_image_path: str = None,
    ):
        super().__init__()

        self.image_viewer = image_viewer
        self.target_width = target_width
        self.target_height = target_height
        self.outputh_path = outputh_path
        self.temp_directory = temp_directory

        self.original_image = None

        self.init_ui()

        self.image_widget.set_layers(mask_image_path)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(1, 3, 0, 0)
        main_layout.setSpacing(0)

        self.image_widget = MaskWidget(
            "Inpaint Mask",
            "inpaint_mask",
            self.image_viewer,
            self.target_width,
            self.target_height,
            self.temp_directory,
        )
        main_layout.addWidget(self.image_widget)

        bottom_layout = QHBoxLayout()
        self.save_mask_button = QPushButton("Save mask")
        self.save_mask_button.setObjectName("green_button")
        self.save_mask_button.clicked.connect(self.on_save_mask)
        bottom_layout.addWidget(self.save_mask_button)
        delete_mask_button = QPushButton("Delete mask")
        delete_mask_button.setObjectName("red_button")
        delete_mask_button.clicked.connect(self.on_delete_mask)
        bottom_layout.addWidget(delete_mask_button)
        cancel_mask_button = QPushButton("Cancel")
        cancel_mask_button.setObjectName("red_button")
        cancel_mask_button.clicked.connect(self.on_cancel_mask)
        bottom_layout.addWidget(cancel_mask_button)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def on_save_mask(self):
        mask_pixmap = self.image_widget.mask_layer.pixmap_item.pixmap()
        opacity = self.image_widget.mask_layer.pixmap_item.opacity()

        self.save_mask_clicked.emit(mask_pixmap, opacity)

    def on_cancel_mask(self):
        self.mask_canceled.emit()

    def on_delete_mask(self):
        self.mask_deleted.emit()
