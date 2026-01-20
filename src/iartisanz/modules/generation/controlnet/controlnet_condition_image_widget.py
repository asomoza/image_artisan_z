from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from iartisanz.modules.generation.image.image_widget import ImageWidget


if TYPE_CHECKING:
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer
    from iartisanz.modules.generation.widgets.image_viewer_simple_widget import ImageViewerSimpleWidget


class ControlNetConditionImageWidget(QWidget):
    def __init__(
        self,
        image_viewer: ImageViewerSimpleWidget,
        image_width: int,
        image_height: int,
        outputh_path: str,
        temp_path: str,
        layers: list[ImageEditorLayer] = None,
        delete_original_on_load: bool = False,
    ):
        super().__init__()

        self.image_viewer = image_viewer
        self.image_width = image_width
        self.image_height = image_height
        self.outputh_path = outputh_path
        self.temp_path = temp_path
        self.delete_original_on_load = delete_original_on_load

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
            "Processed image",
            "cn_condition_image",
            self.image_viewer,
            self.image_width,
            self.image_height,
            show_layer_manager=True,
            layer_manager_to_right=True,
            save_directory=self.outputh_path,
            temp_path=self.temp_path,
        )
        main_layout.addWidget(self.image_widget)

        self.add_button = QPushButton("Add")
        self.add_button.setObjectName("green_button")
        self.add_button.setEnabled(False)
        main_layout.addWidget(self.add_button)

        self.setLayout(main_layout)
