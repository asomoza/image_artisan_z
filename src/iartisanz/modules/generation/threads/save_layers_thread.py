from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING

from PyQt6.QtCore import QThread, pyqtSignal


if TYPE_CHECKING:
    from iartisanz.modules.generation.image.image_editor_layer import ImageEditorLayer


class SaveLayersThread(QThread):
    error = pyqtSignal(str)

    def __init__(self, layers: list[ImageEditorLayer], prefix: str = "layer_img", temp_path: str = "tmp/"):
        super().__init__()

        self.prefix = prefix
        self.temp_path = temp_path
        self.layers = layers

    def run(self):
        if self.layers is not None and len(self.layers) > 0:
            for layer in self.layers:
                if layer.image_path is None:
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    base_name = f"{self.prefix}_{timestamp}_{layer.layer_id}"
                    image_filename = f"{base_name}.png"
                    image_path = os.path.join(self.temp_path, image_filename)
                else:
                    image_path = layer.image_path

                layer.pixmap_item.pixmap().save(image_path)
                layer.image_path = image_path
