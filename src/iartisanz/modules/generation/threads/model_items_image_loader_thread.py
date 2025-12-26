import logging
import time
from io import BytesIO

from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.layouts.flow_layout import FlowLayout
from iartisanz.modules.generation.widgets.model_item_widget import ModelItemWidget


class ModelItemsImageLoaderThread(QThread):
    finished_loading = pyqtSignal()
    image_loaded = pyqtSignal(int, BytesIO)

    def __init__(self, flow_layout: FlowLayout, model_items: list):
        super().__init__()

        self.flow_layout = flow_layout
        self.model_items = model_items
        self.logger = logging.getLogger(__name__)
        self.stop_requested = False

    def stop(self):
        self.stop_requested = True

    def run(self):
        self.stop_requested = False

        for model_item in self.model_items:
            if self.stop_requested:
                break

            if isinstance(model_item, ModelItemWidget):
                image_path = model_item.model_data.thumbnail

                if image_path is not None:
                    try:
                        buffer = None
                        with open(image_path, "rb") as f:
                            buffer = BytesIO(f.read())

                        model_index = self.flow_layout.index_of(model_item)
                        self.image_loaded.emit(model_index, buffer)
                        time.sleep(0.03)
                    except Exception as e:
                        self.logger.error(f"Error loading image: {e}")

        self.finished_loading.emit()
