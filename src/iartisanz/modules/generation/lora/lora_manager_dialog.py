import logging
import os
import shutil
from importlib.resources import files
from typing import Optional, cast

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout

from iartisanz.app.base_dialog import BaseDialog
from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject
from iartisanz.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from iartisanz.modules.generation.dialogs.model_items_view import ModelItemsView
from iartisanz.modules.generation.lora.lora_edit_widget import LoraEditWidget
from iartisanz.modules.generation.lora.lora_info_widget import LoraInfoWidget
from iartisanz.modules.generation.widgets.model_item_widget import ModelItemWidget


class LoraManagerDialog(BaseDialog):
    LORA_IMG = str(files("iartisanz.theme.images").joinpath("lora.webp"))

    def __init__(self, *args):
        super().__init__(*args)

        self.setWindowTitle("LoRA Manager")
        self.setMinimumSize(1160, 800)

        self.logger = logging.getLogger(__name__)

        if len(args) <= 3:
            self.logger.warning("LoraManagerDialog requires the image viewer argument to be able to set images.")

        self.image_viewer = args[3] if len(args) > 3 else None

        self.loading_loras = False
        self.selected_lora = None

        self.settings = QSettings("ZCode", "ImageArtisanZ")
        self.settings.beginGroup("lora_manager_dialog")
        self.load_settings()

        loras_directory = {
            "path": self.directories.models_loras,
            "format": "safetensors",
        }
        self.loras_directories = (loras_directory,)
        self.database_table = "lora_model"
        image_dir = os.path.join(self.directories.data_path, "loras")

        if not os.path.exists(image_dir):
            try:
                os.makedirs(image_dir)
                self.logger.info(f"Directory '{image_dir}' created successfully.")
            except OSError as e:
                self.logger.error(f"Error creating directory '{image_dir}': {e}")

        self.image_dir = image_dir
        self.default_pixmap = QPixmap(self.LORA_IMG)

        self.init_ui()

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())

    def init_ui(self):
        content_layout = QHBoxLayout()

        self.lora_items_view = ModelItemsView(
            self.directories,
            self.preferences,
            self.loras_directories,
            self.default_pixmap,
            self.image_dir,
            self.database_table,
        )
        self.lora_items_view.error.connect(self.show_error)
        self.lora_items_view.item_imported.connect(self.on_lora_imported)
        self.lora_items_view.model_item_clicked.connect(self.on_lora_item_clicked)
        self.lora_items_view.finished_loading.connect(self.on_finished_loading_loras)
        content_layout.addWidget(self.lora_items_view)

        lora_frame = QFrame()
        self.lora_frame_layout = QVBoxLayout()
        lora_frame.setLayout(self.lora_frame_layout)
        lora_frame.setFixedWidth(350)
        lora_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(lora_frame)

        self.main_layout.addLayout(content_layout)

    def show_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})

    def on_lora_imported(self, path: str):
        if path.endswith(".safetensors"):
            file_name = os.path.basename(path)
            model_new_path = os.path.join(self.directories.models_loras, file_name)

            if os.path.exists(model_new_path):
                base, ext = os.path.splitext(file_name)
                counter = 1

                while True:
                    file_name = f"{base}_{counter}{ext}"
                    model_new_path = os.path.join(self.directories.models_loras, file_name)

                    if not os.path.exists(model_new_path):
                        break

                    counter += 1

            if self.preferences.delete_lora_on_import:
                shutil.move(path, model_new_path)
            else:
                shutil.copy2(path, model_new_path)

            self.lora_items_view.add_single_item_from_path(model_new_path, file_name, 0)

    def on_finished_loading_loras(self):
        self.loading_loras = False

    def on_lora_item_clicked(self, model_item_widget: ModelItemWidget):
        self.clear_selected_lora()

        self.selected_lora = LoraDataObject(
            id=model_item_widget.model_data.id,
            name=model_item_widget.model_data.name,
            version=model_item_widget.model_data.version,
            type=model_item_widget.model_data.model_type,
            enabled=True,
            filename=model_item_widget.model_data.root_filename,
            path=model_item_widget.model_data.filepath,
        )

        lora_info_widget = LoraInfoWidget(model_item_widget, self.directories)
        lora_info_widget.lora_edit.connect(self.on_lora_edit_clicked)
        lora_info_widget.lora_deleted.connect(self.on_lora_deleted)
        lora_info_widget.trigger_clicked.connect(self.on_trigger_clicked)
        self.lora_frame_layout.addWidget(lora_info_widget)

    def on_lora_edit_clicked(self, model_data: ModelItemDataObject, pixmap: QPixmap):
        self.clear_selected_lora()

        lora_edit_widget = LoraEditWidget(self.directories, model_data, pixmap, self.image_viewer)
        lora_edit_widget.lora_info_saved.connect(self.on_info_saved)
        self.lora_frame_layout.addWidget(lora_edit_widget)

    def on_info_saved(self, model_data: ModelItemDataObject, pixmap: Optional[QPixmap]):
        lora_items = self.lora_items_view.flow_layout.items()
        edited_item = None

        for i, lora in enumerate(lora_items):
            lora_item = cast(ModelItemWidget, lora.widget())
            if lora_item.model_data.id == model_data.id:
                edited_item = self.lora_items_view.flow_layout.itemAt(i)
                break

        if edited_item is not None:
            edited_item = cast(ModelItemWidget, edited_item.widget())
            edited_item.model_data = model_data
            edited_item.image_widget.name_label.setText(model_data.name)
            edited_item.image_widget.set_model_version(model_data.version)
            edited_item.image_widget.set_model_type(model_data.model_type)

            if pixmap is not None:
                edited_item.update_model_image(pixmap)

        self.on_lora_item_clicked(edited_item)

    def on_lora_deleted(self, model_item: ModelItemWidget):
        self.lora_items_view.on_delete_item(model_item)

    def clear_selected_lora(self):
        self.selected_lora = None

        for i in reversed(range(self.lora_frame_layout.count())):
            widget_to_remove = self.lora_frame_layout.itemAt(i).widget()
            self.lora_frame_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

    def on_trigger_clicked(self, trigger):
        self.event_bus.publish("insert_text_at_cursor", {"value": trigger})
