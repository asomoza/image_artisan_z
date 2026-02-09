from __future__ import annotations

import logging
import os
import shutil
from importlib.resources import files
from typing import Optional, cast

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QSizePolicy, QVBoxLayout

from iartisanz.app.base_dialog import BaseDialog
from iartisanz.modules.generation.data_objects.model_data_object import ModelDataObject
from iartisanz.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from iartisanz.modules.generation.dialogs.model_items_view import ModelItemsView
from iartisanz.modules.generation.model.model_edit_widget import ModelEditWidget
from iartisanz.modules.generation.model.model_info_widget import ModelInfoWidget
from iartisanz.modules.generation.widgets.model_item_widget import ModelItemWidget


logger = logging.getLogger(__name__)


class ModelManagerDialog(BaseDialog):
    MODEL_IMG = str(files("iartisanz.theme.images").joinpath("model.webp"))

    def __init__(self, *args):
        if len(args) <= 3:
            logger.warning("ModelManagerDialog requires the image viewer argument to be able to set images.")
            self.image_viewer = None
            super().__init__(*args)
        else:
            self.image_viewer = args[3]
            super().__init__(*args[:3], *args[4:])

        self.setWindowTitle("Model Manager")
        self.setMinimumSize(1160, 950)

        self.loading_models = False
        self.selected_model = None

        self.settings = QSettings("ZCode", "ImageArtisanZ")
        self.settings.beginGroup("lora_manager_dialog")
        self.load_settings()

        diffusers_directory = {
            "path": self.directories.models_diffusers,
            "format": "diffusers",
        }

        singlefile_directory = {
            "path": self.directories.models_singlefile,
            "format": "singlefile",
        }
        self.model_directories = (diffusers_directory, singlefile_directory)
        self.database_table = "model"

        image_dir = os.path.join(self.directories.data_path, "models")

        if not os.path.exists(image_dir):
            try:
                os.makedirs(image_dir)
                logger.info(f"Directory '{image_dir}' created successfully.")
            except OSError as e:
                logger.error(f"Error creating directory '{image_dir}': {e}")

        self.image_dir = image_dir
        self.default_pixmap = QPixmap(self.MODEL_IMG)

        self.init_ui()

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def init_ui(self):
        content_layout = QHBoxLayout()

        self.model_items_view = ModelItemsView(
            self.directories,
            self.preferences,
            self.model_directories,
            self.default_pixmap,
            self.image_dir,
            self.database_table,
        )
        self.model_items_view.error.connect(self.show_error)
        self.model_items_view.model_item_clicked.connect(self.on_model_item_clicked)
        self.model_items_view.item_imported.connect(self.on_model_imported)
        self.model_items_view.finished_loading.connect(self.on_finished_loading_models)
        content_layout.addWidget(self.model_items_view)

        model_frame = QFrame()
        self.model_frame_layout = QVBoxLayout()
        model_frame.setLayout(self.model_frame_layout)
        model_frame.setFixedWidth(350)
        model_frame.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(model_frame)

        self.main_layout.addLayout(content_layout)

    def show_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})

    def on_finished_loading_models(self):
        self.loading_models = False

    def on_model_imported(self, path: str):
        if path.endswith(".safetensors"):
            if self._is_transformer_safetensors(path):
                self._import_transformer_only(path)
            else:
                self._import_safetensors_model(path)
        elif os.path.isdir(path) and os.path.exists(os.path.join(path, "model_index.json")):
            self._import_diffusers_model(path)

    def _is_transformer_safetensors(self, path: str) -> bool:
        """Check if a .safetensors file is a standalone transformer."""
        try:
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
            # ZImageTransformer2DModel signature keys
            return bool(keys & {"context_refiner.weight", "all_final_layer.adaLN_modulation.1.weight"})
        except Exception:
            return False

    def _import_transformer_only(self, path: str):
        """Import a standalone transformer .safetensors and auto-detect compatible components."""
        from iartisanz.app.app import get_app_database_path
        from iartisanz.app.component_registry import ComponentRegistry
        from iartisanz.modules.generation.component_compatibility import detect_transformer_architecture

        db_path = get_app_database_path()
        if db_path is None:
            self.show_error("Database not available for transformer import.")
            return

        components_base_dir = os.path.join(self.directories.models_diffusers, "_components")
        registry = ComponentRegistry(db_path, components_base_dir)

        architecture = detect_transformer_architecture(path)
        if architecture is None:
            self.show_error("Could not detect transformer architecture from safetensors file.")
            return

        compatible = registry.find_compatible_components(architecture)
        if not all(comp_type in compatible for comp_type in ("text_encoder", "vae", "tokenizer")):
            self.show_error(
                "Cannot import transformer: no compatible text_encoder, VAE, or tokenizer found. "
                "Import a full diffusers model first."
            )
            return

        # Copy transformer to _components
        os.makedirs(components_base_dir, exist_ok=True)
        transformer_dir = os.path.join(components_base_dir, "transformer")
        os.makedirs(transformer_dir, exist_ok=True)

        from iartisanz.utils.model_utils import calculate_file_hash_xxhash

        content_hash = calculate_file_hash_xxhash(path)
        dest_dir = os.path.join(transformer_dir, content_hash)

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)
            file_name = os.path.basename(path)
            dest_file = os.path.join(dest_dir, file_name)
            if self.preferences.delete_model_on_import:
                shutil.move(path, dest_file)
            else:
                shutil.copy2(path, dest_file)

            # Copy config.json from an existing compatible transformer
            existing_transformers = registry.find_compatible_components(architecture).get("text_encoder", [])
            # Actually, we need a transformer config. Find one from existing transformer components.
            from iartisanz.utils.database import Database

            db = Database(db_path)
            rows = db.fetch_all(
                "SELECT storage_path FROM component WHERE component_type = 'transformer' AND architecture = ?",
                (architecture,),
            )
            config_copied = False
            for (existing_path,) in rows:
                existing_config = os.path.join(existing_path, "config.json")
                if os.path.isfile(existing_config):
                    shutil.copy2(existing_config, os.path.join(dest_dir, "config.json"))
                    config_copied = True
                    break

            if not config_copied:
                logger.warning("No config.json found for architecture %s", architecture)

        comp_info = registry.register_component(
            component_type="transformer",
            source_path=dest_dir,
            content_hash=content_hash,
            architecture=architecture,
        )

        # Create model entry and link components
        model_name = os.path.splitext(os.path.basename(path))[0]
        if len(model_name) > 20:
            display_name = model_name[:20] + "..."
        else:
            display_name = model_name

        # Use first compatible component of each type
        component_mapping = {"transformer": comp_info.id}
        for comp_type in ("text_encoder", "vae", "tokenizer"):
            component_mapping[comp_type] = compatible[comp_type][0].id

        # Create the model in the DB and register components
        self.model_items_view.add_single_item_from_path(
            dest_dir, display_name, 1, component_mapping=component_mapping
        )

    def _import_safetensors_model(self, path: str):
        file_name = os.path.basename(path)
        target_dir = self.directories.models_singlefile
        new_path = self._get_unique_path(target_dir, file_name)

        if self.preferences.delete_lora_on_import:
            shutil.move(path, new_path)
        else:
            shutil.copy2(path, new_path)

        self.model_items_view.add_single_item_from_path(new_path, os.path.basename(new_path), 0)

    def _import_diffusers_model(self, path: str):
        from iartisanz.app.app import get_app_database_path
        from iartisanz.app.component_registry import COMPONENT_TYPES, ComponentRegistry
        from iartisanz.utils.model_utils import calculate_component_hash

        model_dir = os.path.basename(path)
        target_dir = self.directories.models_diffusers
        new_path = self._get_unique_path(target_dir, model_dir)

        if self.preferences.delete_model_on_import:
            shutil.move(path, new_path)
        else:
            shutil.copytree(path, new_path)

        # Register components in the registry
        db_path = get_app_database_path()
        component_mapping = None

        if db_path is not None:
            try:
                components_base_dir = os.path.join(self.directories.models_diffusers, "_components")
                registry = ComponentRegistry(db_path, components_base_dir)
                component_mapping = {}

                for comp_type in COMPONENT_TYPES:
                    comp_dir = os.path.join(new_path, comp_type)
                    if not os.path.isdir(comp_dir):
                        continue
                    content_hash = calculate_component_hash(comp_dir)
                    comp_info = registry.register_component(
                        component_type=comp_type,
                        source_path=comp_dir,
                        content_hash=content_hash,
                    )
                    component_mapping[comp_type] = comp_info.id
            except Exception as e:
                logger.error("Failed to register components during import: %s", e)
                component_mapping = None

        self.model_items_view.add_single_item_from_path(
            new_path, os.path.basename(new_path), 1, component_mapping=component_mapping
        )

    def _get_unique_path(self, target_dir: str, name: str) -> str:
        new_path = os.path.join(target_dir, name)

        if not os.path.exists(new_path):
            return new_path

        base, ext = os.path.splitext(name)
        counter = 1

        while True:
            unique_name = f"{base}_{counter}{ext}"
            new_path = os.path.join(target_dir, unique_name)

            if not os.path.exists(new_path):
                return new_path

            counter += 1

    def clear_selected_model(self):
        self.selected_model = None

        for i in reversed(range(self.model_frame_layout.count())):
            widget_to_remove = self.model_frame_layout.itemAt(i).widget()
            self.model_frame_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)

    def on_model_item_clicked(self, model_item_widget: ModelItemWidget):
        self.clear_selected_model()

        self.selected_model = ModelDataObject(
            name=model_item_widget.model_data.name,
            version=model_item_widget.model_data.version,
            model_format=model_item_widget.model_data.model_format,
            filepath=model_item_widget.model_data.filepath,
            model_type=model_item_widget.model_data.model_type,
            id=model_item_widget.model_data.id,
        )

        model_info_widget = ModelInfoWidget(model_item_widget, self.directories)
        model_info_widget.model_edit.connect(self.on_model_edit_clicked)
        model_info_widget.model_deleted.connect(self.on_model_deleted)
        self.model_frame_layout.addWidget(model_info_widget)

    def on_model_edit_clicked(self, model_data: ModelItemDataObject, pixmap: QPixmap):
        self.clear_selected_model()

        model_edit_widget = ModelEditWidget(self.directories, model_data, pixmap, self.image_viewer)
        model_edit_widget.model_info_saved.connect(self.on_info_saved)
        self.model_frame_layout.addWidget(model_edit_widget)

    def on_info_saved(self, model_data: ModelItemDataObject, pixmap: Optional[QPixmap]):
        model_items = self.model_items_view.flow_layout.items()
        edited_item = None

        for i, model in enumerate(model_items):
            model_item = cast(ModelItemWidget, model.widget())
            if model_item.model_data.id == model_data.id:
                edited_item = self.model_items_view.flow_layout.itemAt(i)
                break

        if edited_item is not None:
            edited_item = cast(ModelItemWidget, edited_item.widget())
            edited_item.model_data = model_data
            edited_item.image_widget.name_label.setText(model_data.name)
            edited_item.image_widget.set_model_version(model_data.version)
            edited_item.image_widget.set_model_type(model_data.model_type)

            if pixmap is not None:
                edited_item.update_model_image(pixmap)

        self.on_model_item_clicked(edited_item)

    def on_model_deleted(self, model_item_widget: ModelItemWidget):
        self.model_items_view.on_delete_item(model_item_widget)
