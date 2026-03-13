import logging
import os
from io import BytesIO

from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from iartisanz.utils.database import Database
from iartisanz.utils.model_utils import calculate_file_hash


logger = logging.getLogger(__name__)


class ModelItemsScannerThread(QThread):
    status_changed = pyqtSignal(str)
    item_scanned = pyqtSignal(ModelItemDataObject, object, bool)
    item_deleted = pyqtSignal(int)
    scan_progress = pyqtSignal(int, int)
    finished_scanning = pyqtSignal()
    stop_requested = False

    def __init__(self, model_directories: tuple, image_dir: str, data_path: str, database_table: str):
        super().__init__()

        self.model_directories = model_directories
        self.image_dir = image_dir
        self.data_path = data_path
        self.database_path = os.path.join(data_path, "app.db")
        self.database_table = database_table

    def stop(self):
        self.stop_requested = True

    def run(self):
        self.database = Database(self.database_path)
        self.stop_requested = False
        self.status_changed.emit("Starting scan...")

        total_items = 0
        items_processed = 0

        files_to_check: list[str] = []

        columns = ["id", "filepath", "deleted"]
        all_items = self.database.select(self.database_table, columns)
        model_items = {item[1]: {"id": item[0], "deleted": item[2]} for item in all_items}

        for directory in self.model_directories:
            if not os.path.exists(directory["path"]):
                logger.error(f"Directory not found: {directory['path']}")
                continue

            for filepath in os.listdir(directory["path"]):
                # Skip the _components directory used for deduplicated storage
                if filepath == "_components":
                    continue

                model_directory = os.path.join(directory["path"], filepath)
                if not os.path.isdir(model_directory):
                    continue
                total_items += 1
                files_to_check.append(model_directory)

        self.scan_progress.emit(items_processed, total_items)

        # Check for deleted models
        for key in model_items.keys():
            if key not in files_to_check and model_items[key]["deleted"] == 0:
                self.database.update(self.database_table, {"deleted": 1}, {"id": model_items[key]["id"]})
                self.item_deleted.emit(model_items[key]["id"])

        for directory in self.model_directories:
            for filepath in os.listdir(directory["path"]):
                if self.stop_requested:
                    break

                # Skip the _components directory
                if filepath == "_components":
                    continue

                self.status_changed.emit(f"Scanning {filepath}...")
                image_buffer = None
                replace = False

                model_directory = os.path.join(directory["path"], filepath)
                if not os.path.isdir(model_directory):
                    continue
                full_filepath = self._find_hash_file(model_directory)
                root_filename = filepath
                filepath = model_directory

                if full_filepath is None:
                    logger.warning("No hashable file found in %s, skipping", model_directory)
                    items_processed += 1
                    self.scan_progress.emit(items_processed, total_items)
                    continue

                hash = calculate_file_hash(full_filepath)
                columns = ModelItemDataObject.get_column_names()
                existing_item = self.database.select_one(self.database_table, columns=columns, where={"hash": hash})

                if existing_item:
                    model_item = ModelItemDataObject(**existing_item)

                    if model_item.deleted == 1:
                        model_item.deleted = 0
                        self.database.update(self.database_table, {"deleted": 0, "filepath": filepath}, {"hash": hash})
                    else:
                        replace = True
                        self.database.update(self.database_table, {"filepath": filepath}, {"hash": hash})

                    if model_item.thumbnail is not None and len(model_item.thumbnail) > 0:
                        image_path = os.path.join(self.image_dir, f"{hash}.webp")

                        if os.path.exists(image_path):
                            with open(image_path, "rb") as image_file:
                                img_bytes = image_file.read()
                            image_buffer = BytesIO(img_bytes)
                else:
                    model_type, distilled = self._detect_model_type(filepath)
                    model_item = ModelItemDataObject(
                        root_filename=root_filename,
                        filepath=filepath,
                        name=(root_filename[:20] + "...") if len(root_filename) > 20 else root_filename,
                        version="1.0",
                        model_type=model_type,
                        hash=hash,
                        deleted=0,
                        distilled=int(distilled),
                    )

                    self.database.insert(self.database_table, model_item.to_dict())
                    model_item.id = self.database.last_insert_rowid()

                # Populate component registry
                if model_item.id is not None:
                    self._register_components(model_item.id, filepath)

                self.item_scanned.emit(model_item, image_buffer, replace)

                items_processed += 1
                self.scan_progress.emit(items_processed, total_items)

        self.database.disconnect()
        self.finished_scanning.emit()

    @staticmethod
    def _find_hash_file(model_directory: str) -> str | None:
        """Find the first .safetensors file in the transformer (or unet) subdirectory.

        Tries ``transformer/`` first, then ``unet/`` for legacy layouts.
        Returns the full path, or *None* if nothing is found.
        """
        for subdir in ("transformer", "unet"):
            comp_dir = os.path.join(model_directory, subdir)
            if not os.path.isdir(comp_dir):
                continue
            for fname in sorted(os.listdir(comp_dir)):
                if fname.endswith(".safetensors"):
                    return os.path.join(comp_dir, fname)
        return None

    @staticmethod
    def _detect_model_type(model_path: str) -> tuple[int, bool]:
        """Detect model type and distilled flag from transformer config.json and directory name.

        Returns:
            (model_type, distilled) tuple. model_type is one of:
            1=Z-Image Turbo, 3=Flux.2 Klein 9B, 5=Flux.2 Klein 4B, 7=Flux.2 Dev.
            distilled is True for distilled variants, False for base.
        """
        import json

        config_path = os.path.join(model_path, "transformer", "config.json")
        try:
            with open(config_path) as f:
                config = json.load(f)
        except Exception:
            return 1, True

        class_name = config.get("_class_name", "")
        if "Flux2" not in class_name:
            return 1, True

        dir_name = os.path.basename(model_path).lower()

        # Flux.2 Dev: detected by directory name or config heuristic.
        # Dev has 48 single-stream layers; Klein 9B has 24, Klein 4B has 20.
        num_single_layers = config.get("num_single_layers", 0)
        if "dev" in dir_name or num_single_layers >= 48:
            return 7, True

        # Determine 9B vs 4B from directory name, falling back to config heuristic.
        if "4b" in dir_name:
            is_4b = True
        elif "9b" in dir_name:
            is_4b = False
        else:
            # Heuristic: 9B default uses 48 attention heads; 4B uses fewer.
            num_heads = config.get("num_attention_heads", 48)
            is_4b = num_heads < 48

        # Determine distilled vs base from directory name.
        distilled = "base" not in dir_name

        model_type = 5 if is_4b else 3
        return model_type, distilled

    def _register_components(self, model_id: int, model_path: str) -> None:
        """Register component entries for a diffusers model if not already present."""
        try:
            from iartisanz.app.component_registry import COMPONENT_TYPES, ComponentRegistry
            from iartisanz.utils.model_utils import calculate_component_hash

            # Determine diffusers base dir from model path
            diffusers_dir = os.path.dirname(model_path)
            components_base_dir = os.path.join(diffusers_dir, "_components")
            registry = ComponentRegistry(self.database_path, components_base_dir)

            if registry.model_has_components(model_id):
                return

            component_mapping: dict[str, int] = {}
            for comp_type in COMPONENT_TYPES:
                comp_dir = os.path.join(model_path, comp_type)
                if not os.path.isdir(comp_dir):
                    continue
                content_hash = calculate_component_hash(comp_dir)
                comp_info = registry.register_component(
                    component_type=comp_type,
                    source_path=comp_dir,
                    content_hash=content_hash,
                )
                component_mapping[comp_type] = comp_info.id

            if component_mapping:
                registry.register_model_components(model_id, component_mapping)
                registry.cleanup_after_registration(model_id, model_path)
        except Exception as e:
            logger.debug("Failed to register components for model %d: %s", model_id, e)
