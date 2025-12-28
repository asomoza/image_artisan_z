import logging
import os
from io import BytesIO

from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from iartisanz.utils.database import Database
from iartisanz.utils.model_utils import calculate_file_hash


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
        self.database_path = os.path.join(data_path, "app.db")
        self.database_table = database_table
        self.logger = logging.getLogger(__name__)

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
                self.logger.error(f"Directory not found: {directory['path']}")
                continue

            for filepath in os.listdir(directory["path"]):
                if directory["format"] == "diffusers":
                    model_directory = os.path.join(directory["path"], filepath)
                    total_items += 1
                    files_to_check.append(model_directory)
                elif filepath.endswith(".safetensors"):
                    total_items += 1
                    full_filepath = os.path.join(directory["path"], filepath)
                    files_to_check.append(full_filepath)

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

                self.status_changed.emit(f"Scanning {filepath}...")
                model_format = 0
                image_buffer = None
                replace = False

                if directory["format"] == "diffusers":
                    model_format = 1
                    model_directory = os.path.join(directory["path"], filepath)
                    full_filepath = os.path.join(model_directory, "unet", "diffusion_pytorch_model.fp16.safetensors")
                    root_filename = filepath
                    filepath = model_directory
                elif filepath.endswith(".safetensors"):
                    full_filepath = os.path.join(directory["path"], filepath)
                    filename = os.path.basename(full_filepath)
                    root_filename, _ = os.path.splitext(filename)
                    filepath = full_filepath

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
                    model_item = ModelItemDataObject(
                        root_filename=root_filename,
                        filepath=filepath,
                        name=(root_filename[:20] + "...") if len(root_filename) > 20 else root_filename,
                        version="1.0",
                        model_type=1,
                        model_format=model_format,
                        hash=hash,
                        deleted=0,
                    )

                    self.database.insert(self.database_table, model_item.to_dict())
                    model_item.id = self.database.last_insert_rowid()

                self.item_scanned.emit(model_item, image_buffer, replace)

                items_processed += 1
                self.scan_progress.emit(items_processed, total_items)

        self.database.disconnect()
        self.finished_scanning.emit()
