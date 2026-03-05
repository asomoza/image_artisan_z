from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from iartisanz.app.component_registry import ComponentRegistry
from iartisanz.app.directories import DirectoriesObject
from iartisanz.app.event_bus import EventBus
from iartisanz.modules.generation.constants import MODEL_TYPES
from iartisanz.modules.generation.data_objects.model_data_object import ModelDataObject
from iartisanz.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from iartisanz.modules.generation.widgets.model_item_widget import ModelItemWidget


class ModelInfoWidget(QWidget):
    model_edit = pyqtSignal(ModelItemDataObject, QPixmap)
    model_deleted = pyqtSignal(ModelItemWidget)
    generate_example = pyqtSignal(str)

    def __init__(self, model_item: ModelItemWidget, directories: DirectoriesObject):
        super().__init__()

        self.model_item = model_item
        self.json_graph = None
        self.pixmap = model_item.pixmap

        self.directories = directories
        self.event_bus = EventBus()

        self.init_ui()
        self.load_info()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(3, 0, 3, 0)

        self.model_image_label = QLabel()
        self.model_image_label.setFixedWidth(345)
        self.model_image_label.setMaximumHeight(480)
        self.model_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.model_image_label)

        self.model_name_label = QLabel(self.model_item.model_data.name)
        self.model_name_label.setObjectName("model_name")
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.model_name_label)

        model_type_version_layout = QGridLayout()

        self.model_type_layout = QHBoxLayout()
        self.model_type_title_label = QLabel("Type: ")
        self.model_type_title_label.setObjectName("model_type")
        self.model_type_layout.addWidget(self.model_type_title_label)
        self.model_type_label = QLabel()
        self.model_type_layout.addWidget(self.model_type_label)
        model_type_version_layout.addLayout(self.model_type_layout, 0, 0)

        self.version_layout = QHBoxLayout()
        self.version_layout.setSpacing(5)
        self.version_layout.addStretch()
        self.version_title_label = QLabel("Version")
        self.version_title_label.setObjectName("version_title")
        self.version_layout.addWidget(self.version_title_label)
        self.version_label = QLabel()
        self.version_layout.addWidget(self.version_label)
        model_type_version_layout.addLayout(self.version_layout, 0, 1)

        main_layout.addLayout(model_type_version_layout)
        main_layout.addSpacing(3)

        self.tags_title_label = QLabel("Tags:")
        self.tags_title_label.setObjectName("tags_title")
        self.tags_label = QLabel()
        self.tags_label.setObjectName("tags")
        main_layout.addWidget(self.tags_title_label)
        main_layout.addWidget(self.tags_label)

        self.components_title_label = QLabel("Components:")
        self.components_title_label.setObjectName("tags_title")
        self.components_title_label.setVisible(False)
        main_layout.addWidget(self.components_title_label)
        self.components_grid = QGridLayout()
        self.components_grid.setSpacing(2)
        main_layout.addLayout(self.components_grid)

        main_layout.addStretch()

        self.delete_model_button = QPushButton("Delete")
        self.delete_model_button.setObjectName("red_button")
        self.delete_model_button.clicked.connect(self.on_delete_clicked)
        main_layout.addWidget(self.delete_model_button)

        self.edit_model_button = QPushButton("Edit")
        self.edit_model_button.clicked.connect(self.on_edit_clicked)
        main_layout.addWidget(self.edit_model_button)

        self.generate_example_button = QPushButton("Generate example")
        self.generate_example_button.clicked.connect(self.on_generate_example)
        self.generate_example_button.setVisible(False)
        main_layout.addWidget(self.generate_example_button)

        self.select_model_button = QPushButton("Select model")
        self.select_model_button.setObjectName("green_button")
        self.select_model_button.clicked.connect(self.on_model_selected)
        main_layout.addWidget(self.select_model_button)

        self.setLayout(main_layout)

    def load_info(self):
        self.model_image_label.setPixmap(self.pixmap)

        model_type_string = "Z-Image Turbo"
        if self.model_item.model_data.model_type is not None:
            model_type_string = MODEL_TYPES[self.model_item.model_data.model_type]
        self.model_type_label.setText(model_type_string)

        self.version_label.setText(self.model_item.model_data.version)
        if self.model_item.model_data.version is None:
            self.version_title_label.setVisible(False)

        self.tags_label.setText(self.model_item.model_data.tags)
        if self.model_item.model_data.tags is None:
            self.tags_title_label.setVisible(False)

        if self.model_item.model_data.example is not None:
            self.json_graph = self.model_item.model_data.example
            self.generate_example_button.setVisible(True)

        self._load_components()

    def _load_components(self):
        model_id = self.model_item.model_data.id
        if model_id is None:
            return

        import os

        db_path = os.path.join(self.directories.data_path, "app.db")
        components_base_dir = os.path.join(self.directories.models_diffusers, "_components")
        registry = ComponentRegistry(db_path, components_base_dir)

        try:
            display_info = registry.get_component_display_info(model_id)
        except Exception:
            return

        if not display_info:
            return

        self.components_title_label.setVisible(True)
        for row, comp in enumerate(display_info):
            type_label = QLabel(comp["type"].replace("_", " "))
            type_label.setObjectName("tags")
            dtype_label = QLabel(comp["dtype_label"])
            dtype_label.setObjectName("tags")
            dtype_label.setAlignment(Qt.AlignmentFlag.AlignRight)
            self.components_grid.addWidget(type_label, row, 0)
            self.components_grid.addWidget(dtype_label, row, 1)

    def on_delete_clicked(self):
        confirm = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Are you sure you want to delete '{self.model_item.model_data.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.model_deleted.emit(self.model_item)

    def on_edit_clicked(self):
        self.model_edit.emit(self.model_item.model_data, self.model_item.pixmap)

    def on_generate_example(self):
        self.event_bus.publish("generate", {"action": "generate_from_json", "json_graph": self.json_graph})

    def on_model_selected(self):
        model_data_object = ModelDataObject(
            name=self.model_item.model_data.name,
            version=self.model_item.model_data.version,
            filepath=self.model_item.model_data.filepath,
            model_type=self.model_item.model_data.model_type,
            id=self.model_item.model_data.id,
        )
        self.event_bus.publish("model", {"action": "update", "model_data_object": model_data_object})
