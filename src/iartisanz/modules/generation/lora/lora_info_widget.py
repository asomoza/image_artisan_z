from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from iartisanz.app.event_bus import EventBus
from iartisanz.layouts.simple_flow_layout import SimpleFlowLayout
from iartisanz.modules.generation.constants import MODEL_TYPES
from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject
from iartisanz.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from iartisanz.modules.generation.widgets.model_item_widget import ModelItemWidget


class LoraInfoWidget(QWidget):
    lora_edit = pyqtSignal(ModelItemDataObject, QPixmap)
    lora_deleted = pyqtSignal(ModelItemWidget)
    generate_example = pyqtSignal(str)
    trigger_clicked = pyqtSignal(str)
    example_prompt_clicked = pyqtSignal(str)

    def __init__(self, model_item: ModelItemWidget):
        super().__init__()

        self.model_item = model_item
        self.json_graph = None
        self.pixmap = model_item.pixmap

        self.event_bus = EventBus()

        self.init_ui()
        self.load_info()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setSpacing(1)
        main_layout.setContentsMargins(3, 0, 3, 0)

        self.lora_image_label = QLabel()
        self.lora_image_label.setFixedWidth(345)
        self.lora_image_label.setMaximumHeight(480)
        self.lora_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.lora_image_label)

        self.lora_name_label = QLabel(self.model_item.model_data.name)
        self.lora_name_label.setObjectName("model_name")
        self.lora_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.lora_name_label)

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

        tags_layout = QHBoxLayout()
        self.tags_title_label = QLabel("Tags:")
        self.tags_title_label.setObjectName("tags_title")
        tags_layout.addWidget(self.tags_title_label)
        self.tags_label = QLabel()
        self.tags_label.setObjectName("tags")
        tags_layout.addWidget(self.tags_label, alignment=Qt.AlignmentFlag.AlignLeft)
        tags_layout.setStretch(0, 0)
        tags_layout.setStretch(1, 1)
        main_layout.addLayout(tags_layout)

        self.trigger_label = QLabel("Trigger words:")
        self.trigger_label.setObjectName("trigger_words")
        main_layout.addWidget(self.trigger_label)

        trigger_words_container = QWidget()
        self.triggers_layout = SimpleFlowLayout()
        self.triggers_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.triggers_layout.setSpacing(4)
        trigger_words_container.setLayout(self.triggers_layout)
        main_layout.addWidget(trigger_words_container)

        main_layout.addStretch()

        buttons_layout = QGridLayout()

        self.delete_lora_button = QPushButton("Delete")
        self.delete_lora_button.setObjectName("red_button")
        self.delete_lora_button.clicked.connect(self.on_delete_clicked)
        buttons_layout.addWidget(self.delete_lora_button, 0, 0)

        self.generate_example_button = QPushButton("Example")
        self.generate_example_button.clicked.connect(self.on_generate_example)
        self.generate_example_button.setVisible(False)
        buttons_layout.addWidget(self.generate_example_button, 0, 1)

        self.edit_lora_button = QPushButton("Edit")
        self.edit_lora_button.clicked.connect(self.on_edit_clicked)
        buttons_layout.addWidget(self.edit_lora_button, 1, 0)

        self.add_lora_button = QPushButton("Add LoRA")
        self.add_lora_button.setObjectName("green_button")
        self.add_lora_button.clicked.connect(self.on_lora_selected)
        buttons_layout.addWidget(self.add_lora_button, 1, 1)

        main_layout.addLayout(buttons_layout)

        self.setLayout(main_layout)

    def on_lora_selected(self):
        lora = LoraDataObject(
            id=self.model_item.model_data.id,
            name=self.model_item.model_data.name,
            version=self.model_item.model_data.version,
            type=self.model_item.model_data.model_type,
            enabled=True,
            filename=self.model_item.model_data.root_filename,
            path=self.model_item.model_data.filepath,
            lora_node_name=f"{self.model_item.model_data.name}_{self.model_item.model_data.version}_lora",
        )

        self.event_bus.publish("lora", {"action": "add", "lora": lora})

    def load_info(self):
        self.lora_image_label.setPixmap(self.pixmap)

        lora_type_string = "Z-Image Turbo"
        if self.model_item.model_data.model_type is not None:
            lora_type_string = MODEL_TYPES.get(self.model_item.model_data.model_type, "Unknown")
        self.model_type_label.setText(lora_type_string)

        self.version_label.setText(self.model_item.model_data.version)
        if self.model_item.model_data.version is None:
            self.version_title_label.setVisible(False)

        self.tags_label.setText(self.model_item.model_data.tags)
        if self.model_item.model_data.tags is None:
            self.tags_title_label.setVisible(False)

        if self.model_item.model_data.triggers is not None:
            triggers_list = [tag.strip() for tag in self.model_item.model_data.triggers.split(",")]

            for trigger in triggers_list:
                button = QPushButton(trigger)
                button.setObjectName("trigger_item")
                button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                button.setCursor(Qt.CursorShape.PointingHandCursor)
                button.clicked.connect(self.on_trigger_clicked)
                self.triggers_layout.addWidget(button)
        else:
            self.trigger_label.setVisible(False)

        if self.model_item.model_data.example is not None:
            self.json_graph = self.model_item.model_data.example
            self.generate_example_button.setVisible(True)

    def on_delete_clicked(self):
        confirm = QMessageBox.question(
            self,
            "Confirm deletion",
            f"Are you sure you want to delete '{self.model_item.model_data.name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if confirm == QMessageBox.StandardButton.Yes:
            self.lora_deleted.emit(self.model_item)

    def on_edit_clicked(self):
        self.lora_edit.emit(self.model_item.model_data, self.model_item.pixmap)

    def on_generate_example(self):
        self.event_bus.publish("generate", {"action": "generate_from_json", "json_graph": self.json_graph})

    def on_trigger_clicked(self):
        button = self.sender()
        self.trigger_clicked.emit(button.text())
        self.event_bus.publish("lora", {"action": "trigger_clicked", "trigger": button.text()})
