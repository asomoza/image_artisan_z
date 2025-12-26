from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFrame, QHBoxLayout, QPushButton, QVBoxLayout

from iartisanz.app.event_bus import EventBus
from iartisanz.buttons.remove_button import RemoveButton
from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject


class LoraAddedItem(QFrame):
    remove_clicked = pyqtSignal(object)
    weight_changed = pyqtSignal()
    enabled = pyqtSignal(int, bool)
    advanced_clicked = pyqtSignal(object)

    def __init__(self, lora: LoraDataObject):
        super().__init__()

        self.event_bus = EventBus()

        self.lora = lora
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(3, 3, 3, 3)
        main_layout.setSpacing(1)
        upper_layout = QHBoxLayout()

        lora_name = self.lora.name
        if len(self.lora.version) > 0:
            lora_name = f"{self.lora.name} v{self.lora.version}"

        self.enabled_checkbox = QCheckBox(lora_name)
        self.enabled_checkbox.setChecked(self.lora.enabled)
        self.enabled_checkbox.stateChanged.connect(self.on_check_enabled)
        upper_layout.addWidget(self.enabled_checkbox)

        remove_button = RemoveButton()
        remove_button.setFixedSize(20, 20)
        remove_button.clicked.connect(self.on_removed)
        upper_layout.addWidget(remove_button)

        upper_layout.setStretch(0, 1)
        upper_layout.setStretch(1, 0)

        main_layout.addLayout(upper_layout)

        bottom_layout = QHBoxLayout()
        advanced_button = QPushButton("Advanced")
        advanced_button.clicked.connect(self.open_advanced)
        bottom_layout.addWidget(advanced_button, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(bottom_layout)

        self.setLayout(main_layout)

    def on_check_enabled(self):
        self.enabled.emit(self.lora.lora_id, self.enabled_checkbox.isChecked())

    def on_removed(self):
        self.event_bus.publish("lora", {"action": "remove", "lora": self.lora})

    def open_advanced(self):
        self.advanced_clicked.emit(self.lora)
