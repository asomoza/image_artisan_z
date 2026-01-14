import logging

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from iartisanz.modules.generation.lora.lora_added_item import LoraAddedItem
from iartisanz.modules.generation.panels.base_panel import BasePanel


logger = logging.getLogger(__name__)


class LoraPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.event_bus.subscribe("lora", self.on_lora_event)
        self.event_bus.subscribe("lora_panel", self.on_lora_panel_event)

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_lora_button = QPushButton("Add LoRA")
        add_lora_button.clicked.connect(self.open_lora_dialog)
        main_layout.addWidget(add_lora_button)

        added_loras_widget = QWidget()
        self.loras_layout = QVBoxLayout(added_loras_widget)
        main_layout.addWidget(added_loras_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def open_lora_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "lora_manager", "action": "open"})

    def on_lora_event(self, data: dict):
        action = data.get("action")
        if action == "add":
            lora_widget = LoraAddedItem(data["lora"])

            for i in range(self.loras_layout.count()):
                item = self.loras_layout.itemAt(i).widget()
                if item.lora.id == data["lora"].id:
                    self.event_bus.publish("show_snackbar", {"action": "show", "message": "LoRA already added"})
                    return

            self.loras_layout.addWidget(lora_widget)
        elif action == "remove":
            for i in range(self.loras_layout.count()):
                item = self.loras_layout.itemAt(i).widget()
                if item.lora.id == data["lora"].id:
                    item.setParent(None)
                    break

    def clear_loras_layout(self):
        while self.loras_layout.count():
            child = self.loras_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_lora_panel_event(self, data):
        action = data.get("action")
        if action == "loras_updated":
            loaded_loras = data.get("loaded_loras")

            if loaded_loras is None:
                return

            self.clear_loras_layout()

            for lora_data_object in loaded_loras:
                lora_widget = LoraAddedItem(lora_data_object)
                self.loras_layout.addWidget(lora_widget)
