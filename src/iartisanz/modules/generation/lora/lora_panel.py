import logging
import os

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject
from iartisanz.modules.generation.lora.lora_added_item import LoraAddedItem
from iartisanz.modules.generation.panels.base_panel import BasePanel
from iartisanz.utils.database import Database


logger = logging.getLogger(__name__)


class LoraPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.event_bus.subscribe("lora", self.on_lora_event)
        self.event_bus.subscribe("json_graph", self.on_json_graph_event)

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
    def on_json_graph_event(self, data):
        action = data.get("action")
        if action == "loaded":
            data = data.get("data", {})
            loras = data.get("loras", [])

            if loras:
                self.clear_loras_layout()

                for lora_data in loras:
                    lora_database_id = lora_data.get("database_id", 0)

                    if lora_database_id != 0:
                        database = Database(os.path.join(self.directories.data_path, "app.db"))
                        lora_db_item = database.select_one(
                            "lora_model",
                            ["name", "version", "model_type", "root_filename", "filepath"],
                            {"id": lora_database_id},
                        )

                        if lora_db_item is None:
                            logger.debug(f"LoRA with id {lora_database_id} not found in database.")
                            continue

                        lora_data_object = LoraDataObject(
                            id=lora_data.get("id", 0),
                            name=lora_db_item["name"],
                            version=lora_db_item["version"],
                            type=lora_db_item["model_type"],
                            enabled=lora_data.get("enabled", True),
                            filename=lora_db_item["root_filename"],
                            path=lora_db_item["filepath"],
                        )
                        lora_widget = LoraAddedItem(lora_data_object)
                        self.loras_layout.addWidget(lora_widget)
                    else:
                        logger.debug("LoRA doesn't have a database id.")
