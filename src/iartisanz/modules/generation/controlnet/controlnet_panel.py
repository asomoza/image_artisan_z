from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from iartisanz.modules.generation.controlnet.controlnet_added_item import ControlNetAddedItem
from iartisanz.modules.generation.panels.base_panel import BasePanel


class ControlNetPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.event_bus.subscribe("controlnet", self.on_controlnet_event)

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_controlnet_button = QPushButton("Add Control Image")
        add_controlnet_button.clicked.connect(self.open_controlnet_dialog)
        main_layout.addWidget(add_controlnet_button)

        added_controlnets_widget = QWidget()
        self.controlnets_layout = QVBoxLayout(added_controlnets_widget)
        main_layout.addWidget(added_controlnets_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def open_controlnet_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "controlnet", "action": "open"})

    def clear_controlnets_layout(self):
        while self.controlnets_layout.count():
            child = self.controlnets_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_controlnet_event(self, data: dict):
        action = data.get("action")
        if action in {"add", "update"}:
            self.clear_controlnets_layout()
            controlnet_widget = ControlNetAddedItem(data)
            self.controlnets_layout.addWidget(controlnet_widget)
        elif action == "remove":
            self.clear_controlnets_layout()
