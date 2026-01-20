from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from iartisanz.modules.generation.panels.base_panel import BasePanel


class ControlNetPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

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
