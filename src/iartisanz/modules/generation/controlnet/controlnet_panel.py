from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

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

        add_mask_button = QPushButton("Add Mask Image")
        add_mask_button.clicked.connect(self.open_controlnet_mask_dialog)
        main_layout.addWidget(add_mask_button)

        # Mask preview container
        self.mask_preview_container = QWidget()
        self.mask_preview_layout = QVBoxLayout(self.mask_preview_container)
        self.mask_preview_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.mask_preview_container)
        self.mask_preview_container.hide()

        main_layout.addStretch()
        self.setLayout(main_layout)

    def open_controlnet_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "controlnet", "action": "open"})

    def open_controlnet_mask_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "controlnet_mask", "action": "open"})

    def clear_controlnets_layout(self):
        while self.controlnets_layout.count():
            child = self.controlnets_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def clear_mask_preview(self):
        while self.mask_preview_layout.count():
            child = self.mask_preview_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.mask_preview_container.hide()

    def show_mask_preview(self, mask_thumb_path: str):
        self.clear_mask_preview()

        mask_frame = QFrame()
        mask_frame.setFrameStyle(QFrame.Shape.Box)
        mask_layout = QVBoxLayout()
        mask_layout.setContentsMargins(6, 6, 6, 6)
        mask_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        mask_label = QLabel("Mask:")
        header_layout.addWidget(mask_label, alignment=Qt.AlignmentFlag.AlignLeft)
        header_layout.addStretch(1)

        remove_mask_btn = QPushButton("Remove")
        remove_mask_btn.clicked.connect(self.on_remove_mask_clicked)
        header_layout.addWidget(remove_mask_btn, alignment=Qt.AlignmentFlag.AlignRight)
        mask_layout.addLayout(header_layout)

        thumb_label = QLabel()
        if mask_thumb_path:
            pixmap = QPixmap(mask_thumb_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                thumb_label.setPixmap(pixmap)
        mask_layout.addWidget(thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        mask_frame.setLayout(mask_layout)
        self.mask_preview_layout.addWidget(mask_frame)
        self.mask_preview_container.show()

    def on_remove_mask_clicked(self):
        self.event_bus.publish("controlnet", {"action": "remove_mask"})

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
        elif action in {"add_mask", "update_mask"}:
            mask_thumb_path = data.get("controlnet_mask_thumb_path")
            self.show_mask_preview(mask_thumb_path)
        elif action == "remove_mask":
            self.clear_mask_preview()
