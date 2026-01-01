import os

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from superqt import QLabeledDoubleSlider

from iartisanz.modules.generation.panels.base_panel import BasePanel


class SourceImagePanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.source_image_thumb_path = None

        self.init_ui()

        self.event_bus.subscribe("source_image", self.on_source_image_event)
        self.event_bus.subscribe("json_graph", self.on_json_graph_event)

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.setEnabled(False)
        self.enabled_checkbox.toggled.connect(self.on_enabled_changed)
        main_layout.addWidget(self.enabled_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        strength_layout = QHBoxLayout()
        lbl_strength_text = QLabel("Strength")
        strength_layout.addWidget(lbl_strength_text, 1, Qt.AlignmentFlag.AlignLeft)

        self.strength_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0.01, 1.0)
        self.strength_slider.setValue(float(self.gen_settings.strength))
        self.strength_slider.setEnabled(False)
        self.strength_slider.valueChanged.connect(self.on_strength_changed)
        main_layout.addWidget(self.strength_slider)

        add_image_button = QPushButton("Set Image")
        add_image_button.clicked.connect(self.open_image_dialog)
        add_image_button.setObjectName("green_button")
        main_layout.addWidget(add_image_button)

        self.remove_image_button = QPushButton("Remove Image")
        self.remove_image_button.clicked.connect(self.on_remove_source_image)
        self.remove_image_button.setObjectName("red_button")
        self.remove_image_button.setEnabled(False)
        main_layout.addWidget(self.remove_image_button)

        source_thumb_layout = QVBoxLayout()
        self.image_text_label = QLabel("Image")
        self.image_text_label.setVisible(False)
        source_thumb_layout.addWidget(self.image_text_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.source_thumb_label = QLabel()
        source_thumb_layout.addWidget(self.source_thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.mask_text_label = QLabel("Mask")
        self.mask_text_label.setVisible(False)
        source_thumb_layout.addWidget(self.mask_text_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.source_thumb_mask_label = QLabel()
        source_thumb_layout.addWidget(self.source_thumb_mask_label, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(source_thumb_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def on_enabled_changed(self, checked: bool):
        self.event_bus.publish("source_image", {"action": "enable", "value": checked})

    def on_strength_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "strength", "value": value})

    def open_image_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "source_image", "action": "open"})

    def on_remove_source_image(self):
        self.image_text_label.setVisible(False)
        self.source_thumb_label.clear()
        self.enabled_checkbox.setEnabled(False)
        self.strength_slider.setEnabled(False)
        self.remove_image_button.setEnabled(False)

        blocker = QSignalBlocker(self.enabled_checkbox)
        try:
            self.enabled_checkbox.setChecked(False)
        finally:
            del blocker

        self.event_bus.publish("source_image", {"action": "remove"})

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_source_image_event(self, data: dict):
        action = data.get("action")
        new_source_image_thumb_path = data.get("source_thumb_path")

        if self.source_image_thumb_path is not None and self.directories.temp_path in self.source_image_thumb_path:
            os.remove(self.source_image_thumb_path)

        self.source_image_thumb_path = new_source_image_thumb_path

        if action == "add":
            self.image_text_label.setVisible(True)
            self.source_thumb_label.setPixmap(QPixmap(self.source_image_thumb_path))
            self.enabled_checkbox.setEnabled(True)
            self.strength_slider.setEnabled(True)
            self.remove_image_button.setEnabled(True)

            blocker = QSignalBlocker(self.enabled_checkbox)
            try:
                self.enabled_checkbox.setChecked(True)
            finally:
                del blocker
        elif action == "update":
            self.source_thumb_label.setPixmap(QPixmap(self.source_image_thumb_path))

    def on_json_graph_event(self, data):
        action = data.get("action")
        if action == "loaded":
            data = data.get("data", {})
            source_image_path = data.get("source_image", None)

            if source_image_path is not None:
                source_pixmap = QPixmap(source_image_path)
                source_thumb_pixmap = source_pixmap.scaled(150, 150)
                self.source_thumb_label.setPixmap(source_thumb_pixmap)
                self.image_text_label.setVisible(True)
                self.enabled_checkbox.setEnabled(True)
                self.remove_image_button.setEnabled(True)

                strength_blocker = QSignalBlocker(self.strength_slider)
                enabled_blocker = QSignalBlocker(self.enabled_checkbox)

                try:
                    self.strength_slider.setEnabled(True)
                    strength = data.get("strength", 1.0)
                    self.strength_slider.setValue(float(strength))
                    self.enabled_checkbox.setChecked(True)
                finally:
                    del strength_blocker
                    del enabled_blocker
