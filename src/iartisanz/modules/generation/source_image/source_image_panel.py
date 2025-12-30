from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout
from superqt import QLabeledDoubleSlider

from iartisanz.modules.generation.panels.base_panel import BasePanel


class SourceImagePanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.enabled_checkbox = QCheckBox("Enabled")
        self.enabled_checkbox.stateChanged.connect(self.on_enabled_changed)
        main_layout.addWidget(self.enabled_checkbox, alignment=Qt.AlignmentFlag.AlignCenter)

        strength_layout = QHBoxLayout()
        lbl_strength_text = QLabel("Strength")
        strength_layout.addWidget(lbl_strength_text, 1, Qt.AlignmentFlag.AlignLeft)

        self.lbl_strength = QLabel()
        strength_layout.addWidget(self.lbl_strength, 1, Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(strength_layout)

        self.strength_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0.01, 1.0)

        self.strength_slider.setValue(float(self.gen_settings.strength))
        self.lbl_strength.setText(f"{float(self.gen_settings.strength):0.2f}")

        self.strength_slider.valueChanged.connect(self.on_strength_changed)
        main_layout.addWidget(self.strength_slider)

        add_image_button = QPushButton("Set Image")
        add_image_button.clicked.connect(self.open_image_dialog)
        main_layout.addWidget(add_image_button)

        source_thumb_layout = QHBoxLayout()
        self.source_thumb_label = QLabel()
        source_thumb_layout.addWidget(self.source_thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)
        self.source_thumb_mask_label = QLabel()
        source_thumb_layout.addWidget(self.source_thumb_mask_label, alignment=Qt.AlignmentFlag.AlignCenter)
        main_layout.addLayout(source_thumb_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def on_enabled_changed(self, _state: int):
        pass

    def on_strength_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "strength", "value": value})

    def open_image_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "source_image", "action": "open"})
