from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel
from superqt import QLabeledDoubleSlider

from iartisanz.app.base_simple_dialog import BaseSimpleDialog
from iartisanz.app.event_bus import EventBus


if TYPE_CHECKING:
    from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject


class LoraAdvancedDialog(BaseSimpleDialog):
    def __init__(self, lora: LoraDataObject):
        super().__init__("LoRA Advanced Dialog", minWidth=1000, minHeight=430)

        self.event_bus = EventBus()

        self.lora = lora
        self.low_range = 0.0
        self.high_range = 1.0

        self.init_ui()

    def init_ui(self):
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        sliders_layout = QGridLayout()

        self.unet_label = QLabel("Transformer: ")
        sliders_layout.addWidget(self.unet_label, 0, 0)
        self.transformer_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.transformer_weight_slider.setRange(self.low_range, self.high_range)
        self.transformer_weight_slider.setValue(self.lora.transformer_weight)
        self.transformer_weight_slider.valueChanged.connect(self.on_transformer_weight_changed)
        sliders_layout.addWidget(self.transformer_weight_slider, 0, 1)

        self.main_layout.addLayout(sliders_layout)

        self.main_layout.addStretch()

        self.setLayout(self.dialog_layout)

    def on_transformer_weight_changed(self, value: float):
        self.lora.transformer_weight = value
        self.event_bus.publish("lora", {"action": "update_weight", "lora": self.lora})
