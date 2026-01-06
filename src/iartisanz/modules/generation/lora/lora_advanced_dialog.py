from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout
from superqt import QLabeledDoubleSlider

from iartisanz.app.base_simple_dialog import BaseSimpleDialog
from iartisanz.app.event_bus import EventBus
from iartisanz.buttons.linked_button import LinkedButton


if TYPE_CHECKING:
    from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject


class LoraAdvancedDialog(BaseSimpleDialog):
    def __init__(self, dialog_key: str, lora: LoraDataObject):
        super().__init__("LoRA Advanced Dialog", minWidth=1100, minHeight=430)

        self.event_bus = EventBus()

        self.dialog_key = dialog_key

        self.lora = lora
        self.low_range = 0.0
        self.high_range = 1.0

        self.layer_sliders: dict[str, QLabeledDoubleSlider] = {}

        self.init_ui()

    def init_ui(self):
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.main_layout.setSpacing(5)

        sliders_layout = QGridLayout()

        self.transformer_label = QLabel("Transformer: ")
        sliders_layout.addWidget(self.transformer_label, 0, 0)
        self.transformer_weight_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.transformer_weight_slider.setRange(self.low_range, self.high_range)
        self.transformer_weight_slider.setValue(self.lora.transformer_weight)
        self.transformer_weight_slider.valueChanged.connect(self.on_transformer_weight_changed)
        sliders_layout.addWidget(self.transformer_weight_slider, 0, 1)

        self.main_layout.addLayout(sliders_layout)

        granular_layout = QHBoxLayout()
        granular_scales_checkbox = QCheckBox("Enable granular scales")
        granular_scales_checkbox.toggled.connect(self.on_granular)
        granular_layout.addWidget(granular_scales_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        self.slider_checkbox = QCheckBox("Slider")
        self.slider_checkbox.toggled.connect(self.on_is_slider_changed)
        self.slider_checkbox.setChecked(self.lora.is_slider)
        granular_layout.addWidget(self.slider_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        preset_layout = QHBoxLayout()

        template_label = QLabel("Preset:")
        preset_layout.addWidget(template_label)

        self.layer_templates_combo = QComboBox()
        self.layer_templates_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.layer_templates_combo.addItem("None")
        self.layer_templates_combo.addItem("All to 0.0")
        self.layer_templates_combo.addItem("All to 1.0")
        self.layer_templates_combo.currentIndexChanged.connect(self.on_layer_template_changed)
        preset_layout.addWidget(self.layer_templates_combo)
        granular_layout.addLayout(preset_layout)

        self.main_layout.addLayout(granular_layout)

        self.granular_frame = QFrame()
        self.granular_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.granular_frame.setEnabled(self.lora.granular_transformer_weights_enabled)

        sections_layout = QHBoxLayout()
        self._build_granular_layer_sliders(sections_layout)
        self.granular_frame.setLayout(sections_layout)
        self.main_layout.addWidget(self.granular_frame)

        self.setLayout(self.dialog_layout)

    def _build_granular_layer_sliders(self, sections_layout: QHBoxLayout) -> None:
        self.layer_sliders.clear()

        for idx, (layer_key, weight) in enumerate(self.lora.granular_transformer_weights.items()):
            layer_layout = QVBoxLayout()

            linked_button = LinkedButton()
            layer_layout.addWidget(linked_button, alignment=Qt.AlignmentFlag.AlignCenter)

            layer_slider = QLabeledDoubleSlider(Qt.Orientation.Vertical)
            layer_slider.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
            layer_slider.setRange(self.low_range, self.high_range)
            layer_slider.setValue(weight)
            layer_slider.valueChanged.connect(lambda v, k=layer_key: self.on_granular_layer_weight_changed(k, v))
            layer_layout.addWidget(layer_slider)

            layer_label = QLabel(f"L{idx + 1}")
            layer_layout.addWidget(layer_label, alignment=Qt.AlignmentFlag.AlignCenter)

            sections_layout.addLayout(layer_layout)
            self.layer_sliders[layer_key] = layer_slider

    def on_granular_layer_weight_changed(self, layer_key: str, value: float) -> None:
        self.lora.granular_transformer_weights[layer_key] = value
        self.event_bus.publish("lora", {"action": "update_weights", "lora": self.lora})

    def on_transformer_weight_changed(self, value: float):
        self.lora.transformer_weight = value
        self.event_bus.publish("lora", {"action": "update_weights", "lora": self.lora})

    def closeEvent(self, event):
        event.ignore()
        self.event_bus.publish(
            "manage_dialog",
            {"dialog_type": "lora_advanced", "action": "close", "dialog_key": self.dialog_key},
        )

    def on_granular(self, checked: bool):
        self.lora.granular_transformer_weights_enabled = checked
        self.granular_frame.setEnabled(self.lora.granular_transformer_weights_enabled)
        self.transformer_label.setDisabled(self.lora.granular_transformer_weights_enabled)
        self.transformer_weight_slider.setDisabled(self.lora.granular_transformer_weights_enabled)

        self.event_bus.publish("lora", {"action": "update_lora_transformer_granular_enabled", "lora": self.lora})

    def on_is_slider_changed(self, checked: bool):
        self.lora.is_slider = checked

        if checked:
            self.low_range = -30.0
            self.high_range = 30.0
        else:
            self.low_range = 0.0
            self.high_range = 1.0

        self.transformer_weight_slider.setRange(self.low_range, self.high_range)

        for slider in self.layer_sliders.values():
            slider.setRange(self.low_range, self.high_range)

        self.event_bus.publish("lora", {"action": "update_slider", "is_slider": self.lora.is_slider})

    def on_layer_template_changed(self, index: int):
        if index == 0:
            return
        elif index == 1:
            for layer_key in self.lora.granular_transformer_weights.keys():
                self.lora.granular_transformer_weights[layer_key] = 0.0
                self.layer_sliders[layer_key].setValue(0.0)
        elif index == 2:
            for layer_key in self.lora.granular_transformer_weights.keys():
                self.lora.granular_transformer_weights[layer_key] = 1.0
                self.layer_sliders[layer_key].setValue(1.0)

        self.event_bus.publish("lora", {"action": "update_weights", "lora": self.lora})
