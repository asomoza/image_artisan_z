from __future__ import annotations

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from superqt import QDoubleRangeSlider, QLabeledDoubleSlider

from iartisanz.app.event_bus import EventBus


class ControlNetAddedItem(QFrame):
    def __init__(self, controlnet_data: dict):
        super().__init__()

        self.event_bus = EventBus()
        self.controlnet_data = controlnet_data or {}

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        header_layout = QHBoxLayout()
        model_name = self.controlnet_data.get("controlnet_model_name") or "ControlNet"
        self.model_label = QLabel(f"Model: {model_name}")
        header_layout.addWidget(self.model_label, alignment=Qt.AlignmentFlag.AlignLeft)

        header_layout.addStretch(1)
        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.on_remove_clicked)
        header_layout.addWidget(remove_btn, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(header_layout)

        self.thumb_label = QLabel()
        thumb_path = self.controlnet_data.get("control_image_thumb_path")
        if thumb_path:
            pixmap = QPixmap(thumb_path)
            if not pixmap.isNull():
                pixmap = pixmap.scaled(
                    150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                )
                self.thumb_label.setPixmap(pixmap)
        main_layout.addWidget(self.thumb_label, alignment=Qt.AlignmentFlag.AlignCenter)

        conditioning_scale_layout = QVBoxLayout()
        conditioning_scale_label = QLabel("Conditioning scale")
        conditioning_scale_layout.addWidget(conditioning_scale_label)

        self.conditioning_scale_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_scale_slider.setRange(0.0, 2.0)
        self.conditioning_scale_slider.setSingleStep(0.05)
        self.conditioning_scale_slider.setValue(float(self.controlnet_data.get("conditioning_scale", 0.75)))
        self.conditioning_scale_slider.valueChanged.connect(self.on_conditioning_scale_changed)
        conditioning_scale_layout.addWidget(self.conditioning_scale_slider)
        main_layout.addLayout(conditioning_scale_layout)

        mode_layout = QVBoxLayout()
        mode_layout.addWidget(QLabel("Control mode"))
        self.control_mode_combo = QComboBox()
        self.control_mode_combo.addItem("Balanced", "balanced")
        self.control_mode_combo.addItem("Prompt", "prompt")
        self.control_mode_combo.addItem("ControlNet", "controlnet")
        self.control_mode_combo.currentIndexChanged.connect(self.on_control_mode_changed)
        mode_layout.addWidget(self.control_mode_combo)
        main_layout.addLayout(mode_layout)

        decay_layout = QVBoxLayout()
        decay_layout.addWidget(QLabel("Prompt mode decay"))
        self.prompt_decay_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.prompt_decay_slider.setRange(0.0, 1.0)
        self.prompt_decay_slider.setSingleStep(0.01)
        self.prompt_decay_slider.setValue(float(self.controlnet_data.get("controlnet_prompt_decay", 0.825)))
        self.prompt_decay_slider.valueChanged.connect(self.on_prompt_decay_changed)
        decay_layout.addWidget(self.prompt_decay_slider)
        self.prompt_decay_container = QWidget()
        self.prompt_decay_container.setLayout(decay_layout)
        main_layout.addWidget(self.prompt_decay_container)

        self._set_control_mode_ui(self.controlnet_data.get("controlnet_control_mode", "balanced"))

        guidance_layout = QHBoxLayout()
        self.guidance_start_label = QLabel("0%")
        guidance_layout.addWidget(self.guidance_start_label)

        self.guidance_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(0.0, 1.0)
        self.guidance_slider.setSingleStep(0.01)
        guidance_layout.addWidget(self.guidance_slider)

        self.guidance_end_label = QLabel("100%")
        guidance_layout.addWidget(self.guidance_end_label)
        main_layout.addLayout(guidance_layout)

        self._set_guidance_ui(self.controlnet_data.get("control_guidance_start_end", [0.0, 1.0]))
        self.guidance_slider.valueChanged.connect(self.on_guidance_changed)

        self.setLayout(main_layout)

    def _set_control_mode_ui(self, mode_value):
        mode = str(mode_value or "balanced").strip().lower()
        if mode not in {"balanced", "prompt", "controlnet"}:
            mode = "balanced"

        blocker = QSignalBlocker(self.control_mode_combo)
        try:
            for i in range(self.control_mode_combo.count()):
                if self.control_mode_combo.itemData(i) == mode:
                    self.control_mode_combo.setCurrentIndex(i)
                    break
        finally:
            del blocker

        self._update_prompt_decay_visibility(mode)

    def _set_guidance_ui(self, values):
        try:
            start, end = float(values[0]), float(values[1])
        except Exception:
            start, end = 0.0, 1.0

        self.guidance_start_label.setText(f"{int(start * 100)}%")
        self.guidance_end_label.setText(f"{int(end * 100)}%")

        blocker = QSignalBlocker(self.guidance_slider)
        try:
            self.guidance_slider.setValue((start, end))
        finally:
            del blocker

    def _update_prompt_decay_visibility(self, mode: str):
        if not hasattr(self, "prompt_decay_container"):
            return
        self.prompt_decay_container.setVisible(mode == "prompt")

    def on_conditioning_scale_changed(self, value: float):
        self.event_bus.publish(
            "controlnet",
            {"action": "update_conditioning_scale", "conditioning_scale": float(value)},
        )

    def on_guidance_changed(self, value: tuple):
        start, end = round(value[0], 2), round(value[1], 2)
        self.guidance_start_label.setText(f"{int(start * 100)}%")
        self.guidance_end_label.setText(f"{int(end * 100)}%")
        self.event_bus.publish(
            "controlnet",
            {"action": "update_control_guidance_start_end", "control_guidance_start_end": [start, end]},
        )

    def on_control_mode_changed(self, _index: int):
        mode = self.control_mode_combo.currentData()
        self._update_prompt_decay_visibility(str(mode))
        self.event_bus.publish(
            "controlnet",
            {"action": "update_control_mode", "controlnet_control_mode": str(mode)},
        )

    def on_prompt_decay_changed(self, value: float):
        try:
            v = float(value)
        except Exception:
            v = 0.825
        v = max(0.0, min(1.0, v))
        self.event_bus.publish(
            "controlnet",
            {"action": "update_prompt_decay", "controlnet_prompt_decay": v},
        )

    def on_remove_clicked(self):
        self.event_bus.publish("controlnet", {"action": "remove"})
