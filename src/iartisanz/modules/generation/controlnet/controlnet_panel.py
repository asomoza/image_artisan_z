from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
from superqt import QDoubleRangeSlider, QLabeledDoubleSlider

from iartisanz.modules.generation.panels.base_panel import BasePanel


class ControlNetPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.controlnet_data = {}
        self.init_ui()

        self.event_bus.subscribe("controlnet", self.on_controlnet_event)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # Model selection section
        model_section = self._create_model_section()
        main_layout.addWidget(model_section)

        # Control image section
        self.control_image_section = self._create_control_image_section()
        main_layout.addWidget(self.control_image_section)

        # ControlNet parameters (always visible for both control image and inpainting)
        self.params_section = self._create_params_section()
        main_layout.addWidget(self.params_section)

        # Inpainting section
        self.inpainting_section = self._create_inpainting_section()
        main_layout.addWidget(self.inpainting_section)

        main_layout.addStretch()
        self.setLayout(main_layout)

        # Trigger initial model selection to set the model path
        self.on_model_changed(0)

    def _create_model_section(self):
        section = QFrame()
        section.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Model selection
        model_layout = QHBoxLayout()
        model_label = QLabel("Model:")
        model_layout.addWidget(model_label)

        self.model_combo = QComboBox()
        self.model_combo.addItem("Union", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps")
        self.model_combo.addItem("Union Lite", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-lite-2601-8steps")
        self.model_combo.addItem("Tile", "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-2601-8steps")
        self.model_combo.addItem("Tile Lite", "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-lite-2601-8steps")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo, 1)
        layout.addLayout(model_layout)

        # Enable checkbox
        self.enable_checkbox = QCheckBox("Enable ControlNet")
        self.enable_checkbox.setChecked(False)
        self.enable_checkbox.setEnabled(False)  # Disabled until control image is added
        self.enable_checkbox.stateChanged.connect(self.on_enable_changed)
        layout.addWidget(self.enable_checkbox)

        section.setLayout(layout)
        return section

    def _create_control_image_section(self):
        section = QFrame()
        section.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_label = QLabel("Control Image:")
        header_layout.addWidget(header_label, alignment=Qt.AlignmentFlag.AlignLeft)
        header_layout.addStretch(1)

        add_control_btn = QPushButton("Add")
        add_control_btn.setFixedWidth(80)
        add_control_btn.setObjectName("green_button")
        add_control_btn.clicked.connect(self.open_controlnet_dialog)
        header_layout.addWidget(add_control_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.remove_control_btn = QPushButton("Remove")
        self.remove_control_btn.setFixedWidth(80)
        self.remove_control_btn.setObjectName("red_button")
        self.remove_control_btn.clicked.connect(self.on_remove_control_clicked)
        self.remove_control_btn.setVisible(False)
        header_layout.addWidget(self.remove_control_btn, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header_layout)

        self.control_thumb_label = QLabel()
        self.control_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.control_thumb_label.setVisible(False)
        layout.addWidget(self.control_thumb_label)

        section.setLayout(layout)
        return section

    def _create_params_section(self):
        section = QFrame()
        section.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # Conditioning scale
        conditioning_scale_label = QLabel("Conditioning scale")
        layout.addWidget(conditioning_scale_label)

        self.conditioning_scale_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.conditioning_scale_slider.setRange(0.0, 2.0)
        self.conditioning_scale_slider.setSingleStep(0.05)
        self.conditioning_scale_slider.setValue(0.75)
        self.conditioning_scale_slider.valueChanged.connect(self.on_conditioning_scale_changed)
        layout.addWidget(self.conditioning_scale_slider)

        # Guidance Start/End
        self.guidance_start_end_label = QLabel("Guidance Start/End")
        layout.addWidget(self.guidance_start_end_label)

        guidance_layout = QHBoxLayout()
        self.guidance_start_label = QLabel("0%")
        guidance_layout.addWidget(self.guidance_start_label)

        self.guidance_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.guidance_slider.setRange(0.0, 1.0)
        self.guidance_slider.setSingleStep(0.01)
        self.guidance_slider.setValue((0.0, 1.0))
        self.guidance_slider.valueChanged.connect(self.on_guidance_changed)
        guidance_layout.addWidget(self.guidance_slider)

        self.guidance_end_label = QLabel("100%")
        guidance_layout.addWidget(self.guidance_end_label)
        layout.addLayout(guidance_layout)

        # Control mode
        mode_layout = QVBoxLayout()
        mode_layout.addWidget(QLabel("Control mode"))
        self.control_mode_combo = QComboBox()
        self.control_mode_combo.addItem("Balanced", "balanced")
        self.control_mode_combo.addItem("Prompt", "prompt")
        self.control_mode_combo.addItem("ControlNet", "controlnet")
        self.control_mode_combo.currentIndexChanged.connect(self.on_control_mode_changed)
        mode_layout.addWidget(self.control_mode_combo)
        layout.addLayout(mode_layout)

        # Prompt mode decay
        decay_layout = QVBoxLayout()
        decay_layout.addWidget(QLabel("Prompt mode decay"))
        self.prompt_decay_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.prompt_decay_slider.setRange(0.0, 1.0)
        self.prompt_decay_slider.setSingleStep(0.01)
        self.prompt_decay_slider.setValue(0.825)
        self.prompt_decay_slider.valueChanged.connect(self.on_prompt_decay_changed)
        decay_layout.addWidget(self.prompt_decay_slider)
        self.prompt_decay_container = QWidget()
        self.prompt_decay_container.setLayout(decay_layout)
        layout.addWidget(self.prompt_decay_container)

        section.setLayout(layout)
        return section

    def _create_inpainting_section(self):
        section = QFrame()
        section.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout()
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_label = QLabel("Spatial Mask:")
        header_layout.addWidget(header_label, alignment=Qt.AlignmentFlag.AlignLeft)
        header_layout.addStretch(1)

        self.add_inpaint_btn = QPushButton("Add")
        self.add_inpaint_btn.setFixedWidth(80)
        self.add_inpaint_btn.setObjectName("green_button")
        self.add_inpaint_btn.clicked.connect(self.open_controlnet_mask_dialog)
        header_layout.addWidget(self.add_inpaint_btn, alignment=Qt.AlignmentFlag.AlignRight)

        self.remove_inpaint_btn = QPushButton("Remove")
        self.remove_inpaint_btn.setFixedWidth(80)
        self.remove_inpaint_btn.setObjectName("red_button")
        self.remove_inpaint_btn.clicked.connect(self.on_remove_inpaint_clicked)
        self.remove_inpaint_btn.setVisible(False)
        header_layout.addWidget(self.remove_inpaint_btn, alignment=Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header_layout)

        self.mask_info_label = QLabel(
            "Restricts where ControlNet applies.\nWith source image: inpainting (no diff-diff) or spatial (with diff-diff)."
        )
        self.mask_info_label.setWordWrap(True)
        self.mask_info_label.setStyleSheet("QLabel { color: gray; font-style: italic; font-size: 9pt; padding: 4px; }")
        layout.addWidget(self.mask_info_label)

        # Mask preview
        mask_layout = QVBoxLayout()
        mask_label = QLabel("Mask:")
        mask_layout.addWidget(mask_label)
        self.mask_thumb_label = QLabel()
        self.mask_thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_thumb_label.setVisible(False)
        mask_layout.addWidget(self.mask_thumb_label)
        layout.addLayout(mask_layout)

        section.setLayout(layout)
        return section

    def open_controlnet_dialog(self):
        model_index = self.model_combo.currentIndex()
        is_tile = model_index > 1
        self.event_bus.publish(
            "manage_dialog",
            {
                "dialog_type": "controlnet",
                "action": "open",
                "is_tile": is_tile,
            },
        )

    def open_controlnet_mask_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "controlnet_mask", "action": "open"})

    def on_model_changed(self, index):
        model_label = self.model_combo.itemText(index)
        model_name = self.model_combo.itemData(index)

        # Always publish model change to store the model name
        self.event_bus.publish(
            "controlnet",
            {
                "action": "update_model",
                "controlnet_model_name": model_label,
                "controlnet_model_path": model_name,
            },
        )

    def on_enable_changed(self, state):
        enabled = state == Qt.CheckState.Checked.value
        self.event_bus.publish("controlnet", {"action": "enable" if enabled else "disable"})

    def on_remove_control_clicked(self):
        self.event_bus.publish("controlnet", {"action": "remove"})

    def on_remove_inpaint_clicked(self):
        self.event_bus.publish("controlnet", {"action": "remove_mask"})

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
        v = max(0.0, min(1.0, float(value)))
        self.event_bus.publish(
            "controlnet",
            {"action": "update_prompt_decay", "controlnet_prompt_decay": v},
        )

    def _update_prompt_decay_visibility(self, mode: str):
        if hasattr(self, "prompt_decay_container"):
            self.prompt_decay_container.setVisible(mode == "prompt")

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_controlnet_event(self, data: dict):
        action = data.get("action")

        if action in {"add", "update"}:
            # Store controlnet data
            self.controlnet_data.update(data)

            # Show control image preview
            thumb_path = data.get("control_image_thumb_path")
            if thumb_path:
                pixmap = QPixmap(thumb_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                    )
                    self.control_thumb_label.setPixmap(pixmap)
                    self.control_thumb_label.setVisible(True)

            # Enable the checkbox now that we have a control image
            self.enable_checkbox.setEnabled(True)

            # Auto-enable ControlNet when adding control image
            # Block signals to prevent race condition - enable event will be sent by generation_module
            if action == "add":
                blocker = QSignalBlocker(self.enable_checkbox)
                self.enable_checkbox.setChecked(True)
                del blocker

            # Show remove button
            self.remove_control_btn.setVisible(True)

            # Update sliders with data
            self._update_sliders_from_data(data)

        elif action == "remove":
            # Clear control image
            self.controlnet_data = {}
            self.control_thumb_label.clear()
            self.control_thumb_label.setVisible(False)
            self.remove_control_btn.setVisible(False)

            # Only disable checkbox if mask is also not present
            if not self.mask_thumb_label.isVisible():
                self.enable_checkbox.setEnabled(False)
                self.enable_checkbox.setChecked(False)

        elif action in {"add_mask", "update_mask"}:
            # Show mask preview
            mask_thumb_path = data.get("controlnet_mask_thumb_path")
            if mask_thumb_path:
                pixmap = QPixmap(mask_thumb_path)
                if not pixmap.isNull():
                    pixmap = pixmap.scaled(
                        150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                    )
                    self.mask_thumb_label.setPixmap(pixmap)
                    self.mask_thumb_label.setVisible(True)
                    self.remove_inpaint_btn.setVisible(True)

        elif action == "remove_mask":
            self.mask_thumb_label.clear()
            self.mask_thumb_label.setVisible(False)
            self.remove_inpaint_btn.setVisible(False)

            # Only disable checkbox if control image is also not present
            if not self.control_thumb_label.isVisible():
                self.enable_checkbox.setEnabled(False)
                self.enable_checkbox.setChecked(False)

    def _update_sliders_from_data(self, data: dict):
        """Update UI sliders from controlnet data."""
        blocker_scale = QSignalBlocker(self.conditioning_scale_slider)
        self.conditioning_scale_slider.setValue(float(data.get("conditioning_scale", 0.75)))
        del blocker_scale

        guidance = data.get("control_guidance_start_end", [0.0, 1.0])
        blocker_guidance = QSignalBlocker(self.guidance_slider)
        self.guidance_slider.setValue((float(guidance[0]), float(guidance[1])))
        del blocker_guidance

        mode = data.get("controlnet_control_mode", "balanced")
        blocker_mode = QSignalBlocker(self.control_mode_combo)
        for i in range(self.control_mode_combo.count()):
            if self.control_mode_combo.itemData(i) == mode:
                self.control_mode_combo.setCurrentIndex(i)
                break
        del blocker_mode
        self._update_prompt_decay_visibility(str(mode))

        blocker_decay = QSignalBlocker(self.prompt_decay_slider)
        self.prompt_decay_slider.setValue(float(data.get("controlnet_prompt_decay", 0.825)))
        del blocker_decay
