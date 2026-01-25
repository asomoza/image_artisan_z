from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)
from superqt import QLabeledDoubleSlider

from iartisanz.app.base_simple_dialog import BaseSimpleDialog
from iartisanz.app.event_bus import EventBus
from iartisanz.buttons.linked_button import LinkedButton


if TYPE_CHECKING:
    from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject


logger = logging.getLogger(__name__)


class LoraAdvancedDialog(BaseSimpleDialog):
    def __init__(self, dialog_key: str, lora: LoraDataObject):
        super().__init__("LoRA Advanced Dialog", minWidth=1100, minHeight=550)

        self.event_bus = EventBus()

        self.dialog_key = dialog_key

        self.lora = lora
        self.low_range = 0.0
        self.high_range = 1.0

        self.layer_sliders: dict[str, QLabeledDoubleSlider] = {}
        self.mask_preview_pixmap: Optional[QPixmap] = None

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
        granular_scales_checkbox.setChecked(self.lora.granular_transformer_weights_enabled)
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

        # ============ Spatial Masking Section ============
        self._build_spatial_mask_section()

        self.setLayout(self.dialog_layout)

    def _build_spatial_mask_section(self):
        """Build the spatial masking UI section."""
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("QFrame { color: #555; }")
        self.main_layout.addWidget(separator)

        # Checkbox to enable/disable
        mask_checkbox_layout = QHBoxLayout()
        self.mask_checkbox = QCheckBox("Enable Spatial Masking")
        self.mask_checkbox.setChecked(self.lora.spatial_mask_enabled)
        self.mask_checkbox.toggled.connect(self.on_spatial_mask_toggled)
        mask_checkbox_layout.addWidget(self.mask_checkbox, alignment=Qt.AlignmentFlag.AlignLeft)

        # Info label
        mask_info_label = QLabel("(Restrict where this LoRA applies)")
        mask_info_label.setStyleSheet("QLabel { color: gray; font-style: italic; }")
        mask_checkbox_layout.addWidget(mask_info_label, alignment=Qt.AlignmentFlag.AlignLeft)
        mask_checkbox_layout.addStretch()

        self.main_layout.addLayout(mask_checkbox_layout)

        # Mask frame (collapsible content)
        self.mask_frame = QFrame()
        self.mask_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self.mask_frame.setEnabled(self.lora.spatial_mask_enabled)

        mask_content_layout = QHBoxLayout()
        mask_content_layout.setContentsMargins(10, 10, 10, 10)
        mask_content_layout.setSpacing(15)

        # Left side: Preview
        preview_layout = QVBoxLayout()

        self.mask_preview_label = QLabel()
        self.mask_preview_label.setMinimumSize(120, 120)
        self.mask_preview_label.setMaximumSize(120, 120)
        self.mask_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_preview_label.setStyleSheet(
            "QLabel { border: 1px solid #555; background-color: #2b2b2b; color: #888; font-size: 10px; }"
        )
        self.mask_preview_label.setScaledContents(False)
        preview_layout.addWidget(self.mask_preview_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # Mask stats
        self.mask_stats_label = QLabel("No mask")
        self.mask_stats_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mask_stats_label.setStyleSheet("QLabel { color: gray; font-size: 10px; }")
        preview_layout.addWidget(self.mask_stats_label)

        mask_content_layout.addLayout(preview_layout)

        # Right side: Buttons
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(8)

        self.edit_mask_button = QPushButton("Edit Mask...")
        self.edit_mask_button.setMinimumWidth(100)
        self.edit_mask_button.clicked.connect(self.on_edit_mask)
        buttons_layout.addWidget(self.edit_mask_button)

        self.load_mask_button = QPushButton("Load Mask...")
        self.load_mask_button.setMinimumWidth(100)
        self.load_mask_button.clicked.connect(self.on_load_mask)
        buttons_layout.addWidget(self.load_mask_button)

        self.clear_mask_button = QPushButton("Clear Mask")
        self.clear_mask_button.setMinimumWidth(100)
        self.clear_mask_button.clicked.connect(self.on_clear_mask)
        buttons_layout.addWidget(self.clear_mask_button)

        buttons_layout.addStretch()
        mask_content_layout.addLayout(buttons_layout)
        mask_content_layout.addStretch()

        self.mask_frame.setLayout(mask_content_layout)
        self.main_layout.addWidget(self.mask_frame)

        # Update preview on init
        self._update_mask_preview()
        self._update_mask_stats()

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

        self.event_bus.publish(
            "lora", {"action": "update_slider", "lora": self.lora, "is_slider": self.lora.is_slider}
        )

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

    # ============ Spatial Mask Methods ============

    def on_spatial_mask_toggled(self, checked: bool):
        """Handle spatial masking checkbox toggle."""
        self.lora.spatial_mask_enabled = checked
        self.mask_frame.setEnabled(checked)

        # Publish event
        self.event_bus.publish(
            "lora",
            {
                "action": "update_spatial_mask_enabled",
                "lora": self.lora,
                "enabled": checked,
            },
        )

    def on_edit_mask(self):
        """Open mask editing dialog."""
        # Publish event to open the mask dialog
        # The GenerationModule will handle opening the dialog with proper context
        self.event_bus.publish(
            "lora",
            {
                "action": "open_mask_dialog",
                "lora": self.lora,
            },
        )

    def on_load_mask(self):
        """Load mask from file dialog."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Spatial Mask",
            "",
            "Image Files (*.png *.jpg *.jpeg *.bmp)",
        )

        if file_path:
            self.lora.spatial_mask_path = file_path
            self._calculate_mask_stats()
            self._update_mask_preview()
            self._update_mask_stats()

            # Publish event
            self.event_bus.publish(
                "lora",
                {
                    "action": "update_mask",
                    "lora": self.lora,
                    "lora_mask_path": file_path,
                },
            )

    def on_clear_mask(self):
        """Clear the spatial mask."""
        self.lora.spatial_mask_path = ""
        self._update_mask_preview()
        self._update_mask_stats()

        # Publish event
        self.event_bus.publish(
            "lora",
            {
                "action": "remove_mask",
                "lora": self.lora,
            },
        )

    def on_mask_saved(self, mask_path: str, final_path: str, thumb_path: str):
        """Handle mask saved from editor.

        Called when LoraMaskDialog saves a mask.

        Args:
            mask_path: Path to saved mask file (with alpha)
            final_path: Path to grayscale composite
            thumb_path: Path to thumbnail
        """
        self.lora.spatial_mask_path = mask_path
        self._calculate_mask_stats()
        self._update_mask_preview()
        self._update_mask_stats()

    def _update_mask_preview(self):
        """Update the mask preview thumbnail."""
        if self.lora.spatial_mask_path and os.path.exists(self.lora.spatial_mask_path):
            try:
                # Load mask image
                pixmap = QPixmap(self.lora.spatial_mask_path)

                if not pixmap.isNull():
                    # Scale to fit preview
                    scaled_pixmap = pixmap.scaled(
                        118,
                        118,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )

                    self.mask_preview_label.setPixmap(scaled_pixmap)
                    self.mask_preview_pixmap = scaled_pixmap
                else:
                    self._show_mask_placeholder()

            except Exception as e:
                logger.error(f"Failed to load mask preview: {e}")
                self._show_mask_placeholder("Error loading\nmask")
        else:
            self._show_mask_placeholder()

    def _show_mask_placeholder(self, text: str = None):
        """Show placeholder text in mask preview."""
        self.mask_preview_label.clear()
        if text:
            self.mask_preview_label.setText(text)
        else:
            self.mask_preview_label.setText("No mask\n\nClick 'Edit Mask'\nto create")
        self.mask_preview_pixmap = None

    def _update_mask_stats(self):
        """Update mask statistics label."""
        if self.lora.spatial_mask_path and os.path.exists(self.lora.spatial_mask_path):
            try:
                # Calculate stats if not cached
                self._calculate_mask_stats()

                # Get stats from calculation
                mask_img = Image.open(self.lora.spatial_mask_path)
                width, height = mask_img.size

                # Extract mask data
                if mask_img.mode == "RGBA":
                    mask_array = np.array(mask_img)[:, :, 3]
                elif mask_img.mode == "L":
                    mask_array = np.array(mask_img)
                else:
                    mask_array = np.array(mask_img.convert("L"))

                painted_percent = (mask_array > 127).sum() / mask_array.size * 100

                self.mask_stats_label.setText(f"{painted_percent:.0f}% painted | {width}x{height}")
            except Exception:
                self.mask_stats_label.setText("Mask loaded")
        else:
            self.mask_stats_label.setText("No mask")

    def _calculate_mask_stats(self):
        """Calculate and cache statistics for loaded mask."""
        # Stats are calculated on-demand in _update_mask_stats
        pass
