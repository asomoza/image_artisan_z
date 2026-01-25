from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from superqt import QLabeledDoubleSlider, QLabeledSlider

from iartisanz.app.base_simple_dialog import BaseSimpleDialog
from iartisanz.app.event_bus import EventBus
from iartisanz.buttons.brush_erase_button import BrushEraseButton
from iartisanz.buttons.color_button import ColorButton
from iartisanz.buttons.eyedropper_button import EyeDropperButton
from iartisanz.buttons.linked_button import LinkedButton
from iartisanz.modules.generation.image.mask_widget import MaskWidget
from iartisanz.modules.generation.threads.mask_pixmap_save_thread import MaskPixmapSaveThread


if TYPE_CHECKING:
    from iartisanz.app.directories import Directories
    from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject
    from iartisanz.modules.generation.widgets.image_viewer_simple_widget import ImageViewerSimpleWidget


logger = logging.getLogger(__name__)


class LoraAdvancedDialog(BaseSimpleDialog):
    # Dialog sizes
    COLLAPSED_MIN_HEIGHT = 550
    EXPANDED_MIN_HEIGHT = 900

    def __init__(
        self,
        dialog_key: str,
        lora: "LoraDataObject",
        image_viewer: "ImageViewerSimpleWidget" = None,
        image_width: int = 1024,
        image_height: int = 1024,
        directories: "Directories" = None,
    ):
        super().__init__("LoRA Advanced Dialog", minWidth=1100, minHeight=self.COLLAPSED_MIN_HEIGHT)

        self.event_bus = EventBus()

        self.dialog_key = dialog_key
        self.lora = lora
        self.image_viewer = image_viewer
        self.image_width = image_width
        self.image_height = image_height
        self.directories = directories

        self.low_range = 0.0
        self.high_range = 1.0

        self.layer_sliders: dict[str, QLabeledDoubleSlider] = {}

        # Mask editor state
        self.mask_widget: MaskWidget = None
        self.pixmap_save_thread: MaskPixmapSaveThread = None
        self.dialog_busy = False

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
        """Build the spatial masking UI section with embedded mask editor."""
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

        # Mask editor frame (expandable content)
        self.mask_editor_frame = QWidget()
        mask_editor_layout = QVBoxLayout()
        mask_editor_layout.setContentsMargins(5, 5, 5, 5)
        mask_editor_layout.setSpacing(8)

        # Info label explaining mask purpose
        lora_name = self.lora.name if self.lora else "LoRA"
        info_label = QLabel(
            f"Define where '{lora_name}' applies its effect. "
            "White = full LoRA effect, Black = no LoRA (base model only), Gray = partial effect."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: gray; font-style: italic; padding: 3px; }")
        mask_editor_layout.addWidget(info_label)

        # Brush controls row
        brush_layout = QHBoxLayout()
        brush_layout.setContentsMargins(0, 0, 0, 0)
        brush_layout.setSpacing(8)

        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label)
        self.brush_size_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(3, 300)
        self.brush_size_slider.setValue(150)
        self.brush_size_slider.setMaximumWidth(150)
        brush_layout.addWidget(self.brush_size_slider)

        brush_hardness_label = QLabel("Hardness:")
        brush_layout.addWidget(brush_hardness_label)
        self.brush_hardness_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_hardness_slider.setRange(0.0, 0.99)
        self.brush_hardness_slider.setValue(0.0)
        self.brush_hardness_slider.setMaximumWidth(120)
        brush_layout.addWidget(self.brush_hardness_slider)

        brush_steps_label = QLabel("Steps:")
        brush_layout.addWidget(brush_steps_label)
        self.brush_steps_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_steps_slider.setRange(0.01, 10.00)
        self.brush_steps_slider.setValue(1.25)
        self.brush_steps_slider.setMaximumWidth(120)
        brush_layout.addWidget(self.brush_steps_slider)

        self.brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(self.brush_erase_button)

        self.color_button = ColorButton("Color:")
        brush_layout.addWidget(self.color_button, 0)

        self.eyedropper_button = EyeDropperButton(25, 25)
        self.eyedropper_button.clicked.connect(self.on_eyedropper_clicked)
        brush_layout.addWidget(self.eyedropper_button, 0)

        brush_layout.addStretch()
        mask_editor_layout.addLayout(brush_layout)

        # Embedded MaskWidget
        if self.image_viewer and self.directories:
            mask_path = self.lora.spatial_mask_path if self.lora.spatial_mask_path else None
            self.mask_widget = MaskWidget(
                "LoRA Spatial Mask",
                "lora_mask",
                self.image_viewer,
                self.image_width,
                self.image_height,
                self.directories.temp_path,
            )
            self.mask_widget.set_layers(mask_path)

            # Connect brush controls to mask editor
            editor = self.mask_widget.image_editor
            self.brush_size_slider.valueChanged.connect(editor.set_brush_size)
            self.brush_hardness_slider.valueChanged.connect(editor.set_brush_hardness)
            self.brush_steps_slider.valueChanged.connect(editor.set_brush_steps)
            self.color_button.color_changed.connect(editor.set_brush_color)
            self.brush_erase_button.brush_selected.connect(self.mask_widget.set_erase_mode)

            mask_editor_layout.addWidget(self.mask_widget)
        else:
            # Fallback: show placeholder if no image_viewer/directories
            placeholder = QLabel("Mask editor unavailable.\nPlease close and reopen this dialog.")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("QLabel { color: #888; font-style: italic; padding: 20px; }")
            mask_editor_layout.addWidget(placeholder)

        # Action buttons row
        action_buttons_layout = QHBoxLayout()
        action_buttons_layout.setSpacing(8)

        self.save_mask_button = QPushButton("Save mask")
        self.save_mask_button.setObjectName("green_button")
        self.save_mask_button.clicked.connect(self._save_current_mask)
        action_buttons_layout.addWidget(self.save_mask_button)

        self.delete_mask_button = QPushButton("Delete mask")
        self.delete_mask_button.setObjectName("red_button")
        self.delete_mask_button.clicked.connect(self._delete_mask)
        action_buttons_layout.addWidget(self.delete_mask_button)

        self.cancel_mask_button = QPushButton("Cancel changes")
        self.cancel_mask_button.clicked.connect(self._cancel_mask_edit)
        action_buttons_layout.addWidget(self.cancel_mask_button)

        action_buttons_layout.addStretch()
        mask_editor_layout.addLayout(action_buttons_layout)

        self.mask_editor_frame.setLayout(mask_editor_layout)

        # Initially hide if spatial masking not enabled
        self.mask_editor_frame.setVisible(self.lora.spatial_mask_enabled)
        self.main_layout.addWidget(self.mask_editor_frame)

        # Adjust dialog size based on initial state
        if self.lora.spatial_mask_enabled:
            self.setMinimumHeight(self.EXPANDED_MIN_HEIGHT)

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
        self.mask_editor_frame.setVisible(checked)

        # Resize dialog
        if checked:
            self.setMinimumHeight(self.EXPANDED_MIN_HEIGHT)
        else:
            self.setMinimumHeight(self.COLLAPSED_MIN_HEIGHT)

        # Publish event
        self.event_bus.publish(
            "lora",
            {
                "action": "update_spatial_mask_enabled",
                "lora": self.lora,
                "enabled": checked,
            },
        )

    def on_eyedropper_clicked(self):
        """Activate eyedropper tool for color picking."""
        QApplication.instance().setOverrideCursor(Qt.CursorShape.CrossCursor)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        """Handle eyedropper color picking."""
        if (
            QApplication.instance().overrideCursor() == Qt.CursorShape.CrossCursor
            and event.type() == QEvent.Type.MouseButtonPress
        ):
            QApplication.instance().restoreOverrideCursor()
            QApplication.instance().removeEventFilter(self)

            global_pos = QCursor.pos()
            screen = QGuiApplication.screenAt(global_pos) or QGuiApplication.primaryScreen()
            if screen is None:
                return super().eventFilter(obj, event)

            screen_pos = global_pos - screen.geometry().topLeft()

            pixmap = screen.grabWindow(0, screen_pos.x(), screen_pos.y(), 1, 1)
            if pixmap.isNull():
                return True

            color = QColor(pixmap.toImage().pixel(0, 0))
            rgb_color = (color.red(), color.green(), color.blue())
            self.color_button.set_color(rgb_color)
            return True

        return super().eventFilter(obj, event)

    def _save_current_mask(self):
        """Save the current mask from the embedded editor."""
        if self.dialog_busy or not self.mask_widget:
            return

        self.dialog_busy = True

        # Remove old mask file if it's in temp directory
        if (
            self.lora.spatial_mask_path
            and self.directories
            and self.directories.temp_path in self.lora.spatial_mask_path
        ):
            if os.path.isfile(self.lora.spatial_mask_path):
                try:
                    os.remove(self.lora.spatial_mask_path)
                except Exception:
                    pass

        # Get mask pixmap from the mask layer
        mask_pixmap = self.mask_widget.mask_layer.pixmap_item.pixmap()
        opacity = self.mask_widget.mask_layer.pixmap_item.opacity()

        # Generate unique prefix for this LoRA
        lora_name_safe = ""
        if self.lora:
            lora_name_safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.lora.name)
        prefix = f"lora_mask_{lora_name_safe}" if lora_name_safe else "lora_mask"

        self.pixmap_save_thread = MaskPixmapSaveThread(
            mask_pixmap,
            opacity,
            prefix=prefix,
            temp_path=self.directories.temp_path if self.directories else "/tmp",
            thumb_width=150,
            thumb_height=150,
        )

        self.pixmap_save_thread.save_finished.connect(self._on_mask_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self._on_save_thread_finished)
        self.pixmap_save_thread.error.connect(self._on_save_error)
        self.pixmap_save_thread.start()

    def _on_mask_pixmap_saved(self, mask_image_path: str, mask_image_final_path: str, mask_thumbnail_path: str):
        """Handle mask save completion."""
        previous_path = self.lora.spatial_mask_path

        if previous_path == mask_image_path:
            return

        self.lora.spatial_mask_path = mask_image_path
        self.lora.spatial_mask_enabled = True

        # Ensure checkbox is checked
        if not self.mask_checkbox.isChecked():
            self.mask_checkbox.setChecked(True)

        # Publish event
        action = "add_mask" if not previous_path else "update_mask"
        self.event_bus.publish(
            "lora",
            {
                "action": action,
                "lora": self.lora,
                "lora_mask_path": mask_image_path,
                "lora_mask_final_path": mask_image_final_path,
                "lora_mask_thumb_path": mask_thumbnail_path,
            },
        )

    def _on_save_thread_finished(self):
        """Clean up after save thread completes."""
        if self.pixmap_save_thread:
            self.pixmap_save_thread.save_finished.disconnect(self._on_mask_pixmap_saved)
            self.pixmap_save_thread.finished.disconnect(self._on_save_thread_finished)
            self.pixmap_save_thread.error.disconnect(self._on_save_error)
            self.pixmap_save_thread = None

        self.dialog_busy = False

    def _on_save_error(self, message: str):
        """Handle mask save error."""
        logger.error(f"Failed to save mask: {message}")
        self.event_bus.publish("show_snackbar", {"action": "show", "message": f"Failed to save mask: {message}"})

    def _cancel_mask_edit(self):
        """Discard changes and reload last saved mask."""
        if not self.mask_widget:
            return

        # Clear the current mask layer
        mask_pixmap = QPixmap(self.image_width, self.image_height)
        mask_pixmap.fill(Qt.GlobalColor.transparent)
        self.mask_widget.image_editor.selected_layer = self.mask_widget.mask_layer
        self.mask_widget.image_editor.change_layer_image(mask_pixmap)

        # Reload from saved path if exists
        if self.lora.spatial_mask_path and os.path.exists(self.lora.spatial_mask_path):
            self.mask_widget.image_editor.change_layer_image(self.lora.spatial_mask_path)

    def _delete_mask(self):
        """Delete the mask and clear the editor."""
        # Remove file if in temp directory
        if (
            self.lora.spatial_mask_path
            and self.directories
            and self.directories.temp_path in self.lora.spatial_mask_path
        ):
            if os.path.isfile(self.lora.spatial_mask_path):
                try:
                    os.remove(self.lora.spatial_mask_path)
                except Exception:
                    pass

        self.lora.spatial_mask_path = ""

        # Clear the mask layer in the editor
        if self.mask_widget:
            mask_pixmap = QPixmap(self.image_width, self.image_height)
            mask_pixmap.fill(Qt.GlobalColor.transparent)
            self.mask_widget.image_editor.selected_layer = self.mask_widget.mask_layer
            self.mask_widget.image_editor.change_layer_image(mask_pixmap)

        # Publish event
        self.event_bus.publish(
            "lora",
            {
                "action": "remove_mask",
                "lora": self.lora,
            },
        )
