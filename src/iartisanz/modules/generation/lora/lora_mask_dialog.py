"""LoRA spatial mask editing dialog.

This dialog provides mask editing for LoRA spatial masking. The mask controls
where the LoRA effect is applied:
- White/painted areas: LoRA applies fully
- Black/unpainted areas: LoRA is blocked (base model only)
- Gray values: Partial LoRA application
"""

import logging
import os
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QPixmap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout
from superqt import QLabeledDoubleSlider, QLabeledSlider

from iartisanz.app.base_dialog import BaseDialog
from iartisanz.buttons.brush_erase_button import BrushEraseButton
from iartisanz.buttons.color_button import ColorButton
from iartisanz.buttons.eyedropper_button import EyeDropperButton
from iartisanz.modules.generation.source_image.mask_section_widget import MaskSectionWidget
from iartisanz.modules.generation.threads.mask_pixmap_save_thread import MaskPixmapSaveThread


if TYPE_CHECKING:
    from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject


logger = logging.getLogger(__name__)


class LoraMaskDialog(BaseDialog):
    """Dialog for creating/editing LoRA spatial masks.

    This dialog provides mask editing functionality for LoRA spatial masking.
    The mask restricts where the LoRA adapter's effect is applied:
    - Painted (white) regions: LoRA fully applied
    - Unpainted (black) regions: LoRA blocked, base model only
    - Gray values: Partial LoRA strength

    Attributes:
        mask_saved: Signal emitted when mask is saved (path, final_path, thumb_path)
    """

    mask_saved = pyqtSignal(str, str, str)

    def __init__(
        self,
        *args,
        lora: "LoraDataObject" = None,
        lora_mask_path: str = None,
    ):
        if len(args) <= 5:
            logger.warning("LoRA Mask Dialog requires the image viewer, the width and the height.")

        self.image_viewer = args[3] if len(args) > 3 else None
        self.image_width = args[4] if len(args) > 4 else 1024
        self.image_height = args[5] if len(args) > 5 else 1024

        self.lora = lora
        self.lora_mask_path = lora_mask_path

        super().__init__(*args[:3], *args[6:])

        lora_name = lora.name if lora else "LoRA"
        self.setWindowTitle(f"Spatial Mask - {lora_name}")
        self.setMinimumSize(900, 900)

        self.settings = QSettings("ZCode", "ImageArtisanZ")
        self.settings.beginGroup("lora_mask_dialog")
        geometry = self.settings.value("geometry")

        if geometry:
            self.restoreGeometry(geometry)
        self.load_settings()

        self.dialog_busy = False
        self.pixmap_save_thread = None

        self.init_ui()

    def init_ui(self):
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 0, 10, 0)
        content_layout.setSpacing(10)

        # Info label explaining mask purpose
        lora_name = self.lora.name if self.lora else "LoRA"
        info_label = QLabel(
            f"Define where '{lora_name}' applies its effect.\n"
            "White = full LoRA effect, Black = no LoRA (base model only), Gray = partial effect."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { color: gray; font-style: italic; padding: 5px; }")
        content_layout.addWidget(info_label)

        # Brush controls
        brush_layout = QHBoxLayout()
        brush_layout.setContentsMargins(10, 0, 10, 0)
        brush_layout.setSpacing(10)

        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label)
        self.brush_size_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(3, 300)
        self.brush_size_slider.setValue(150)
        brush_layout.addWidget(self.brush_size_slider)

        brush_hardness_label = QLabel("Brush hardness:")
        brush_layout.addWidget(brush_hardness_label)
        self.brush_hardness_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_hardness_slider.setRange(0.0, 0.99)
        self.brush_hardness_slider.setValue(0.0)
        brush_layout.addWidget(self.brush_hardness_slider)

        brush_steps_label = QLabel("Brush steps:")
        brush_layout.addWidget(brush_steps_label)
        self.brush_steps_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_steps_slider.setRange(0.01, 10.00)
        self.brush_steps_slider.setValue(1.25)
        brush_layout.addWidget(self.brush_steps_slider)

        self.brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(self.brush_erase_button)

        self.color_button = ColorButton("Color:")
        brush_layout.addWidget(self.color_button, 0)

        eyedropper_button = EyeDropperButton(25, 25)
        eyedropper_button.clicked.connect(self.on_eyedropper_clicked)
        brush_layout.addWidget(eyedropper_button, 0)

        content_layout.addLayout(brush_layout)

        # Mask editing widget
        self.mask_section_widget = MaskSectionWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_images,
            self.directories.temp_path,
            mask_image_path=self.lora_mask_path,
        )

        # Connect brush controls to mask editor
        editor = self.mask_section_widget.image_widget.image_editor
        self.brush_size_slider.valueChanged.connect(editor.set_brush_size)
        self.brush_hardness_slider.valueChanged.connect(editor.set_brush_hardness)
        self.brush_steps_slider.valueChanged.connect(editor.set_brush_steps)
        self.color_button.color_changed.connect(editor.set_brush_color)
        self.brush_erase_button.brush_selected.connect(self.mask_section_widget.image_widget.set_erase_mode)

        self.mask_section_widget.save_mask_clicked.connect(self.on_mask_saved)
        self.mask_section_widget.mask_canceled.connect(self.on_cancel_mask)
        self.mask_section_widget.mask_deleted.connect(self.on_delete_mask)
        content_layout.addWidget(self.mask_section_widget)

        self.main_layout.addLayout(content_layout)

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self.save_settings()
        event.accept()  # Actually close (don't use BaseDialog's event bus pattern)

    def on_eyedropper_clicked(self):
        QApplication.instance().setOverrideCursor(Qt.CursorShape.CrossCursor)
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
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

    def on_mask_saved(self, mask_pixmap: QPixmap, opacity: float):
        if self.dialog_busy:
            return

        self.dialog_busy = True

        # Remove old mask file if it's in temp directory
        if self.lora_mask_path is not None and self.directories.temp_path in self.lora_mask_path:
            if os.path.isfile(self.lora_mask_path):
                try:
                    os.remove(self.lora_mask_path)
                except Exception:
                    pass

        # Generate unique prefix for this LoRA
        lora_name_safe = ""
        if self.lora:
            # Make name safe for filenames
            lora_name_safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in self.lora.name)
        prefix = f"lora_mask_{lora_name_safe}" if lora_name_safe else "lora_mask"

        self.pixmap_save_thread = MaskPixmapSaveThread(
            mask_pixmap,
            opacity,
            prefix=prefix,
            temp_path=self.directories.temp_path,
            thumb_width=150,
            thumb_height=150,
        )

        self.pixmap_save_thread.save_finished.connect(self.on_mask_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self.on_save_mask_pixmap_thread_finished)
        self.pixmap_save_thread.error.connect(self.on_error)
        self.pixmap_save_thread.start()

    def on_mask_pixmap_saved(self, mask_image_path: str, mask_image_final_path: str, mask_thumbnail_path: str):
        previous_path = self.lora_mask_path

        if previous_path == mask_image_path:
            return

        self.lora_mask_path = mask_image_path

        # Emit signal for LoraAdvancedDialog to handle
        self.mask_saved.emit(mask_image_path, mask_image_final_path, mask_thumbnail_path)

        # Also publish event for other listeners
        action = "add_mask" if previous_path is None else "update_mask"
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

    def on_save_mask_pixmap_thread_finished(self):
        self.pixmap_save_thread.save_finished.disconnect(self.on_mask_pixmap_saved)
        self.pixmap_save_thread.finished.disconnect(self.on_save_mask_pixmap_thread_finished)
        self.pixmap_save_thread.error.disconnect(self.on_error)
        self.pixmap_save_thread = None

        self.dialog_busy = False

    def on_cancel_mask(self):
        """User canceled mask editing - close dialog without saving."""
        self.close()

    def on_delete_mask(self):
        """User deleted the mask - remove it and close dialog."""
        if self.lora_mask_path is not None and self.directories.temp_path in self.lora_mask_path:
            if os.path.isfile(self.lora_mask_path):
                try:
                    os.remove(self.lora_mask_path)
                except Exception:
                    pass

        self.lora_mask_path = None
        self.event_bus.publish(
            "lora",
            {
                "action": "remove_mask",
                "lora": self.lora,
            },
        )
        self.close()

    def on_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})
