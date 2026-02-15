import logging
import os

from PyQt6.QtCore import QEvent, QSettings, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QPixmap
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout
from superqt import QLabeledDoubleSlider, QLabeledSlider

from iartisanz.app.base_dialog import BaseDialog
from iartisanz.buttons.brush_erase_button import BrushEraseButton
from iartisanz.buttons.color_button import ColorButton
from iartisanz.buttons.eyedropper_button import EyeDropperButton
from iartisanz.modules.generation.source_image.mask_section_widget import MaskSectionWidget
from iartisanz.modules.generation.threads.mask_pixmap_save_thread import MaskPixmapSaveThread


logger = logging.getLogger(__name__)


class EditImageMaskDialog(BaseDialog):
    """Dialog for creating/editing an inpainting mask on edit image slot 0.

    Paint dark over the areas you want to change. Unpainted (white) regions
    are preserved from the original edit image via differential diffusion.
    """

    def __init__(
        self,
        *args,
        edit_image_mask_path=None,
        edit_image_path=None,
        mask_strength=1.0,
    ):
        if len(args) <= 5:
            logger.warning("Edit Image Mask Dialog requires the image viewer, the width and the height.")

        self.image_viewer = args[3] if len(args) > 3 else None
        self.image_width = args[4] if len(args) > 4 else 1024
        self.image_height = args[5] if len(args) > 5 else 1024

        self.edit_image_mask_path = edit_image_mask_path
        self.edit_image_path = edit_image_path
        self.mask_strength = mask_strength

        super().__init__(*args[:3], *args[6:])

        self.setWindowTitle("Edit Image Inpainting Mask")
        self.setMinimumSize(900, 900)

        self.settings = QSettings("ZCode", "ImageArtisanZ")
        self.settings.beginGroup("edit_image_mask_dialog")
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

        info_label = QLabel(
            "Paint over the areas you want to change.\n"
            "Painted (dark) = generate new content, Unpainted (white) = preserve original."
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

        # Strength slider
        strength_layout = QHBoxLayout()
        strength_layout.setContentsMargins(10, 0, 10, 0)
        strength_layout.setSpacing(10)
        strength_label = QLabel("Strength:")
        strength_layout.addWidget(strength_label)
        self.strength_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0.0, 1.0)
        self.strength_slider.setValue(self.mask_strength)
        self.strength_slider.valueChanged.connect(self.on_strength_changed)
        strength_layout.addWidget(self.strength_slider)
        content_layout.addLayout(strength_layout)

        # Mask editing widget
        self.mask_section_widget = MaskSectionWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_images,
            self.directories.temp_path,
            mask_image_path=self.edit_image_mask_path,
        )

        # Load the edit image as the background/reference layer
        if self.edit_image_path:
            editor = self.mask_section_widget.image_widget.image_editor
            editor.selected_layer = self.mask_section_widget.image_widget.image_layer
            editor.change_layer_image(self.edit_image_path)
            editor.selected_layer = self.mask_section_widget.image_widget.mask_layer

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
        super().closeEvent(event)

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

    def on_strength_changed(self, value: float):
        self.mask_strength = value
        self.event_bus.publish(
            "edit_images",
            {"action": "update_mask_strength", "image_index": 0, "strength": value},
        )

    def on_mask_saved(self, mask_pixmap: QPixmap, opacity: float):
        if self.dialog_busy:
            return

        self.dialog_busy = True

        if self.edit_image_mask_path is not None and self.directories.temp_path in self.edit_image_mask_path:
            if os.path.isfile(self.edit_image_mask_path):
                os.remove(self.edit_image_mask_path)

        self.pixmap_save_thread = MaskPixmapSaveThread(
            mask_pixmap,
            opacity,
            prefix="edit_image_mask",
            temp_path=self.directories.temp_path,
            thumb_width=150,
            thumb_height=150,
        )

        self.pixmap_save_thread.save_finished.connect(self.on_mask_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self.on_save_mask_pixmap_thread_finished)
        self.pixmap_save_thread.error.connect(self.on_error)
        self.pixmap_save_thread.start()

    def on_mask_pixmap_saved(self, mask_image_path: str, mask_image_final_path: str, mask_thumbnail_path: str):
        previous_path = self.edit_image_mask_path

        if previous_path == mask_image_path:
            return

        self.edit_image_mask_path = mask_image_path

        self.event_bus.publish(
            "edit_images",
            {
                "action": "add_mask" if previous_path is None else "update_mask",
                "image_index": 0,
                "mask_path": mask_image_final_path,
                "mask_thumb_path": mask_thumbnail_path,
            },
        )

    def on_save_mask_pixmap_thread_finished(self):
        self.pixmap_save_thread.save_finished.disconnect(self.on_mask_pixmap_saved)
        self.pixmap_save_thread.finished.disconnect(self.on_save_mask_pixmap_thread_finished)
        self.pixmap_save_thread.error.disconnect(self.on_error)
        self.pixmap_save_thread = None

        self.dialog_busy = False

    def on_cancel_mask(self):
        self.close()

    def on_delete_mask(self):
        if self.edit_image_mask_path is not None and self.directories.temp_path in self.edit_image_mask_path:
            if os.path.isfile(self.edit_image_mask_path):
                os.remove(self.edit_image_mask_path)

        self.edit_image_mask_path = None
        self.event_bus.publish("edit_images", {"action": "remove_mask", "image_index": 0})
        self.close()

    def on_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})
