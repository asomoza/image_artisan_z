import copy
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
from iartisanz.modules.generation.source_image.image_section_widget import ImageSectionWidget
from iartisanz.modules.generation.source_image.mask_section_widget import MaskSectionWidget
from iartisanz.modules.generation.threads.mask_pixmap_save_thread import MaskPixmapSaveThread
from iartisanz.modules.generation.threads.pixmap_save_thread import PixmapSaveThread
from iartisanz.modules.generation.threads.save_layers_thread import SaveLayersThread


logger = logging.getLogger(__name__)


class SourceImageDialog(BaseDialog):
    def __init__(self, *args, source_image_path=None, source_image_layers=None, source_image_mask_path=None):
        if len(args) <= 5:
            logger.warning("Source Image Dialog requires the image viewer, the width and the height.")

        self.image_viewer = args[3] if len(args) > 3 else None
        self.image_width = args[4] if len(args) > 4 else 1024
        self.image_height = args[5] if len(args) > 5 else 1024

        self.source_image_path = source_image_path
        self.source_image_layers = source_image_layers
        self.source_image_mask_path = source_image_mask_path

        super().__init__(*args[:3], *args[6:])

        self.setWindowTitle("Source Image")
        self.setMinimumSize(900, 900)

        self.settings = QSettings("ZCode", "ImageArtisanZ")
        self.settings.beginGroup("source_image_dialog")
        geometry = self.settings.value("geometry")

        if geometry:
            self.restoreGeometry(geometry)
        self.load_settings()

        self.dialog_busy = False

        self.active_section_widget = None
        self.active_editor = None

        self.init_ui()

        self._connect_editor(self.image_section_widget, self.image_section_widget.image_widget.image_editor)

        if self.source_image_layers is None and self.source_image_path is not None:
            self.image_section_widget.image_widget.image_editor.change_layer_image(self.source_image_path)

    def init_ui(self):
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 0, 10, 0)
        content_layout.setSpacing(10)

        brush_layout = QHBoxLayout()
        brush_layout.setContentsMargins(10, 0, 10, 0)
        brush_layout.setSpacing(10)

        brush_size_label = QLabel("Brush size:")
        brush_layout.addWidget(brush_size_label)
        self.brush_size_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.brush_size_slider.setRange(3, 300)
        self.brush_size_slider.setValue(20)
        brush_layout.addWidget(self.brush_size_slider)

        brush_hardness_label = QLabel("Brush hardness:")
        brush_layout.addWidget(brush_hardness_label)
        self.brush_hardness_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_hardness_slider.setRange(0.0, 0.99)
        self.brush_hardness_slider.setValue(0.5)
        brush_layout.addWidget(self.brush_hardness_slider)

        brush_steps_label = QLabel("Brush steps:")
        brush_layout.addWidget(brush_steps_label)
        self.brush_steps_slider = QLabeledDoubleSlider(Qt.Orientation.Horizontal)
        self.brush_steps_slider.setRange(0.01, 10.00)
        self.brush_steps_slider.setValue(0.25)
        brush_layout.addWidget(self.brush_steps_slider)

        self.brush_erase_button = BrushEraseButton()
        brush_layout.addWidget(self.brush_erase_button)

        self.color_button = ColorButton("Color:")
        brush_layout.addWidget(self.color_button, 0)

        eyedropper_button = EyeDropperButton(25, 25)
        eyedropper_button.clicked.connect(self.on_eyedropper_clicked)
        brush_layout.addWidget(eyedropper_button, 0)

        content_layout.addLayout(brush_layout)

        self.image_section_widget = ImageSectionWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_images,
            self.directories.temp_path,
            layers=self.source_image_layers,
        )
        self.image_section_widget.source_image_added.connect(self.on_source_image_added)
        self.image_section_widget.add_mask_clicked.connect(self.on_add_mask_clicked)
        content_layout.addWidget(self.image_section_widget)

        self.mask_section_widget = MaskSectionWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_images,
            self.directories.temp_path,
        )

        # set mask editor defaults
        self.mask_section_widget.image_widget.image_editor.brush_size = 150
        self.mask_section_widget.image_widget.image_editor.hardness = 0.0
        self.mask_section_widget.image_widget.image_editor.steps = 1.25

        self.mask_section_widget.save_mask_clicked.connect(self.on_mask_saved)
        self.mask_section_widget.mask_canceled.connect(self.on_cancel_mask)
        self.mask_section_widget.mask_deleted.connect(self.on_delete_mask)
        content_layout.addWidget(self.mask_section_widget)
        self.mask_section_widget.setVisible(False)

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

    def _connect_editor(self, section_widget, editor):
        self._disconnect_active_editor()

        self.color_button.color_changed.connect(editor.set_brush_color)
        self.brush_size_slider.valueChanged.connect(editor.set_brush_size)
        self.brush_size_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_hardness_slider.valueChanged.connect(editor.set_brush_hardness)
        self.brush_hardness_slider.sliderReleased.connect(editor.hide_brush_preview)
        self.brush_steps_slider.valueChanged.connect(editor.set_brush_steps)
        self.brush_erase_button.brush_selected.connect(section_widget.image_widget.set_erase_mode)

        self.brush_size_slider.setValue(editor.brush_size)
        self.color_button.set_color(editor.brush_color.getRgb()[:3])
        self.brush_hardness_slider.setValue(editor.hardness)
        self.brush_steps_slider.setValue(editor.steps)

        self.active_section_widget = section_widget
        self.active_editor = editor

    def _disconnect_active_editor(self):
        if self.active_editor is not None and self.active_section_widget is not None:
            try:
                self.color_button.color_changed.disconnect(self.active_editor.set_brush_color)
                self.brush_size_slider.valueChanged.disconnect(self.active_editor.set_brush_size)
                self.brush_size_slider.sliderReleased.disconnect(self.active_editor.hide_brush_preview)
                self.brush_hardness_slider.valueChanged.disconnect(self.active_editor.set_brush_hardness)
                self.brush_hardness_slider.sliderReleased.disconnect(self.active_editor.hide_brush_preview)
                self.brush_steps_slider.valueChanged.disconnect(self.active_editor.set_brush_steps)
                self.brush_erase_button.brush_selected.disconnect(
                    self.active_section_widget.image_widget.set_erase_mode
                )
            except TypeError:
                logger.warning("Tried to disconnect signals that were not connected.")

            self.active_editor = None
            self.active_section_widget = None

    def on_source_image_added(self, pixmap: QPixmap):
        if self.dialog_busy:
            return

        self.dialog_busy = True

        if self.source_image_path is not None and self.directories.temp_path in self.source_image_path:
            if os.path.isfile(self.source_image_path):
                os.remove(self.source_image_path)

        self.pixmap_save_thread = PixmapSaveThread(
            pixmap, prefix="source_image", temp_path=self.directories.temp_path, thumb_width=150, thumb_height=150
        )

        self.pixmap_save_thread.save_finished.connect(self.on_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.connect(self.on_error)
        self.pixmap_save_thread.start()

    def on_pixmap_saved(self, image_path: str, thumbnail_path: str):
        previous_path = self.source_image_path

        if previous_path == image_path:
            return

        self.source_image_path = image_path
        self.source_thumb_path = thumbnail_path

        self.event_bus.publish(
            "source_image",
            {
                "action": "add" if previous_path is None else "update",
                "source_image_path": image_path,
                "source_thumb_path": thumbnail_path,
            },
        )

        self.save_layers()

    def save_layers(self):
        self.source_image_layers = self.image_section_widget.image_widget.image_editor.get_all_layers()
        self.save_layers_thread = SaveLayersThread(
            self.source_image_layers, "source_img_layer", self.directories.temp_path
        )
        self.save_layers_thread.error.connect(self.on_error)
        self.save_layers_thread.finished.connect(self.on_save_layers_thread_finished)
        self.save_layers_thread.start()

    def on_save_pixmap_thread_finished(self):
        self.pixmap_save_thread.save_finished.disconnect(self.on_pixmap_saved)
        self.pixmap_save_thread.finished.disconnect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.disconnect(self.on_error)
        self.pixmap_save_thread = None

        self.dialog_busy = False

    def on_save_layers_thread_finished(self):
        self.save_layers_thread.finished.disconnect(self.on_save_layers_thread_finished)
        self.save_layers_thread.error.disconnect(self.on_error)
        self.save_layers_thread = None

        layers = self.image_section_widget.image_widget.image_editor.get_all_layers()

        copied_layers = [copy.copy(layer) for layer in layers]
        for layer in copied_layers:
            layer.pixmap_item = None

        self.event_bus.publish(
            "source_image",
            {
                "action": "update_layers",
                "layers": copied_layers,
            },
        )

    def on_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})

    def on_add_mask_clicked(self, pixmap: QPixmap):
        self.mask_section_widget.image_widget.image_editor.selected_layer = (
            self.mask_section_widget.image_widget.image_layer
        )
        self.mask_section_widget.image_widget.image_editor.change_layer_image(pixmap)
        self.mask_section_widget.image_widget.image_editor.selected_layer = (
            self.mask_section_widget.image_widget.mask_layer
        )

        self._connect_editor(self.mask_section_widget, self.mask_section_widget.image_widget.image_editor)
        self.image_section_widget.hide()
        self.mask_section_widget.show()

    def on_cancel_mask(self):
        self._connect_editor(self.image_section_widget, self.image_section_widget.image_widget.image_editor)
        self.mask_section_widget.hide()
        self.image_section_widget.show()

    def on_mask_saved(self, mask_pixmap: QPixmap, opacity: float):
        self._connect_editor(self.image_section_widget, self.image_section_widget.image_widget.image_editor)
        self.mask_section_widget.hide()
        self.image_section_widget.show()
        self.image_section_widget.add_mask_button.setText("Edit mask")

        self.pixmap_save_thread = MaskPixmapSaveThread(
            mask_pixmap,
            opacity,
            prefix="source_mask_image",
            temp_path=self.directories.temp_path,
            thumb_width=150,
            thumb_height=150,
        )

        self.pixmap_save_thread.save_finished.connect(self.on_mask_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self.on_save_mask_pixmap_thread_finished)
        self.pixmap_save_thread.error.connect(self.on_error)
        self.pixmap_save_thread.start()

    def on_mask_pixmap_saved(self, mask_image_path: str, mask_image_final_path: str, mask_thumbnail_path: str):
        previous_path = self.source_image_mask_path

        if previous_path == mask_image_path:
            return

        self.source_image_mask_path = mask_image_path

        self.event_bus.publish(
            "source_image",
            {
                "action": "add_mask" if previous_path is None else "update_mask",
                "source_image_mask_path": mask_image_path,
                "source_image_mask_final_path": mask_image_final_path,
                "source_image_mask_thumb_path": mask_thumbnail_path,
            },
        )

    def on_save_mask_pixmap_thread_finished(self):
        self.pixmap_save_thread.save_finished.disconnect(self.on_mask_pixmap_saved)
        self.pixmap_save_thread.finished.disconnect(self.on_save_mask_pixmap_thread_finished)
        self.pixmap_save_thread.error.disconnect(self.on_error)
        self.pixmap_save_thread = None

    def on_delete_mask(self):
        self._connect_editor(self.image_section_widget, self.image_section_widget.image_widget.image_editor)
        self.mask_section_widget.hide()
        self.image_section_widget.show()
        self.image_section_widget.add_mask_button.setText("Add mask")

        if self.source_image_mask_path is not None and self.directories.temp_path in self.source_image_mask_path:
            if os.path.isfile(self.source_image_mask_path):
                os.remove(self.source_image_mask_path)

        self.source_image_mask_path = None
        self.event_bus.publish("source_image", {"action": "remove_mask"})
