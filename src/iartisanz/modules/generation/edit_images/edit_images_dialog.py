import copy
import logging
import os

import cv2
from PyQt6.QtCore import QEvent, QSettings, Qt, QTimer
from PyQt6.QtGui import QColor, QCursor, QGuiApplication, QImage
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QVBoxLayout
from superqt import QLabeledDoubleSlider, QLabeledSlider

from iartisanz.app.base_dialog import BaseDialog
from iartisanz.app.model_manager import get_model_manager
from iartisanz.buttons.brush_erase_button import BrushEraseButton
from iartisanz.buttons.color_button import ColorButton
from iartisanz.buttons.eyedropper_button import EyeDropperButton
from iartisanz.modules.generation.controlnet.canny_preprocessor_widget import CannyPreprocessorWidget
from iartisanz.modules.generation.controlnet.preprocessor_option_widget import PreprocessorOptionWidget
from iartisanz.modules.generation.edit_images.edit_image_result_widget import EditImageResultWidget
from iartisanz.modules.generation.edit_images.edit_image_source_widget import EditImageSourceWidget
from iartisanz.modules.generation.image.image_widget import ImageWidget
from iartisanz.modules.generation.threads.controlnet_preprocess_thread import ControlnetPreprocessThread
from iartisanz.modules.generation.threads.pixmap_save_thread import PixmapSaveThread
from iartisanz.modules.generation.threads.save_layers_thread import SaveLayersThread
from iartisanz.utils.image_converters import numpy_to_pixmap, pixmap_to_numpy_rgb


logger = logging.getLogger(__name__)


class EditImagesDialog(BaseDialog):
    def __init__(
        self,
        *args,
        image_index: int = 0,
        source_image_layers=None,
        result_image_path=None,
        result_image_layers=None,
    ):
        if len(args) <= 5:
            logger.warning("EditImagesDialog requires the image viewer, the width and the height.")

        self.image_viewer = args[3] if len(args) > 3 else None
        self.image_width = args[4] if len(args) > 4 else 1024
        self.image_height = args[5] if len(args) > 5 else 1024

        self.image_index = image_index
        self.source_image_layers = source_image_layers
        self.result_image_path = result_image_path
        self.result_image_layers = result_image_layers

        super().__init__(*args[:3], *args[6:])

        self.preprocess_thread = None
        self._clear_preprocessor_on_finish = False
        self.dialog_busy = False
        self.pixmap_save_thread = None
        self.save_layers_thread = None
        self.source_save_layers_thread = None
        self.result_thumb_path = None
        self.canny_update_timer = QTimer(self)
        self.canny_update_timer.setSingleShot(True)
        self.canny_update_timer.setInterval(75)
        self.canny_update_timer.timeout.connect(self._apply_canny_preprocessor)

        self.setWindowTitle(f"Edit Image {image_index + 1}")
        self.setMinimumSize(1300, 800)

        self.settings = QSettings("ZCode", "ImageArtisanZ")
        self.settings.beginGroup("edit_images_dialog")
        geometry = self.settings.value("geometry")

        if geometry:
            self.restoreGeometry(geometry)
        self.load_settings()

        self.init_ui()

        self.connect_image_widgets()
        self.on_preprocessor_changed(self.preprocessing_combo.currentIndex())

    def init_ui(self):
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 0, 10, 0)
        content_layout.setSpacing(10)

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        self.preprocessor_layout = QHBoxLayout()
        self.preprocessor_label = QLabel("Preprocessor:")
        self.preprocessor_layout.addWidget(self.preprocessor_label)

        self.preprocessing_combo = QComboBox()
        self.preprocessing_combo.addItem("Depth", "depth")
        self.preprocessing_combo.addItem("Lines", "lines")
        self.preprocessing_combo.addItem("Edges", "edges")
        self.preprocessing_combo.currentIndexChanged.connect(self.on_preprocessor_changed)
        self.preprocessor_layout.addWidget(self.preprocessing_combo)
        control_layout.addLayout(self.preprocessor_layout)

        control_layout.setStretch(0, 3)
        control_layout.setStretch(1, 3)
        control_layout.setStretch(2, 6)

        content_layout.addLayout(control_layout)

        preprocessor_widget_layout = QVBoxLayout()
        self.depth_widget = PreprocessorOptionWidget(
            options=[
                ("DepthAnythingV2 Small", "depth-anything/Depth-Anything-V2-Small-hf"),
                ("DepthAnythingV2 Base", "depth-anything/Depth-Anything-V2-Base-hf"),
                ("DepthAnythingV2 Large", "depth-anything/Depth-Anything-V2-Large-hf"),
                ("ZoeDepth (NYU)", "Intel/zoedepth-nyu"),
                ("ZoeDepth (KITTI)", "Intel/zoedepth-kitti"),
                ("ZoeDepth (NYU and KITTI)", "Intel/zoedepth-nyu-kitti"),
            ]
        )
        preprocessor_widget_layout.addWidget(self.depth_widget)
        self.depth_widget.preprocessor_changed.connect(self.on_preprocessor_option_changed)
        self.lines_widget = PreprocessorOptionWidget(
            options=[
                ("Canny (Realtime)", "canny"),
                ("Lineart", "OzzyGT/lineart"),
                ("Lineart Standard", None),
            ]
        )
        preprocessor_widget_layout.addWidget(self.lines_widget)
        self.lines_widget.preprocessor_changed.connect(self.on_preprocessor_option_changed)
        self.canny_widget = CannyPreprocessorWidget()
        self.canny_widget.setVisible(False)
        self.canny_widget.parameters_changed.connect(self.on_canny_params_changed)
        preprocessor_widget_layout.addWidget(self.canny_widget)
        self.edges_widget = PreprocessorOptionWidget(
            options=[
                ("Teed", "OzzyGT/teed"),
            ]
        )
        preprocessor_widget_layout.addWidget(self.edges_widget)
        self.edges_widget.preprocessor_changed.connect(self.on_preprocessor_option_changed)
        content_layout.addLayout(preprocessor_widget_layout)

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

        images_layout = QHBoxLayout()
        images_layout.setContentsMargins(2, 0, 4, 0)
        images_layout.setSpacing(2)

        self.source_widget = EditImageSourceWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_edit_source_images,
            self.directories.temp_path,
            layers=self.source_image_layers,
            delete_original_on_load=False,
        )
        self.source_widget.preprocess_button.clicked.connect(self.on_preprocess_clicked)
        images_layout.addWidget(self.source_widget)

        self.result_widget = EditImageResultWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_edit_images,
            self.directories.temp_path,
            layers=self.result_image_layers,
            delete_original_on_load=False,
        )
        self.result_widget.save_button.clicked.connect(self.on_save_clicked)
        images_layout.addWidget(self.result_widget)
        content_layout.addLayout(images_layout)

        self.main_layout.addLayout(content_layout)

    def load_settings(self):
        geometry = self.settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)

    def save_settings(self):
        self.settings.setValue("geometry", self.saveGeometry())

    def closeEvent(self, event):
        self._clear_preprocessor_model()
        self.save_settings()
        event.ignore()
        self.event_bus.publish(
            "manage_dialog",
            {"dialog_type": self.dialog_type, "action": "close", "image_index": self.image_index},
        )

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

    def connect_image_widgets(self):
        self.connect_image_widget(self.source_widget.image_widget)
        self.connect_image_widget(self.result_widget.image_widget)
        self.result_widget.image_widget.image_changed.connect(self._update_save_button_state)
        self.result_widget.image_widget.image_loaded.connect(self._update_save_button_state)
        self.source_widget.image_widget.image_changed.connect(self._schedule_canny_update)
        if hasattr(self.result_widget.image_widget, "layer_manager_widget"):
            self.result_widget.image_widget.layer_manager_widget.delete_layer_clicked.connect(
                lambda: QTimer.singleShot(0, self._update_save_button_state)
            )

    def connect_image_widget(self, widget: ImageWidget):
        self.brush_size_slider.valueChanged.connect(widget.image_editor.set_brush_size)
        self.brush_hardness_slider.valueChanged.connect(widget.image_editor.set_brush_hardness)
        self.brush_steps_slider.valueChanged.connect(widget.image_editor.set_brush_steps)
        self.color_button.color_changed.connect(widget.image_editor.set_brush_color)
        self.brush_erase_button.brush_selected.connect(widget.set_erase_mode)

    def on_preprocessor_changed(self, index):
        preprocessor = self.preprocessing_combo.itemData(index)
        self.depth_widget.setVisible(preprocessor == "depth")
        self.lines_widget.setVisible(preprocessor == "lines")
        self.edges_widget.setVisible(preprocessor == "edges")
        self._update_canny_visibility()
        self._clear_preprocessor_model()

    def on_preprocessor_option_changed(self, _index: int):
        self._update_canny_visibility()
        self._clear_preprocessor_model()

    def on_canny_params_changed(self):
        self._schedule_canny_update()

    def _schedule_canny_update(self):
        if not self._is_canny_selected():
            return
        self.canny_update_timer.start()

    def _is_canny_selected(self) -> bool:
        return (
            self.preprocessing_combo.itemData(self.preprocessing_combo.currentIndex()) == "lines"
            and self.lines_widget.repo_id == "canny"
        )

    def _update_canny_visibility(self):
        canny_selected = self._is_canny_selected()
        self.canny_widget.setVisible(canny_selected)
        self.lines_widget.set_resolution_visible(not canny_selected)
        if canny_selected:
            self._schedule_canny_update()

    def _clear_preprocessor_model(self):
        if self.preprocess_thread is not None and self.preprocess_thread.isRunning():
            self._clear_preprocessor_on_finish = True
            return
        try:
            get_model_manager().clear_component("preprocessor")
        except Exception:
            logger.debug("Failed clearing preprocessor model", exc_info=True)

    def _set_preprocess_buttons_blocked(self, blocked: bool):
        self.source_widget.disable_buttons(blocked)
        self.result_widget.disable_buttons(blocked)
        self.preprocessing_combo.setEnabled(not blocked)
        self.depth_widget.setEnabled(not blocked)
        self.lines_widget.setEnabled(not blocked)
        self.edges_widget.setEnabled(not blocked)
        self.canny_widget.setEnabled(not blocked)

    def on_preprocess_clicked(self):
        if self.preprocess_thread is not None and self.preprocess_thread.isRunning():
            return

        if self._is_canny_selected():
            self._apply_canny_preprocessor()
            return

        pixmap = self.source_widget.image_widget.image_editor.get_scene_as_pixmap()

        index = self.preprocessing_combo.currentIndex()
        preprocessor_type = self.preprocessing_combo.itemData(index)

        preprocessor_name = ""
        preprocessor_model = ""
        resolution_scale = 1.0

        if preprocessor_type == "depth":
            preprocessor_name = self.depth_widget.option_combo.currentText()
            preprocessor_model = self.depth_widget.repo_id
            resolution_scale = float(self.depth_widget.preprocessor_resolution_slider.value())
        elif preprocessor_type == "lines":
            preprocessor_name = self.lines_widget.option_combo.currentText()
            preprocessor_model = self.lines_widget.repo_id
            resolution_scale = float(self.lines_widget.preprocessor_resolution_slider.value())
            if preprocessor_model is None:
                preprocessor_type = "lineart_standard"
        elif preprocessor_type == "edges":
            preprocessor_name = self.edges_widget.option_combo.currentText()
            preprocessor_model = self.edges_widget.repo_id
            resolution_scale = float(self.edges_widget.preprocessor_resolution_slider.value())
            preprocessor_type = "teed"

        self.preprocess_thread = ControlnetPreprocessThread(
            pixmap,
            preprocessor_type,
            preprocessor_name,
            preprocessor_model,
            resolution_scale,
        )
        self.preprocess_thread.preprocessor_finished.connect(self.on_preprocess_finished)
        self.preprocess_thread.error.connect(self.on_preprocess_error)
        self.preprocess_thread.finished.connect(self.on_preprocess_thread_finished)

        self._set_preprocess_buttons_blocked(True)
        self.preprocess_thread.start()

    def _apply_canny_preprocessor(self):
        if not self._is_canny_selected():
            return

        pixmap = self.source_widget.image_widget.image_editor.get_scene_as_pixmap()
        if pixmap is None or pixmap.isNull():
            return

        low, high, aperture, l2_gradient = self.canny_widget.get_params()

        rgb = pixmap_to_numpy_rgb(pixmap)
        if rgb.size == 0:
            return

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, low, high, apertureSize=aperture, L2gradient=l2_gradient)

        processed_pixmap = numpy_to_pixmap(edges)
        self.result_widget.image_widget.image_editor.change_layer_image(processed_pixmap)
        self._update_save_button_state()

    def on_preprocess_finished(self, pixmap):
        self.result_widget.image_widget.image_editor.change_layer_image(pixmap)
        self._update_save_button_state()

    def on_preprocess_error(self, message: str):
        logger.error("Edit images preprocess failed: %s", message)

    def on_preprocess_thread_finished(self):
        self._set_preprocess_buttons_blocked(False)
        if self._clear_preprocessor_on_finish:
            self._clear_preprocessor_on_finish = False
            self._clear_preprocessor_model()
        self.preprocess_thread = None

    def _update_save_button_state(self):
        if self.result_widget is None:
            return

        editor = self.result_widget.image_widget.image_editor
        layers = editor.get_all_layers()
        if not layers:
            self.result_widget.save_button.setEnabled(False)
            return

        pixmap = editor.get_scene_as_pixmap()
        self.result_widget.save_button.setEnabled(self._pixmap_has_content(pixmap))

    def _pixmap_has_content(self, pixmap) -> bool:
        if pixmap is None or pixmap.isNull():
            return False

        image = pixmap.toImage().convertToFormat(QImage.Format.Format_ARGB32)
        if image.isNull():
            return False

        ptr = image.bits()
        ptr.setsize(image.sizeInBytes())
        data = memoryview(ptr)
        for index in range(3, len(data), 4):
            if data[index] != 0:
                return True
        return False

    def on_save_clicked(self):
        if self.dialog_busy:
            return

        pixmap = self.result_widget.image_widget.image_editor.get_scene_as_pixmap()
        if pixmap is None or pixmap.isNull():
            return

        self.dialog_busy = True

        if (
            self.result_image_path is not None
            and self.directories.temp_path in self.result_image_path
        ):
            if os.path.isfile(self.result_image_path):
                os.remove(self.result_image_path)

        self.pixmap_save_thread = PixmapSaveThread(
            pixmap,
            prefix="edit_image",
            temp_path=self.directories.temp_path,
            thumb_width=120,
            thumb_height=120,
        )
        self.pixmap_save_thread.save_finished.connect(self.on_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.connect(self.on_error)
        self.pixmap_save_thread.start()

    def on_pixmap_saved(self, image_path: str, thumbnail_path: str):
        previous_path = self.result_image_path

        if previous_path == image_path:
            return

        self.result_image_path = image_path
        self.result_thumb_path = thumbnail_path

        self.event_bus.publish(
            "edit_images",
            {
                "action": "add" if previous_path is None else "update",
                "image_index": self.image_index,
                "image_path": image_path,
                "image_thumb_path": thumbnail_path,
            },
        )

        self.save_result_layers()
        self.save_source_layers()

    def save_result_layers(self):
        layers = self.result_widget.image_widget.image_editor.get_all_layers()
        self.save_layers_thread = SaveLayersThread(layers, "edit_result_layer", self.directories.temp_path)
        self.save_layers_thread.error.connect(self.on_error)
        self.save_layers_thread.finished.connect(self.on_save_result_layers_finished)
        self.save_layers_thread.start()

    def save_source_layers(self):
        layers = self.source_widget.image_widget.image_editor.get_all_layers()
        self.source_save_layers_thread = SaveLayersThread(layers, "edit_source_layer", self.directories.temp_path)
        self.source_save_layers_thread.error.connect(self.on_error)
        self.source_save_layers_thread.finished.connect(self.on_save_source_layers_finished)
        self.source_save_layers_thread.start()

    def on_save_pixmap_thread_finished(self):
        self.pixmap_save_thread.save_finished.disconnect(self.on_pixmap_saved)
        self.pixmap_save_thread.finished.disconnect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.disconnect(self.on_error)
        self.pixmap_save_thread = None

        self.dialog_busy = False

    def on_save_result_layers_finished(self):
        self.save_layers_thread.finished.disconnect(self.on_save_result_layers_finished)
        self.save_layers_thread.error.disconnect(self.on_error)
        self.save_layers_thread = None

        layers = self.result_widget.image_widget.image_editor.get_all_layers()
        copied_layers = [copy.copy(layer) for layer in layers]
        for layer in copied_layers:
            layer.pixmap_item = None

        self.event_bus.publish(
            "edit_images",
            {
                "action": "update_layers",
                "image_index": self.image_index,
                "layers": copied_layers,
            },
        )

    def on_save_source_layers_finished(self):
        self.source_save_layers_thread.finished.disconnect(self.on_save_source_layers_finished)
        self.source_save_layers_thread.error.disconnect(self.on_error)
        self.source_save_layers_thread = None

        layers = self.source_widget.image_widget.image_editor.get_all_layers()
        copied_layers = [copy.copy(layer) for layer in layers]
        for layer in copied_layers:
            layer.pixmap_item = None

        self.event_bus.publish(
            "edit_images",
            {
                "action": "update_source_layers",
                "image_index": self.image_index,
                "layers": copied_layers,
            },
        )

    def on_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})
