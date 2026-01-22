import copy
import logging
import os

from PyQt6.QtCore import QEvent, QSettings, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QVBoxLayout
from superqt import QLabeledDoubleSlider, QLabeledSlider

from iartisanz.app.base_dialog import BaseDialog
from iartisanz.app.model_manager import get_model_manager
from iartisanz.buttons.brush_erase_button import BrushEraseButton
from iartisanz.buttons.color_button import ColorButton
from iartisanz.buttons.eyedropper_button import EyeDropperButton
from iartisanz.modules.generation.controlnet.controlnet_condition_image_widget import ControlNetConditionImageWidget
from iartisanz.modules.generation.controlnet.controlnet_source_image_widget import ControlNetSourceImageWidget
from iartisanz.modules.generation.controlnet.preprocessor_option_widget import PreprocessorOptionWidget
from iartisanz.modules.generation.image.image_widget import ImageWidget
from iartisanz.modules.generation.threads.controlnet_preprocess_thread import ControlnetPreprocessThread
from iartisanz.modules.generation.threads.pixmap_save_thread import PixmapSaveThread
from iartisanz.modules.generation.threads.save_layers_thread import SaveLayersThread


logger = logging.getLogger(__name__)


class ControlNetImageDialog(BaseDialog):
    def __init__(
        self,
        *args,
        controlnet_source_image_path=None,
        controlnet_source_image_layers=None,
        controlnet_processed_image_path=None,
        controlnet_processed_image_layers=None,
    ):
        if len(args) <= 5:
            logger.warning("ControlNet Dialog requires the image viewer, the width and the height.")

        self.image_viewer = args[3] if len(args) > 3 else None
        self.image_width = args[4] if len(args) > 4 else 1024
        self.image_height = args[5] if len(args) > 5 else 1024

        self.controlnet_source_image_path = controlnet_source_image_path
        self.controlnet_source_image_layers = controlnet_source_image_layers
        self.controlnet_processed_image_path = controlnet_processed_image_path
        self.controlnet_processed_image_layers = controlnet_processed_image_layers

        super().__init__(*args[:3], *args[6:])

        self.controlnet_preprocess_thread = None
        self._clear_preprocessor_on_finish = False
        self.dialog_busy = False
        self.pixmap_save_thread = None
        self.save_layers_thread = None
        self.controlnet_processed_thumb_path = None

        self.setWindowTitle("ControlNet")
        self.setMinimumSize(1300, 800)

        self.settings = QSettings("ZCode", "ImageArtisanZ")
        self.settings.beginGroup("controlnet_dialog")
        geometry = self.settings.value("geometry")

        if geometry:
            self.restoreGeometry(geometry)
        self.load_settings()

        self.init_ui()

        self.connect_image_widgets()
        self.on_model_changed(self.model_combo.currentIndex())
        self.on_preprocessor_changed(self.preprocessing_combo.currentIndex())

    def init_ui(self):
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(10, 0, 10, 0)
        content_layout.setSpacing(10)

        control_layout = QHBoxLayout()
        control_layout.setContentsMargins(10, 0, 10, 0)
        control_layout.setSpacing(10)

        model_label = QLabel("ControlNet model:")
        control_layout.addWidget(model_label)

        # the model weights live in ~/.iartisanz/models/controlnet eg: Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps.safetensors
        self.model_combo = QComboBox()
        self.model_combo.addItem("Union", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps")
        self.model_combo.addItem("Union Lite", "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps")
        self.model_combo.addItem("Tile", "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-2601-8steps")
        self.model_combo.addItem("Tile Lite", "Z-Image-Turbo-Fun-Controlnet-Tile-2.1-2601-8steps")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        control_layout.addWidget(self.model_combo)

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
                ("Lineart", "OzzyGT/lineart"),
                ("Lineart Standard", None),
            ]
        )
        preprocessor_widget_layout.addWidget(self.lines_widget)
        self.lines_widget.preprocessor_changed.connect(self.on_preprocessor_option_changed)
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

        self.controlnet_source_image_widget = ControlNetSourceImageWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_controlnet_source_images,
            self.directories.temp_path,
            layers=self.controlnet_source_image_layers,
            delete_original_on_load=False,
        )
        self.controlnet_source_image_widget.preprocess_button.clicked.connect(self.on_preprocess_clicked)
        images_layout.addWidget(self.controlnet_source_image_widget)

        self.controlnet_condition_image_widget = ControlNetConditionImageWidget(
            self.image_viewer,
            self.image_width,
            self.image_height,
            self.directories.outputs_conditioning_images,
            self.directories.temp_path,
            layers=self.controlnet_processed_image_layers,
            delete_original_on_load=False,
        )
        self.controlnet_condition_image_widget.add_button.clicked.connect(self.on_add_clicked)
        images_layout.addWidget(self.controlnet_condition_image_widget)
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

    def connect_image_widgets(self):
        self.connect_image_widget(self.controlnet_source_image_widget.image_widget)
        self.connect_image_widget(self.controlnet_condition_image_widget.image_widget)

    def connect_image_widget(self, widget: ImageWidget):
        self.brush_size_slider.valueChanged.connect(widget.image_editor.set_brush_size)
        self.brush_hardness_slider.valueChanged.connect(widget.image_editor.set_brush_hardness)
        self.brush_steps_slider.valueChanged.connect(widget.image_editor.set_brush_steps)
        self.color_button.color_changed.connect(widget.image_editor.set_brush_color)
        self.brush_erase_button.brush_selected.connect(widget.set_erase_mode)

    def on_model_changed(self, index):
        model_name = self.model_combo.itemData(index)
        if "tile" in model_name.lower():
            self.preprocessing_combo.setVisible(False)
            self.preprocessor_label.setVisible(False)
            self.depth_widget.setVisible(False)
            self.lines_widget.setVisible(False)
            self.edges_widget.setVisible(False)
            self._clear_preprocessor_model()
        else:
            self.preprocessing_combo.setVisible(True)
            self.preprocessor_label.setVisible(True)
            self.on_preprocessor_changed(self.preprocessing_combo.currentIndex())

    def on_preprocessor_changed(self, index):
        preprocessor = self.preprocessing_combo.itemData(index)
        self.depth_widget.setVisible(preprocessor == "depth")
        self.lines_widget.setVisible(preprocessor == "lines")
        self.edges_widget.setVisible(preprocessor == "edges")
        self._clear_preprocessor_model()

    def on_preprocessor_option_changed(self, _index: int):
        self._clear_preprocessor_model()

    def _clear_preprocessor_model(self):
        if self.controlnet_preprocess_thread is not None and self.controlnet_preprocess_thread.isRunning():
            self._clear_preprocessor_on_finish = True
            return
        try:
            get_model_manager().clear_component("preprocessor")
        except Exception:
            logger.debug("Failed clearing preprocessor model", exc_info=True)

    def _set_preprocess_buttons_blocked(self, blocked: bool):
        self.controlnet_source_image_widget.disable_buttons(blocked)
        self.controlnet_condition_image_widget.disable_buttons(blocked)
        self.preprocessing_combo.setEnabled(not blocked)
        self.depth_widget.setEnabled(not blocked)
        self.lines_widget.setEnabled(not blocked)
        self.edges_widget.setEnabled(not blocked)

    def on_preprocess_clicked(self):
        if self.controlnet_preprocess_thread is not None and self.controlnet_preprocess_thread.isRunning():
            return

        pixmap = self.controlnet_source_image_widget.image_widget.image_editor.get_scene_as_pixmap()

        index = self.preprocessing_combo.currentIndex()
        preprocessor_type = self.preprocessing_combo.itemData(index)

        # grab the preprocessor type and model from the selected preprocessor widget
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

        self.controlnet_preprocess_thread = ControlnetPreprocessThread(
            pixmap,
            preprocessor_type,
            preprocessor_name,
            preprocessor_model,
            resolution_scale,
        )
        self.controlnet_preprocess_thread.preprocessor_finished.connect(self.on_preprocess_finished)
        self.controlnet_preprocess_thread.error.connect(self.on_preprocess_error)
        self.controlnet_preprocess_thread.finished.connect(self.on_preprocess_thread_finished)

        self._set_preprocess_buttons_blocked(True)
        self.controlnet_preprocess_thread.start()

    def on_preprocess_finished(self, pixmap):
        self.controlnet_condition_image_widget.image_widget.image_editor.change_layer_image(pixmap)
        self.controlnet_condition_image_widget.add_button.setEnabled(True)

    def on_preprocess_error(self, message: str):
        logger.error("ControlNet preprocess failed: %s", message)

    def on_preprocess_thread_finished(self):
        self._set_preprocess_buttons_blocked(False)
        if self._clear_preprocessor_on_finish:
            self._clear_preprocessor_on_finish = False
            self._clear_preprocessor_model()
        self.controlnet_preprocess_thread = None

    def on_add_clicked(self):
        if self.dialog_busy:
            return

        pixmap = self.controlnet_condition_image_widget.image_widget.image_editor.get_scene_as_pixmap()
        if pixmap is None or pixmap.isNull():
            return

        self.dialog_busy = True

        if (
            self.controlnet_processed_image_path is not None
            and self.directories.temp_path in self.controlnet_processed_image_path
        ):
            if os.path.isfile(self.controlnet_processed_image_path):
                os.remove(self.controlnet_processed_image_path)

        self.pixmap_save_thread = PixmapSaveThread(
            pixmap,
            prefix="controlnet_condition",
            temp_path=self.directories.temp_path,
            thumb_width=150,
            thumb_height=150,
        )
        self.pixmap_save_thread.save_finished.connect(self.on_controlnet_pixmap_saved)
        self.pixmap_save_thread.finished.connect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.connect(self.on_error)
        self.pixmap_save_thread.start()

    def _resolve_controlnet_model_path(self, model_name: str) -> str:
        if not model_name:
            return ""

        # Default to the base "models" directory next to diffusers/singlefile paths.
        base_models_dir = None
        for candidate in (
            self.directories.models_singlefile,
            self.directories.models_diffusers,
            self.directories.models_loras,
            self.directories.data_path,
        ):
            if candidate:
                base_models_dir = os.path.dirname(candidate)
                break

        if base_models_dir is None:
            base_models_dir = os.path.expanduser("~")

        filename = model_name
        if not filename.endswith(".safetensors"):
            filename = f"{filename}.safetensors"

        return os.path.join(base_models_dir, "controlnet", filename)

    def on_controlnet_pixmap_saved(self, image_path: str, thumbnail_path: str):
        previous_path = self.controlnet_processed_image_path

        if previous_path == image_path:
            return

        self.controlnet_processed_image_path = image_path
        self.controlnet_processed_thumb_path = thumbnail_path

        model_name = self.model_combo.itemData(self.model_combo.currentIndex())
        model_label = self.model_combo.currentText()
        model_path = self._resolve_controlnet_model_path(model_name)

        self.event_bus.publish(
            "controlnet",
            {
                "action": "add" if previous_path is None else "update",
                "controlnet_model_name": model_label,
                "controlnet_model_path": model_path,
                "control_image_path": image_path,
                "control_image_thumb_path": thumbnail_path,
                "conditioning_scale": 0.75,
                "control_guidance_start_end": [0.0, 1.0],
            },
        )

        self.save_layers()

    def save_layers(self):
        layers = self.controlnet_condition_image_widget.image_widget.image_editor.get_all_layers()
        self.save_layers_thread = SaveLayersThread(layers, "controlnet_condition_layer", self.directories.temp_path)
        self.save_layers_thread.error.connect(self.on_error)
        self.save_layers_thread.finished.connect(self.on_save_layers_thread_finished)
        self.save_layers_thread.start()

    def on_save_pixmap_thread_finished(self):
        self.pixmap_save_thread.save_finished.disconnect(self.on_controlnet_pixmap_saved)
        self.pixmap_save_thread.finished.disconnect(self.on_save_pixmap_thread_finished)
        self.pixmap_save_thread.error.disconnect(self.on_error)
        self.pixmap_save_thread = None

        self.dialog_busy = False

    def on_save_layers_thread_finished(self):
        self.save_layers_thread.finished.disconnect(self.on_save_layers_thread_finished)
        self.save_layers_thread.error.disconnect(self.on_error)
        self.save_layers_thread = None

        layers = self.controlnet_condition_image_widget.image_widget.image_editor.get_all_layers()
        copied_layers = [copy.copy(layer) for layer in layers]
        for layer in copied_layers:
            layer.pixmap_item = None

        self.event_bus.publish(
            "controlnet",
            {
                "action": "update_layers",
                "layers": copied_layers,
            },
        )

    def on_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})
