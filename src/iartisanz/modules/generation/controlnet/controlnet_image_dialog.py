import logging

from PyQt6.QtCore import QEvent, QSettings, Qt
from PyQt6.QtGui import QColor, QCursor, QGuiApplication
from PyQt6.QtWidgets import QApplication, QComboBox, QHBoxLayout, QLabel, QVBoxLayout
from superqt import QLabeledDoubleSlider, QLabeledSlider

from iartisanz.app.base_dialog import BaseDialog
from iartisanz.buttons.brush_erase_button import BrushEraseButton
from iartisanz.buttons.color_button import ColorButton
from iartisanz.buttons.eyedropper_button import EyeDropperButton
from iartisanz.modules.generation.controlnet.controlnet_condition_image_widget import ControlNetConditionImageWidget
from iartisanz.modules.generation.controlnet.controlnet_source_image_widget import ControlNetSourceImageWidget
from iartisanz.modules.generation.controlnet.preprocessor_option_widget import PreprocessorOptionWidget
from iartisanz.modules.generation.image.image_widget import ImageWidget


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
        self.lines_widget = PreprocessorOptionWidget(
            options=[
                ("Lineart", "OzzyGT/lineart"),
                ("Lineart Standard", None),
            ]
        )
        preprocessor_widget_layout.addWidget(self.lines_widget)
        self.edges_widget = PreprocessorOptionWidget(
            options=[
                ("Teed", "OzzyGT/teed"),
            ]
        )
        preprocessor_widget_layout.addWidget(self.edges_widget)
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
        else:
            self.preprocessing_combo.setVisible(True)
            self.preprocessor_label.setVisible(True)
            self.on_preprocessor_changed(self.preprocessing_combo.currentIndex())

    def on_preprocessor_changed(self, index):
        preprocessor = self.preprocessing_combo.itemData(index)
        self.depth_widget.setVisible(preprocessor == "depth")
        self.lines_widget.setVisible(preprocessor == "lines")
        self.edges_widget.setVisible(preprocessor == "edges")
