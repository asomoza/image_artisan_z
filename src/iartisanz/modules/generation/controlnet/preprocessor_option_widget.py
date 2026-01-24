from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QWidget
from superqt import QDoubleSlider


class PreprocessorOptionWidget(QWidget):
    preprocessor_changed = pyqtSignal(int)
    resolution_changed = pyqtSignal(float)

    def __init__(
        self,
        options: list[tuple[str, object]],
        resolution: float = 0.5,
        selected_index: int = 0,
    ):
        super().__init__()

        self.options = options
        self.option_index = None
        self.repo_id = None
        self.resolution = resolution

        self.init_ui()

        self.option_combo.currentIndexChanged.connect(self.on_option_changed)
        self.option_combo.setCurrentIndex(selected_index)
        self.on_option_changed(selected_index)

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 0, 10, 0)

        self.left_layout = QHBoxLayout()
        main_layout.addLayout(self.left_layout)

        self.option_combo = QComboBox()
        for label, data in self.options:
            self.option_combo.addItem(label, data)
        self.left_layout.addWidget(self.option_combo)

        self.resolution_label = QLabel("Resolution:")
        main_layout.addWidget(self.resolution_label)

        self.preprocessor_resolution_slider = QDoubleSlider(Qt.Orientation.Horizontal)
        self.preprocessor_resolution_slider.setRange(0.05, 1.0)
        self.preprocessor_resolution_slider.setValue(self.resolution)

        main_layout.addWidget(self.preprocessor_resolution_slider)
        self.preprocessor_resolution_value_label = QLabel(f"{int(self.resolution * 100)}%")
        main_layout.addWidget(self.preprocessor_resolution_value_label)

        self.preprocessor_resolution_slider.valueChanged.connect(self.on_preprocessor_resolution_changed)

        self.setLayout(main_layout)

    def on_preprocessor_resolution_changed(self, value: float):
        self.preprocessor_resolution_value_label.setText(f"{int(value * 100)}%")
        self.resolution_changed.emit(value)

    def on_option_changed(self, value: int):
        if value != self.option_index:
            self.option_index = value
            self.repo_id = self.option_combo.currentData()
            self.preprocessor_changed.emit(value)

    def set_resolution_visible(self, visible: bool):
        self.resolution_label.setVisible(visible)
        self.preprocessor_resolution_slider.setVisible(visible)
        self.preprocessor_resolution_value_label.setVisible(visible)
