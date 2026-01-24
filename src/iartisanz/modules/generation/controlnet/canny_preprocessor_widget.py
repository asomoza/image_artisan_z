from __future__ import annotations

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QComboBox, QHBoxLayout, QLabel, QWidget
from superqt import QLabeledSlider


class CannyPreprocessorWidget(QWidget):
    parameters_changed = pyqtSignal()

    def __init__(
        self,
        low_threshold: int = 100,
        high_threshold: int = 200,
        aperture_size: int = 3,
        l2_gradient: bool = False,
    ):
        super().__init__()

        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.aperture_size = aperture_size
        self.l2_gradient = l2_gradient

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 0, 10, 0)

        low_label = QLabel("Canny Low:")
        main_layout.addWidget(low_label)

        self.low_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.low_slider.setRange(0, 255)
        self.low_slider.setValue(self.low_threshold)
        main_layout.addWidget(self.low_slider)

        high_label = QLabel("Canny High:")
        main_layout.addWidget(high_label)

        self.high_slider = QLabeledSlider(Qt.Orientation.Horizontal)
        self.high_slider.setRange(0, 255)
        self.high_slider.setValue(self.high_threshold)
        main_layout.addWidget(self.high_slider)

        aperture_label = QLabel("Aperture:")
        main_layout.addWidget(aperture_label)

        self.aperture_combo = QComboBox()
        for value in (3, 5, 7):
            self.aperture_combo.addItem(str(value), value)
        self.aperture_combo.setCurrentIndex(self._find_aperture_index(self.aperture_size))
        main_layout.addWidget(self.aperture_combo)

        self.l2_checkbox = QCheckBox("L2 Gradient")
        self.l2_checkbox.setChecked(self.l2_gradient)
        main_layout.addWidget(self.l2_checkbox)

        self.low_slider.valueChanged.connect(self._on_low_changed)
        self.high_slider.valueChanged.connect(self._on_high_changed)
        self.aperture_combo.currentIndexChanged.connect(self._emit_changed)
        self.l2_checkbox.stateChanged.connect(self._emit_changed)

        self.setLayout(main_layout)

    def _find_aperture_index(self, aperture_size: int) -> int:
        for index in range(self.aperture_combo.count()):
            if self.aperture_combo.itemData(index) == aperture_size:
                return index
        return 0

    def _on_low_changed(self, value: int):
        if value > self.high_slider.value():
            self.high_slider.setValue(value)
        self._emit_changed()

    def _on_high_changed(self, value: int):
        if value < self.low_slider.value():
            self.low_slider.setValue(value)
        self._emit_changed()

    def _emit_changed(self, *args):
        self.low_threshold = self.low_slider.value()
        self.high_threshold = self.high_slider.value()
        self.aperture_size = int(self.aperture_combo.currentData())
        self.l2_gradient = self.l2_checkbox.isChecked()
        self.parameters_changed.emit()

    def get_params(self) -> tuple[int, int, int, bool]:
        return (
            self.low_slider.value(),
            self.high_slider.value(),
            int(self.aperture_combo.currentData()),
            self.l2_checkbox.isChecked(),
        )
