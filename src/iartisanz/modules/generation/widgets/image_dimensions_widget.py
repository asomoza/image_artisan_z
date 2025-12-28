from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QSlider, QVBoxLayout, QWidget

from iartisanz.app.event_bus import EventBus


ALLOWED_VALUES = [
    512,
    544,
    576,
    608,
    640,
    672,
    704,
    736,
    768,
    800,
    832,
    864,
    896,
    928,
    960,
    992,
    1024,
    1056,
    1088,
    1120,
    1152,
    1184,
    1216,
    1248,
    1280,
    1312,
    1344,
    1376,
    1408,
    1440,
    1472,
    1504,
    1536,
    1568,
    1600,
    1632,
    1664,
    1696,
    1728,
    1760,
    1792,
    1824,
    1856,
    1888,
    1920,
    1952,
    1984,
    2016,
    2048,
    2080,
    2112,
    2144,
    2176,
    2208,
    2240,
    2272,
    2304,
    2336,
    2368,
    2400,
    2432,
    2464,
    2496,
    2528,
    2560,
    2592,
    2624,
    2656,
    2688,
    2720,
    2752,
    2784,
    2816,
    2848,
    2880,
    2912,
    2944,
    2976,
    3008,
    3040,
    3072,
]


class ImageDimensionsWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.event_bus = EventBus()

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()

        image_sliders_layout = QGridLayout()

        width_label = QLabel("Width")
        image_sliders_layout.addWidget(width_label, 0, 0)

        self.width_slider = QSlider()
        self.width_slider.setRange(512, 3072)
        self.width_slider.setSingleStep(1)
        self.width_slider.setPageStep(1)
        self.width_slider.setOrientation(Qt.Orientation.Horizontal)
        image_sliders_layout.addWidget(self.width_slider, 0, 1)

        self.image_width_value_label = QLabel()
        image_sliders_layout.addWidget(self.image_width_value_label, 0, 2)

        height_label = QLabel("Height")
        image_sliders_layout.addWidget(height_label, 1, 0)

        self.height_slider = QSlider()
        self.height_slider.setRange(512, 3072)
        self.height_slider.setSingleStep(1)
        self.height_slider.setPageStep(1)
        self.height_slider.setOrientation(Qt.Orientation.Horizontal)
        image_sliders_layout.addWidget(self.height_slider, 1, 1)

        self.image_height_value_label = QLabel()
        image_sliders_layout.addWidget(self.image_height_value_label, 1, 2)

        self.width_slider.valueChanged.connect(self.on_slider_value_changed)
        self.height_slider.valueChanged.connect(self.on_slider_value_changed)

        main_layout.addLayout(image_sliders_layout)

        self.setLayout(main_layout)

    def on_slider_value_changed(self):
        slider = self.sender()
        current_value = slider.value()
        nearest_value = min(ALLOWED_VALUES, key=lambda x: abs(x - current_value))

        if slider == self.width_slider:
            self.image_width_value_label.setText(str(nearest_value))
            self.event_bus.publish("generation_change", {"attr": "image_width", "value": nearest_value})
        else:
            self.image_height_value_label.setText(str(nearest_value))
            self.event_bus.publish("generation_change", {"attr": "image_height", "value": nearest_value})
