from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGridLayout, QLabel, QVBoxLayout
from superqt import QLabeledSlider

from iartisanz.modules.generation.panels.base_panel import BasePanel
from iartisanz.modules.generation.widgets.image_dimensions_widget import ImageDimensionsWidget


class GenerationPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.update_panel(
            self.module_options.get("image_width"),
            self.module_options.get("image_height"),
            self.module_options.get("num_inference_steps"),
        )

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.image_dimensions = ImageDimensionsWidget()
        main_layout.addWidget(self.image_dimensions)

        step_guidance_layout = QGridLayout()
        steps_label = QLabel("Steps:")
        step_guidance_layout.addWidget(steps_label, 0, 0)

        self.steps_slider = QLabeledSlider()
        self.steps_slider.setRange(1, 100)
        self.steps_slider.setSingleStep(1)
        self.steps_slider.setOrientation(Qt.Orientation.Horizontal)
        self.steps_slider.valueChanged.connect(self.on_steps_value_changed)
        step_guidance_layout.addWidget(self.steps_slider, 0, 1)

        main_layout.addLayout(step_guidance_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def on_steps_value_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "num_inference_steps", "value": value})

    def update_panel(self, width: int, height: int, num_inference_steps: int):
        self.image_dimensions.width_slider.setValue(width)
        self.image_dimensions.height_slider.setValue(height)
        self.image_dimensions.image_width_value_label.setText(str(width))
        self.image_dimensions.image_height_value_label.setText(str(height))
        self.steps_slider.setValue(num_inference_steps)
