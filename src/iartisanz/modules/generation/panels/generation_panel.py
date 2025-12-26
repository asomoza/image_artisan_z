from PyQt6.QtWidgets import QVBoxLayout

from iartisanz.modules.generation.panels.base_panel import BasePanel
from iartisanz.modules.generation.widgets.image_dimensions_widget import ImageDimensionsWidget


class GenerationPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.update_panel(self.module_options.get("image_width"), self.module_options.get("image_height"))

    def init_ui(self):
        main_layout = QVBoxLayout()

        self.image_dimensions = ImageDimensionsWidget()
        main_layout.addWidget(self.image_dimensions)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def update_panel(self, width: int, height: int):
        self.image_dimensions.width_slider.setValue(width)
        self.image_dimensions.height_slider.setValue(height)
        self.image_dimensions.image_width_value_label.setText(str(width))
        self.image_dimensions.image_height_value_label.setText(str(height))
