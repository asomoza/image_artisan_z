from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import QComboBox, QGridLayout, QHBoxLayout, QLabel, QVBoxLayout
from superqt import QDoubleRangeSlider, QLabeledDoubleSlider, QLabeledSlider

from iartisanz.modules.generation.constants import SCHEDULER_NAME_CLASS_MAPPING, SCHEDULER_NAMES
from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.panels.base_panel import BasePanel
from iartisanz.modules.generation.widgets.image_dimensions_widget import ImageDimensionsWidget
from iartisanz.utils.json_utils import cast_number_range


class GenerationPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.scheduler_data_object = SchedulerDataObject()

        self.update_panel(
            self.module_options.get("image_width"),
            self.module_options.get("image_height"),
            self.module_options.get("num_inference_steps"),
            self.module_options.get("guidance_scale"),
            self.module_options.get("guidance_start_end"),
            self.module_options.get("scheduler"),
        )

        self.event_bus.subscribe("json_graph", self.on_json_graph_event)

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

        guidance_label = QLabel("Guidance")
        guidance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        step_guidance_layout.addWidget(guidance_label, 1, 0)

        self.guidance_slider = QLabeledDoubleSlider()
        self.guidance_slider.setObjectName("guidance_slider")
        self.guidance_slider.setRange(1.0, 20.0)
        self.guidance_slider.setSingleStep(0.1)
        self.guidance_slider.setOrientation(Qt.Orientation.Horizontal)
        self.guidance_slider.valueChanged.connect(self.on_guidance_value_changed)
        step_guidance_layout.addWidget(self.guidance_slider, 1, 1)

        guidance_start_end_layout = QHBoxLayout()
        self.guidance_start_value_label = QLabel("0%")
        guidance_start_end_layout.addWidget(self.guidance_start_value_label)

        self.guidance_start_end_slider = QDoubleRangeSlider(Qt.Orientation.Horizontal)
        self.guidance_start_end_slider.setRange(0.0, 1.0)
        self.guidance_start_end_slider.setValue((0.0, 1.0))
        self.guidance_start_end_slider.valueChanged.connect(self.on_guidance_start_end_changed)
        guidance_start_end_layout.addWidget(self.guidance_start_end_slider)

        self.guidance_end_value_label = QLabel("100%")
        guidance_start_end_layout.addWidget(self.guidance_end_value_label)

        step_guidance_layout.addLayout(guidance_start_end_layout, 2, 0, 1, 3)
        main_layout.addLayout(step_guidance_layout)

        scheduler_label = QLabel("Scheduler")
        scheduler_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(scheduler_label)

        self.scheduler_combobox = QComboBox()
        self.scheduler_combobox.addItems(SCHEDULER_NAMES)
        self.scheduler_combobox.currentIndexChanged.connect(self.scheduler_selected)
        main_layout.addWidget(self.scheduler_combobox)

        scheduler_config_layout = QGridLayout()

        shift_label = QLabel("Shift")
        shift_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scheduler_config_layout.addWidget(shift_label, 0, 0)

        self.shift_slider = QLabeledDoubleSlider()
        self.shift_slider.setObjectName("shift_slider")
        self.shift_slider.setRange(1.0, 20.0)
        self.shift_slider.setSingleStep(0.1)
        self.shift_slider.setOrientation(Qt.Orientation.Horizontal)
        self.shift_slider.valueChanged.connect(self.on_shift_value_changed)
        scheduler_config_layout.addWidget(self.shift_slider, 0, 1)

        main_layout.addLayout(scheduler_config_layout)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def _set_guidance_start_end_ui(self, start: float, end: float):
        start = round(float(start), 2)
        end = round(float(end), 2)

        self.guidance_start_value_label.setText(f"{int(start * 100)}%")
        self.guidance_end_value_label.setText(f"{int(end * 100)}%")

        # Prevent valueChanged from firing "generation_change" during programmatic updates
        blocker = QSignalBlocker(self.guidance_start_end_slider)
        try:
            self.guidance_start_end_slider.setValue((start, end))
        finally:
            del blocker

    def on_steps_value_changed(self, value: int):
        self.event_bus.publish("generation_change", {"attr": "num_inference_steps", "value": value})

    def on_guidance_value_changed(self, value: float):
        self.event_bus.publish("generation_change", {"attr": "guidance_scale", "value": value})

    def on_guidance_start_end_changed(self, value: tuple):
        guidance_start, guidance_end = round(value[0], 2), round(value[1], 2)
        self.guidance_start_value_label.setText(f"{int(guidance_start * 100)}%")
        self.guidance_end_value_label.setText(f"{int(guidance_end * 100)}%")

        # Graph expects list-only
        self.event_bus.publish(
            "generation_change",
            {"attr": "guidance_start_end", "value": [guidance_start, guidance_end]},
        )

    def _set_scheduler_ui(self, scheduler_value):
        self.scheduler_data_object = SchedulerDataObject.from_dict(scheduler_value)

        index = 0

        if getattr(self.scheduler_data_object, "name", None) in SCHEDULER_NAMES:
            index = SCHEDULER_NAMES.index(self.scheduler_data_object.name)

        blocker = QSignalBlocker(self.scheduler_combobox)
        try:
            self.scheduler_combobox.setCurrentIndex(index)
            self.shift_slider.setValue(self.scheduler_data_object.shift)
        finally:
            del blocker

    def scheduler_selected(self, index: int):
        selected_scheduler_name = SCHEDULER_NAMES[index]
        selected_scheduler_class = SCHEDULER_NAME_CLASS_MAPPING[selected_scheduler_name]
        self.scheduler_data_object = SchedulerDataObject(
            name=selected_scheduler_name, scheduler_index=index, scheduler_class=selected_scheduler_class
        )

        self.event_bus.publish("generation_change", {"attr": "scheduler", "value": self.scheduler_data_object})

    def on_shift_value_changed(self, value: float):
        self.scheduler_data_object.shift = value
        self.event_bus.publish("generation_change", {"attr": "scheduler", "value": self.scheduler_data_object})

    def update_panel(
        self, width: int, height: int, num_inference_steps: int, guidance_scale: float, guidance_start_end, scheduler
    ):
        # Block signals so we don't emit generation_change while initializing
        blockers = [
            QSignalBlocker(self.image_dimensions.width_slider),
            QSignalBlocker(self.image_dimensions.height_slider),
            QSignalBlocker(self.steps_slider),
            QSignalBlocker(self.guidance_slider),
        ]
        try:
            self.image_dimensions.width_slider.setValue(width)
            self.image_dimensions.height_slider.setValue(height)
            self.image_dimensions.image_width_value_label.setText(str(width))
            self.image_dimensions.image_height_value_label.setText(str(height))
            self.steps_slider.setValue(num_inference_steps)
            self.guidance_slider.setValue(guidance_scale)
        finally:
            for b in blockers:
                del b

        # Guidance start/end (list-only)
        try:
            start, end = cast_number_range(guidance_start_end)
        except Exception:
            start, end = (0.0, 1.0)

        self._set_guidance_start_end_ui(start, end)
        self._set_scheduler_ui(scheduler)

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_json_graph_event(self, data):
        action = data.get("action")
        if action == "loaded":
            data = data.get("data", {})

            width = data.get("image_width")
            height = data.get("image_height")
            num_inference_steps = data.get("num_inference_steps")
            guidance_scale = data.get("guidance_scale")
            guidance_start_end = data.get("guidance_start_end")
            scheduler = data.get("scheduler")

            self.update_panel(width, height, num_inference_steps, guidance_scale, guidance_start_end, scheduler)
