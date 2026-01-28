from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from superqt import QDoubleRangeSlider, QLabeledDoubleSlider, QLabeledSlider

from iartisanz.modules.generation.constants import SCHEDULER_NAME_CLASS_MAPPING, SCHEDULER_NAMES
from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.panels.base_panel import BasePanel
from iartisanz.modules.generation.widgets.image_dimensions_widget import ImageDimensionsWidget
from iartisanz.utils.json_utils import cast_number_range, cast_scheduler


if TYPE_CHECKING:
    from iartisanz.modules.generation.data_objects.model_data_object import ModelDataObject


class GenerationPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self.init_ui()

        self.scheduler_data_object = SchedulerDataObject()

        self.update_panel(
            self.gen_settings.image_width,
            self.gen_settings.image_height,
            self.gen_settings.num_inference_steps,
            self.gen_settings.guidance_scale,
            self.gen_settings.guidance_start_end,
            self.gen_settings.scheduler,
            self.gen_settings.model,
            bool(getattr(self.gen_settings, "use_torch_compile", False)),
        )

        self.event_bus.subscribe("model", self.on_model_event)
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

        main_layout.addSpacing(10)

        select_base_model_button = QPushButton("Load model")
        select_base_model_button.clicked.connect(self.open_model_manager_dialog)
        main_layout.addWidget(select_base_model_button)

        base_model_layout = QHBoxLayout()
        base_model_label = QLabel("Model: ")
        base_model_layout.addWidget(base_model_label, 0)
        self.selected_base_model_label = QLabel("no model selected")
        base_model_layout.addWidget(self.selected_base_model_label, 1, alignment=Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(base_model_layout)

        self.use_torch_compile_checkbox = QCheckBox("Use torch.compile")
        self.use_torch_compile_checkbox.setChecked(bool(getattr(self.gen_settings, "use_torch_compile", False)))
        self.use_torch_compile_checkbox.toggled.connect(self.on_use_torch_compile_toggled)
        main_layout.addWidget(self.use_torch_compile_checkbox)

        main_layout.addStretch()

        clear_graph_button = QPushButton("Clear Graph")
        clear_graph_button.setObjectName("red_button")
        clear_graph_button.clicked.connect(self.on_clear_graph_clicked)
        main_layout.addWidget(clear_graph_button)

        clear_vram_button = QPushButton("Clear VRAM")
        clear_vram_button.setObjectName("red_button")
        clear_vram_button.clicked.connect(self.on_clear_vram_clicked)
        main_layout.addWidget(clear_vram_button)

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
        self.scheduler_data_object = cast_scheduler(scheduler_value)

        index = 0
        if getattr(self.scheduler_data_object, "name", None) in SCHEDULER_NAMES:
            index = SCHEDULER_NAMES.index(self.scheduler_data_object.name)

        blocker = QSignalBlocker(self.scheduler_combobox)
        try:
            self.scheduler_combobox.setCurrentIndex(index)
            self.shift_slider.setValue(float(getattr(self.scheduler_data_object, "shift", 0.0)))
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
        self,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        guidance_start_end,
        scheduler: SchedulerDataObject,
        model: ModelDataObject,
        use_torch_compile: bool = False,
    ):
        # Block signals so we don't emit generation_change while initializing
        blockers = [
            QSignalBlocker(self.image_dimensions.width_slider),
            QSignalBlocker(self.image_dimensions.height_slider),
            QSignalBlocker(self.steps_slider),
            QSignalBlocker(self.guidance_slider),
            QSignalBlocker(self.use_torch_compile_checkbox),
        ]
        try:
            self.image_dimensions.width_slider.setValue(int(width))
            self.image_dimensions.height_slider.setValue(int(height))
            self.image_dimensions.image_width_value_label.setText(str(int(width)))
            self.image_dimensions.image_height_value_label.setText(str(int(height)))
            self.steps_slider.setValue(int(num_inference_steps))
            self.guidance_slider.setValue(float(guidance_scale))
            self.selected_base_model_label.setText(model.name)
            self.use_torch_compile_checkbox.setChecked(bool(use_torch_compile))
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

    def open_model_manager_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "model_manager", "action": "open"})

    def on_use_torch_compile_toggled(self, checked: bool):
        self.event_bus.publish("generation_change", {"attr": "use_torch_compile", "value": bool(checked)})

    def _confirm_destructive_action(self, title: str, text: str) -> bool:
        res = QMessageBox.question(
            self,
            title,
            text,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return res == QMessageBox.StandardButton.Yes

    def on_clear_graph_clicked(self):
        if not self._confirm_destructive_action(
            "Clear Graph",
            "This will reset the node graph to defaults (removing LoRAs/source image wiring). Continue?",
        ):
            return
        self.event_bus.publish("module", {"action": "clear_graph"})

    def on_clear_vram_clicked(self):
        if not self._confirm_destructive_action(
            "Clear VRAM",
            "This will abort any running generation and unload models from GPU memory. Continue?",
        ):
            return
        self.event_bus.publish("module", {"action": "clear_vram"})

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_model_event(self, data):
        action = data.get("action")
        if action == "update":
            model_data_object = data.get("model_data_object")
            if model_data_object is not None:
                self.selected_base_model_label.setText(model_data_object.name)
            else:
                self.selected_base_model_label.setText("no model selected")

    def on_json_graph_event(self, data):
        action = data.get("action")
        if action == "loaded":
            data = data.get("data", {})

            width = data.get("image_width", self.gen_settings.image_width)
            height = data.get("image_height", self.gen_settings.image_height)
            num_inference_steps = data.get("num_inference_steps", self.gen_settings.num_inference_steps)
            guidance_scale = data.get("guidance_scale", self.gen_settings.guidance_scale)
            guidance_start_end = data.get("guidance_start_end", self.gen_settings.guidance_start_end)
            scheduler = data.get("scheduler", self.gen_settings.scheduler)
            model = data.get("model", self.gen_settings.model)
            # use_torch_compile is NOT loaded from graphs - it's a runtime config
            # Always use the user's persisted setting from gen_settings
            use_torch_compile = self.gen_settings.use_torch_compile

            self.update_panel(
                width,
                height,
                num_inference_steps,
                guidance_scale,
                guidance_start_end,
                scheduler,
                model,
                use_torch_compile,
            )
