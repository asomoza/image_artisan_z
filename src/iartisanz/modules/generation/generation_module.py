import logging

import torch
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import QHBoxLayout, QProgressBar, QSizePolicy, QSpacerItem, QVBoxLayout

from iartisanz.modules.base_module import BaseModule
from iartisanz.modules.generation.constants import LATENT_RGB_FACTORS
from iartisanz.modules.generation.graph.new_graph import create_default_graph
from iartisanz.modules.generation.menus.generation_right_menu import GenerationRightMenu
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread
from iartisanz.modules.generation.widgets.image_viewer_simple_widget import ImageViewerSimpleWidget
from iartisanz.modules.generation.widgets.prompts_widget import PromptsWidget
from iartisanz.utils.image.image_converters import convert_latents_to_rgb, convert_numpy_to_pixmap
from iartisanz.utils.image.image_utils import fast_upscale_and_denoise


class GenerationModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(__name__)

        self.settings = QSettings("ZCode", "ImageArtisanZ")

        self.settings.beginGroup("generation")
        self.module_options = {
            "right_menu_expanded": self.settings.value("right_menu_expanded", True, type=bool),
            "image_width": self.settings.value("image_width", 1024, type=int),
            "image_height": self.settings.value("image_height", 1024, type=int),
        }
        self.settings.endGroup()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.node_graph = create_default_graph()

        self.generation_thread = NodeGraphThread(self.directories, self.node_graph, self.dtype, self.device)
        self.generation_thread.progress_update.connect(self.step_progress_update)
        self.generation_thread.status_changed.connect(self.on_status_changed)
        self.generation_thread.generation_finished.connect(self.generation_finished)
        self.generation_thread.force_new_run = True

        # set initial values for generation
        self.image_width = self.module_options.get("image_width")
        self.image_height = self.module_options.get("image_height")
        self.generation_thread.update_node("image_width", self.image_width)
        self.generation_thread.update_node("image_height", self.image_height)

        self.init_ui()

        self.event_bus.subscribe("generation_change", self.on_generation_change_event)
        self.event_bus.subscribe("open_dialog", self.on_open_dialog_event)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        self.image_viewer = ImageViewerSimpleWidget(self.directories.outputs_images, self.preferences)
        self.image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.image_viewer)

        self.right_menu = GenerationRightMenu(self.module_options, self.preferences, self.directories)
        top_layout.addWidget(self.right_menu)
        top_layout.setStretch(0, 1)

        main_layout.addLayout(top_layout)

        spacer = QSpacerItem(5, 5, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        main_layout.addSpacerItem(spacer)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.prompts_widget = PromptsWidget()
        self.prompts_widget.generate_signal.connect(self.on_generate)
        main_layout.addWidget(self.prompts_widget)

        main_layout.setStretch(0, 16)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 4)

        self.setLayout(main_layout)

    def closeEvent(self, event):
        self.settings.beginGroup("generation")
        self.settings.setValue("right_menu_expanded", self.module_options.get("right_menu_expanded"))
        self.settings.setValue("image_width", self.image_width)
        self.settings.setValue("image_height", self.image_height)
        self.settings.endGroup()

    def on_generate(
        self,
        seed: int,
        positive_prompt: str,
        negative_prompt: str,
        positive_prompt_changed: bool,
        negative_prompt_changed: bool,
        seed_changed: bool,
    ):
        self.progress_bar.setMaximum(9)

        if positive_prompt_changed:
            self.generation_thread.update_positive_prompt(positive_prompt)

        if negative_prompt_changed:
            self.generation_thread.update_negative_prompt(negative_prompt)

        if seed_changed:
            self.generation_thread.update_seed(seed)

        self.generation_thread.start()

    def step_progress_update(self, step: int, latents: torch.Tensor):
        self.progress_bar.setValue(step)

        numpy_image = convert_latents_to_rgb(latents, LATENT_RGB_FACTORS)
        high = 50
        low = 3
        denoise_strength = high + (low - high) * step / (9 - 1)
        numpy_image = fast_upscale_and_denoise(numpy_image, 1, denoise_strength)

        qpixmap = convert_numpy_to_pixmap(numpy_image)

        self.image_viewer.set_pixmap(qpixmap)

        if step == 0:
            self.image_viewer.reset_view()

    def on_status_changed(self, message: str):
        self.event_bus.publish("change_status_message", {"value": message})

    def generation_finished(self, image):
        denoise_node = self.node_graph.get_node_by_name("denoise")
        duration = denoise_node.elapsed_time

        if duration is not None:
            self.event_bus.publish(
                "change_status_message",
                {"value": f"Ready - last generation time: {round(duration, 1)} s ({round(duration * 1000, 2)} ms)"},
            )
        else:
            self.event_bus.publish("change_status_message", {"value": "Ready"})

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)

        pixmap = convert_numpy_to_pixmap(image)

        self.image_viewer.set_pixmap(pixmap)
        self.image_viewer.reset_view()

    def on_generation_change_event(self, data):
        attribute = data.get("attr")
        value = data.get("value")

        if attribute == "image_width":
            self.image_width = value
        elif attribute == "image_height":
            self.image_height = value

        self.generation_thread.update_node(attribute, value)
