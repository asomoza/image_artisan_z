import logging
import os

import torch
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QImageReader
from PyQt6.QtWidgets import QHBoxLayout, QProgressBar, QSizePolicy, QSpacerItem, QVBoxLayout

from iartisanz.modules.base_module import BaseModule
from iartisanz.modules.generation.constants import LATENT_RGB_FACTORS
from iartisanz.modules.generation.generation_settings import GenerationSettings  # <-- remove Proxy import
from iartisanz.modules.generation.graph.new_graph import create_default_graph
from iartisanz.modules.generation.lora.lora_manager_dialog import LoraManagerDialog
from iartisanz.modules.generation.menus.generation_right_menu import GenerationRightMenu
from iartisanz.modules.generation.source_image.source_image_dialog import SourceImageDialog
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread
from iartisanz.modules.generation.widgets.drop_lightbox_widget import DropLightBox
from iartisanz.modules.generation.widgets.image_viewer_simple_widget import ImageViewerSimpleWidget
from iartisanz.modules.generation.widgets.prompts_widget import PromptsWidget
from iartisanz.utils.image_converters import convert_latents_to_rgb, convert_numpy_to_pixmap
from iartisanz.utils.image_utils import fast_upscale_and_denoise
from iartisanz.utils.json_utils import extract_dict_from_json_graph


logger = logging.getLogger(__name__)


class GenerationModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.settings = QSettings("ZCode", "ImageArtisanZ")

        self.gen_settings = GenerationSettings.load(self.settings)

        self.setAcceptDrops(True)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16
        self.node_graph = create_default_graph()
        self.generating = False

        self.generation_thread = NodeGraphThread(self.directories, self.node_graph, self.dtype, self.device)
        self.generation_thread.progress_update.connect(self.step_progress_update)
        self.generation_thread.status_changed.connect(self.on_status_changed)
        self.generation_thread.generation_finished.connect(self.generation_finished)
        self.generation_thread.generation_error.connect(self.on_generation_error)
        self.generation_thread.generation_aborted.connect(self.generation_aborted)
        self.generation_thread.force_new_run = True

        # Initialize graph nodes from settings
        self.generation_thread.update_nodes(self.gen_settings.to_graph_nodes())

        self.init_ui()

        self.dialogs = {}

        # runtime variables
        self.source_image_path = None
        self.source_thumb_path = None

        self.event_bus.subscribe("generation_change", self.on_generation_change_event)
        self.event_bus.subscribe("manage_dialog", self.on_manage_dialog_event)
        self.event_bus.subscribe("lora", self.on_lora_event)
        self.event_bus.subscribe("generate", self.on_generate_event)
        self.event_bus.subscribe("source_image", self.on_source_image_event)

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(0)
        self.image_viewer = ImageViewerSimpleWidget(self.directories, self.preferences)
        self.image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(self.image_viewer)

        self.right_menu = GenerationRightMenu(self.gen_settings, self.preferences, self.directories)
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

        self.drop_lightbox = DropLightBox(self)
        self.drop_lightbox.setText("Drop image here")

    def closeEvent(self, event):
        self.gen_settings.save(self.settings)

        self.event_bus.unsubscribe("manage_dialog", self.on_manage_dialog_event)
        self.close_all_dialogs()

    def on_generate(
        self,
        seed: int,
        positive_prompt: str,
        negative_prompt: str,
        positive_prompt_changed: bool,
        negative_prompt_changed: bool,
        seed_changed: bool,
    ):
        if self.generating:
            self.on_abort()
            return

        self.generating = True
        self.prompts_widget.set_button_abort()

        self.progress_bar.setMaximum(self.gen_settings.num_inference_steps)
        self.progress_bar.setValue(0)

        if positive_prompt_changed:
            self.generation_thread.update_node("positive_prompt", positive_prompt)

        if negative_prompt_changed:
            self.generation_thread.update_node("negative_prompt", negative_prompt)

        if seed_changed:
            self.generation_thread.update_node("seed", seed)

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
        self.event_bus.publish("status_message", {"action": "change", "message": message})

    def generation_finished(self, image):
        denoise_node = self.node_graph.get_node_by_name("denoise")
        duration = denoise_node.elapsed_time

        if duration is not None:
            self.event_bus.publish(
                "status_message",
                {
                    "action": "change",
                    "message": f"Ready - last generation time: {round(duration, 1)} s ({round(duration * 1000, 2)} ms)",
                },
            )
        else:
            self.event_bus.publish("status_message", {"action": "change", "message": "Ready"})
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)

        pixmap = convert_numpy_to_pixmap(image)

        self.image_viewer.set_pixmap(pixmap)
        self.image_viewer.reset_view()

        json_graph = self.node_graph.to_json()
        self.image_viewer.set_json_graph(json_graph)

        self.prompts_widget.set_button_generate()
        self.generating = False

    def on_abort(self):
        self.generation_thread.abort_graph()

    def generation_aborted(self):
        self.event_bus.publish("status_message", {"action": "change", "message": "Generation aborted"})
        self.prompts_widget.set_button_generate()
        self.generating = False

    def on_generation_error(self, message: str):
        self.event_bus.publish("show_snackbar", {"action": "show", "message": message})
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.prompts_widget.set_button_generate()
        self.generating = False

    def close_all_dialogs(self):
        for dialog in self.dialogs.values():
            dialog.close()

        self.dialogs = {}

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
            self.drop_lightbox.show()
        else:
            event.ignore()

    def dragLeaveEvent(self, event):
        self.drop_lightbox.hide()
        event.accept()

    def dropEvent(self, event):
        self.drop_lightbox.hide()

        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.endswith(".png"):
                reader = QImageReader(path)

                fmt = bytes(reader.format()).decode("ascii", errors="ignore").lower()
                if fmt and fmt != "png":
                    self.event_bus.publish(
                        "show_snackbar", {"action": "show", "message": "Only PNG images are supported"}
                    )

                key = "iartisanz_json_graph"

                if key not in reader.textKeys():
                    self.event_bus.publish(
                        "show_snackbar", {"action": "show", "message": "No generation data found in image."}
                    )

                json_graph = reader.text(key)
                self.generation_thread.loaded_node_graph = json_graph

                wanted_nodes = [
                    "positive_prompt",
                    "negative_prompt",
                    "num_inference_steps",
                    "image_width",
                    "image_height",
                    "seed",
                    "guidance_scale",
                    "guidance_start_end",
                    "scheduler",
                    "loras",
                    "source_image",
                    "strength",
                ]
                subset = extract_dict_from_json_graph(json_graph, wanted_nodes)

                self.event_bus.publish("json_graph", {"action": "loaded", "data": subset})

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_generation_change_event(self, data):
        attr = data.get("attr")
        if not attr:
            return

        value = data.get("value")

        # Update the settings object if this is a known settings key
        handled, graph_value = self.gen_settings.apply_change(attr, value)

        # Forward to the graph:
        # - if it's a settings-backed graph key -> use casted graph_value
        # - else -> forward raw value (existing behavior)
        if handled and attr in self.gen_settings.GRAPH_KEYS:
            if graph_value is None:
                return
            self.generation_thread.update_node(attr, graph_value)
        else:
            self.generation_thread.update_node(attr, value)

    def on_manage_dialog_event(self, data):
        dialog_type = data.get("dialog_type")
        action = data.get("action")

        if dialog_type == "lora_manager":
            if action == "open":
                if "lora_manager" not in self.dialogs:
                    self.dialogs[dialog_type] = LoraManagerDialog(
                        dialog_type, self.directories, self.preferences, self.image_viewer
                    )
                    self.dialogs[dialog_type].setParent(None)
                    self.dialogs[dialog_type].show()
                else:
                    self.dialogs[dialog_type].raise_()
                    self.dialogs[dialog_type].activateWindow()
            elif action == "close":
                self.dialogs[dialog_type].close()
                del self.dialogs[dialog_type]
        elif dialog_type == "source_image":
            if action == "open":
                if "source_image" not in self.dialogs:
                    self.dialogs[dialog_type] = SourceImageDialog(
                        dialog_type,
                        self.directories,
                        self.preferences,
                        self.image_viewer,
                        self.gen_settings.image_width,
                        self.gen_settings.image_height,
                    )
                    self.dialogs[dialog_type].setParent(None)
                    self.dialogs[dialog_type].show()
                else:
                    self.dialogs[dialog_type].raise_()
                    self.dialogs[dialog_type].activateWindow()
            elif action == "close":
                self.dialogs[dialog_type].close()
                del self.dialogs[dialog_type]

    def on_lora_event(self, data: dict):
        action = data.get("action")
        if action == "add":
            lora_data = data.get("lora")
            self.generation_thread.add_lora(lora_data)
        elif action == "remove":
            lora_data = data.get("lora")
            self.generation_thread.remove_lora(lora_data)

    def on_generate_event(self, data: dict):
        action = data.get("action")
        if action == "generate_from_json":
            json_graph = data.get("json_graph")

            wanted_nodes = [
                "positive_prompt",
                "negative_prompt",
                "num_inference_steps",
                "image_width",
                "image_height",
                "seed",
                "guidance_scale",
                "guidance_start_end",
                "scheduler",
                "loras",
            ]
            subset = extract_dict_from_json_graph(json_graph, wanted_nodes)
            self.generation_thread.loaded_node_graph = json_graph

            self.event_bus.publish("json_graph", {"action": "loaded", "data": subset})

            self.on_generate(
                subset.get("seed", 0),
                subset.get("positive_prompt", ""),
                subset.get("negative_prompt", ""),
                True,
                True,
                True,
            )

    def on_source_image_event(self, data: dict):
        action = data.get("action")
        if action == "add":
            self.source_image_path = data.get("source_image_path")
            self.source_thumb_path = data.get("source_thumb_path")
            self.generation_thread.add_source_image(self.source_image_path, self.gen_settings.strength)
        elif action == "update":
            self.generation_thread.update_source_image(data.get("source_image_path"))
        elif action == "enable":
            self.generation_thread.enable_source_image(data.get("value"))
        elif action == "remove":
            self.generation_thread.remove_source_image()
            os.remove(self.source_image_path)
            os.remove(self.source_thumb_path)
            self.source_image_path = None
            self.source_thumb_path = None
