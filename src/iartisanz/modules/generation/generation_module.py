from __future__ import annotations

import gc
import logging
import os

import torch
from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QImageReader
from PyQt6.QtWidgets import QHBoxLayout, QProgressBar, QSizePolicy, QSpacerItem, QVBoxLayout

from iartisanz.modules.base_module import BaseModule
from iartisanz.modules.generation.constants import LATENT_RGB_FACTORS
from iartisanz.modules.generation.controlnet.controlnet_image_dialog import ControlNetImageDialog
from iartisanz.modules.generation.controlnet.controlnet_mask_dialog import ControlNetMaskDialog
from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject
from iartisanz.modules.generation.generation_settings import GenerationSettings
from iartisanz.modules.generation.graph.new_graph import create_default_graph
from iartisanz.modules.generation.lora.lora_advanced_dialog import LoraAdvancedDialog
from iartisanz.modules.generation.lora.lora_manager_dialog import LoraManagerDialog
from iartisanz.modules.generation.menus.generation_right_menu import GenerationRightMenu
from iartisanz.modules.generation.model.model_manager_dialog import ModelManagerDialog
from iartisanz.modules.generation.source_image.source_image_dialog import SourceImageDialog
from iartisanz.modules.generation.threads.generation_thread import NodeGraphThread
from iartisanz.modules.generation.widgets.drop_lightbox_widget import DropLightBox
from iartisanz.modules.generation.widgets.image_viewer_simple_widget import ImageViewerSimpleWidget
from iartisanz.modules.generation.widgets.prompts_widget import PromptsWidget
from iartisanz.utils.database import Database
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

        self._last_generation_json_graph: str | None = None

        self.init_ui()
        self.dialogs = {}

        # runtime variables
        self.selected_model = self.gen_settings.model
        self.source_image_path = None
        self.source_image_thumb_path = None
        self.source_image_layers = None
        self.source_image_mask_path = None
        self.source_image_mask_thumb_path = None
        self.loaded_loras: list[LoraDataObject] = []
        self.controlnet_source_image_path = None
        self.controlnet_source_image_layers = None
        self.controlnet_processed_image_path = None
        self.controlnet_processed_image_layers = None
        self.controlnet_model_path = os.path.join(
            self.directories.models_controlnets, "Z-Image-Turbo-Fun-Controlnet-Union-2.1-2601-8steps"
        )
        self.controlnet_condition_thumb_path = None
        self.controlnet_mask_path = None
        self.controlnet_mask_final_path = None
        self.controlnet_mask_thumb_path = None
        self.controlnet_init_image_path = None
        self.controlnet_init_image_layers = None

        # ControlNet denoise behavior controls
        self.controlnet_control_mode = "balanced"
        self.controlnet_prompt_decay = 0.825

        self.create_generation_thread()

        self.event_bus.subscribe("model", self.on_model_event)
        self.event_bus.subscribe("module", self.on_module_event)
        self.event_bus.subscribe("generation_change", self.on_generation_change_event)
        self.event_bus.subscribe("manage_dialog", self.on_manage_dialog_event)
        self.event_bus.subscribe("lora", self.on_lora_event)
        self.event_bus.subscribe("generate", self.on_generate_event)
        self.event_bus.subscribe("source_image", self.on_source_image_event)
        self.event_bus.subscribe("controlnet", self.on_controlnet_event)

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

    def create_generation_thread(self):
        self.generation_thread = NodeGraphThread(self.directories, self.node_graph, self.dtype, self.device)
        self.generation_thread.progress_update.connect(self.step_progress_update)
        self.generation_thread.status_changed.connect(self.on_status_changed)
        self.generation_thread.generation_finished.connect(self.generation_finished)
        self.generation_thread.generation_error.connect(self.on_generation_error)
        self.generation_thread.generation_aborted.connect(self.generation_aborted)

        # Initialize graph nodes from settings
        self.generation_thread.update_nodes(self.gen_settings.to_graph_nodes())

        # model is a special case
        self.generation_thread.update_model(self.selected_model)

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

        if self.selected_model.id == 0:
            self.event_bus.publish(
                "show_snackbar", {"action": "show", "message": "No model selected. Please select a model first."}
            )
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

        json_graph = self.generation_thread.get_staged_json_graph()
        self._last_generation_json_graph = json_graph
        self.generation_thread.start_generation(json_graph)

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

    def generation_finished(self, image, duration=None):
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

        if self._last_generation_json_graph is not None:
            self.image_viewer.set_json_graph(self._last_generation_json_graph)

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
                self.generation_thread.load_json_graph(json_graph)

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
                    "source_image_mask",
                    "use_torch_compile",
                ]
                subset = extract_dict_from_json_graph(json_graph, wanted_nodes)

                # set local values that are not settings-backed
                if "source_image" in subset:
                    self.source_image_path = subset.get("source_image", None)

                if "source_image_mask" in subset:
                    self.source_image_mask_path = subset.get("source_image_mask", None)
                    self.source_image_mask_thumb_path = None

                self.process_lora_data(subset.get("loras", []))

                self.event_bus.publish("json_graph", {"action": "loaded", "data": subset})
                self.event_bus.publish("lora_panel", {"action": "loras_updated", "loaded_loras": self.loaded_loras})

    def process_lora_data(self, loras_data: list[dict]):
        self.loaded_loras = []
        database = Database(os.path.join(self.directories.data_path, "app.db"))

        for lora_data in loras_data:
            database_id = lora_data.get("database_id", 0)

            # needed in case of removal from graph
            lora_data_object = LoraDataObject(
                name=lora_data.get("adapter_name", ""),
                version=lora_data.get("version", ""),
                filename="",
                path="",
                lora_node_name="",
            )

            if database_id != 0:
                lora_db_item = database.select_one(
                    "lora_model",
                    ["name", "version", "model_type", "root_filename", "filepath"],
                    {"id": database_id},
                )

                if lora_db_item is None:
                    # if lora not found in database remove it from the graph
                    logger.debug(
                        f"LoRA {lora_data.get('adapter_name').name} with id {database_id} not found in database."
                    )
                    self.generation_thread.remove_lora(lora_data_object)
                    continue

                lora_data_object = LoraDataObject(
                    id=lora_data.get("id", 0),
                    name=lora_db_item["name"],
                    version=lora_db_item["version"],
                    type=lora_db_item["model_type"],
                    enabled=lora_data.get("enabled", True),
                    is_slider=lora_data.get("is_slider", False),
                    filename=lora_db_item["root_filename"],
                    path=lora_db_item["filepath"],
                    transformer_weight=lora_data.get("transformer_weight", 1.0),
                    lora_node_name=f"{lora_db_item['name']}_{lora_db_item['version']}_lora",
                    granular_transformer_weights_enabled=lora_data.get("granular_transformer_weights_enabled", False),
                    granular_transformer_weights=lora_data.get("granular_transformer_weights", {}),
                )
                self.loaded_loras.append(lora_data_object)
            else:
                # if no database id remove it from the graph
                self.generation_thread.remove_lora(lora_data_object)

        database.disconnect()

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_module_event(self, data: dict):
        action = data.get("action")
        if action == "clear_vram":
            if self.generating and self.generation_thread is not None:
                self.on_abort()
                self.generation_thread.wait()

            self.node_graph = create_default_graph()
            self.generation_thread.clean_up()
            self.generation_thread = None
            self.node_graph = None

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

            self.node_graph = create_default_graph()
            self.create_generation_thread()

            # load loras if any
            if len(self.loaded_loras) > 0:
                for lora_data in self.loaded_loras:
                    self.generation_thread.add_lora(lora_data)

            # reset of prompts widget previous values
            self.prompts_widget.previous_positive_prompt = None
            self.prompts_widget.previous_negative_prompt = None
            self.prompts_widget.previous_seed = None

            # force activate generation
            self.prompts_widget.set_button_generate()
            self.generating = False

        if action == "clear_graph":
            if self.generating and self.generation_thread is not None:
                self.on_abort()
                self.generation_thread.wait()

            # Reset settings to defaults but preserve selected model.
            self.gen_settings.reset_to_defaults(preserve_model=True)
            self.selected_model = self.gen_settings.model

            # Reset runtime state related to optional graph branches.
            self.loaded_loras = []
            self.source_image_path = None
            self.source_image_thumb_path = None
            self.source_image_layers = None
            self.source_image_mask_path = None
            self.source_image_mask_thumb_path = None
            self.controlnet_source_image_path = None
            self.controlnet_source_image_layers = None
            self.controlnet_processed_image_path = None
            self.controlnet_processed_image_layers = None
            self.controlnet_model_path = None
            self.controlnet_condition_thumb_path = None

            # Recreate staged graph/thread without clearing ModelManager / VRAM.
            old_thread = self.generation_thread
            self.generation_thread = None
            self.node_graph = create_default_graph()
            self.create_generation_thread()
            if old_thread is not None:
                try:
                    old_thread.deleteLater()
                except Exception:
                    pass

            # Reset prompts widget defaults
            self.prompts_widget.previous_positive_prompt = None
            self.prompts_widget.previous_negative_prompt = None
            self.prompts_widget.previous_seed = None
            self.prompts_widget.positive_prompt.setPlainText("")
            self.prompts_widget.negative_prompt.setPlainText("")
            self.prompts_widget.seed_text.setText("")
            self.prompts_widget.random_checkbox.setChecked(True)
            self.prompts_widget.seed_text.setDisabled(True)
            self.prompts_widget.use_random_seed = True

            # Update panels that depend on extracted JSON graph state.
            self.event_bus.publish(
                "json_graph",
                {
                    "action": "loaded",
                    "data": {
                        "image_width": self.gen_settings.image_width,
                        "image_height": self.gen_settings.image_height,
                        "num_inference_steps": self.gen_settings.num_inference_steps,
                        "guidance_scale": self.gen_settings.guidance_scale,
                        "guidance_start_end": self.gen_settings.guidance_start_end,
                        "scheduler": self.gen_settings.scheduler,
                        "model": self.gen_settings.model,
                        "strength": self.gen_settings.strength,
                        "use_torch_compile": self.gen_settings.use_torch_compile,
                        "positive_prompt": "",
                        "negative_prompt": "",
                        "seed": "",
                    },
                },
            )
            self.event_bus.publish("lora_panel", {"action": "loras_updated", "loaded_loras": []})

            self.prompts_widget.set_button_generate()
            self.generating = False

    def on_generation_change_event(self, data):
        attr = data.get("attr")
        if not attr:
            return

        value = data.get("value")

        # Update the settings object if this is a known settings key
        handled, graph_value = self.gen_settings.apply_change(attr, value)

        # Forward to the graph:
        if handled and attr in self.gen_settings.GRAPH_KEYS:
            if graph_value is None:
                return
            self.generation_thread.update_node(attr, graph_value)

            # Special-case: if compilation was turned off, actively disable any already-compiled
            # model submodules so toggling behaves as expected.
            if attr == "use_torch_compile" and bool(graph_value) is False:
                try:
                    from iartisanz.app.model_manager import get_model_manager

                    get_model_manager().disable_compiled("transformer")
                except Exception:
                    pass
        else:
            self.generation_thread.update_node(attr, value)

    def _open_dialog(self, dialog_key, dialog_factory):
        if dialog_key not in self.dialogs:
            self.dialogs[dialog_key] = dialog_factory()
            self.dialogs[dialog_key].setParent(self, Qt.WindowType.Window)
            self.dialogs[dialog_key].show()
        else:
            self.dialogs[dialog_key].raise_()
            self.dialogs[dialog_key].activateWindow()

    def _close_dialog(self, dialog_key):
        if dialog_key in self.dialogs:
            self.dialogs[dialog_key].close()
            del self.dialogs[dialog_key]

    def _lora_advanced_key(self, lora_data):
        return f"lora_advanced_{lora_data.name}_{lora_data.version}"

    def _get_dialog_specs(self):
        return {
            "model_manager": {
                "key": lambda _data: "model_manager",
                "factory": lambda _data: ModelManagerDialog(
                    "model_manager", self.directories, self.preferences, self.image_viewer
                ),
            },
            "lora_manager": {
                "key": lambda _data: "lora_manager",
                "factory": lambda _data: LoraManagerDialog(
                    "lora_manager", self.directories, self.preferences, self.image_viewer
                ),
            },
            "source_image": {
                "key": lambda _data: "source_image",
                "factory": lambda _data: SourceImageDialog(
                    "source_image",
                    self.directories,
                    self.preferences,
                    self.image_viewer,
                    self.gen_settings.image_width,
                    self.gen_settings.image_height,
                    source_image_path=self.source_image_path,
                    source_image_layers=self.source_image_layers,
                    source_image_mask_path=self.source_image_mask_path,
                ),
            },
            "lora_advanced": {
                "key": lambda d: self._lora_advanced_key(d.get("lora")),
                "factory": lambda d: LoraAdvancedDialog(
                    self._lora_advanced_key(d.get("lora")),
                    d.get("lora"),
                ),
                "close_key": lambda d: d.get("dialog_key"),
            },
            "controlnet": {
                "key": lambda _data: "controlnet",
                "factory": lambda dialog_data: ControlNetImageDialog(
                    "controlnet",
                    self.directories,
                    self.preferences,
                    self.image_viewer,
                    self.gen_settings.image_width,
                    self.gen_settings.image_height,
                    controlnet_source_image_path=self.controlnet_source_image_path,
                    controlnet_source_image_layers=self.controlnet_source_image_layers,
                    controlnet_processed_image_path=self.controlnet_processed_image_path,
                    controlnet_processed_image_layers=self.controlnet_processed_image_layers,
                    is_tile=dialog_data.get("is_tile", False),
                ),
            },
            "controlnet_mask": {
                "key": lambda _data: "controlnet_mask",
                "factory": lambda dialog_data: ControlNetMaskDialog(
                    "controlnet_mask",
                    self.directories,
                    self.preferences,
                    self.image_viewer,
                    self.gen_settings.image_width,
                    self.gen_settings.image_height,
                    controlnet_mask_path=self.controlnet_mask_path,
                    controlnet_init_image_path=self.controlnet_init_image_path,
                    controlnet_init_image_layers=self.controlnet_init_image_layers,
                ),
            },
        }

    def on_manage_dialog_event(self, data):
        dialog_type = data.get("dialog_type")
        action = data.get("action")

        spec = self._get_dialog_specs().get(dialog_type)
        if not spec:
            return

        if action == "open":
            dialog_key = spec["key"](data)
            if dialog_key is None:
                return
            self._open_dialog(dialog_key, lambda: spec["factory"](data))
        elif action == "close":
            close_key_fn = spec.get("close_key", spec["key"])
            dialog_key = close_key_fn(data)
            if dialog_key is None:
                return
            self._close_dialog(dialog_key)

    def on_model_event(self, data: dict):
        action = data.get("action")
        if action == "update":
            model_data = data.get("model_data_object")

            if self.selected_model != model_data:
                self.selected_model = model_data
                self.generation_thread.update_model(model_data)
                self.gen_settings.apply_change("model", model_data)

    def on_lora_event(self, data: dict):
        action = data.get("action")
        if action == "add":
            lora_data = data.get("lora")
            self.generation_thread.add_lora(lora_data)
            self.loaded_loras.append(lora_data)
        elif action == "remove":
            self.generation_thread.remove_lora(data.get("lora"))
        elif action == "enable":
            self.generation_thread.update_lora_enabled(data.get("lora"))
        elif action == "update_weights":
            self.generation_thread.update_lora_weights(data.get("lora"))
        elif action == "update_lora_transformer_granular_enabled":
            self.generation_thread.update_lora_transformer_granular_enabled(data.get("lora"))
        elif action == "update_slider":
            self.generation_thread.update_lora_slider_enabled(data.get("lora"))

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
                "source_image",
                "strength",
                "source_image_mask",
                "loras",
                "use_torch_compile",
            ]
            subset = extract_dict_from_json_graph(json_graph, wanted_nodes)
            self.generation_thread.load_json_graph(json_graph)

            # Keep runtime state in sync so panels/dialogs reflect what was loaded.
            self.source_image_path = subset.get("source_image")
            self.source_image_layers = None
            self.source_image_mask_path = subset.get("source_image_mask")
            self.source_image_mask_thumb_path = None

            self.process_lora_data(subset.get("loras", []))

            self.event_bus.publish("json_graph", {"action": "loaded", "data": subset})
            self.event_bus.publish("lora_panel", {"action": "loras_updated", "loaded_loras": self.loaded_loras})

            self.on_generate(
                subset.get("seed"), subset.get("positive_prompt"), subset.get("negative_prompt"), True, True, True
            )

    # TODO: refactor this method to make the logic better and cleaner (remove if elif chain)
    def on_source_image_event(self, data: dict):
        action = data.get("action")
        if action == "add":
            self.source_image_path = data.get("source_image_path")
            self.source_thumb_path = data.get("source_thumb_path")
            self.generation_thread.add_source_image(self.source_image_path, self.gen_settings.strength)
        elif action == "update":
            self.source_image_path = data.get("source_image_path")
            self.source_thumb_path = data.get("source_thumb_path")
            self.generation_thread.update_source_image(self.source_image_path)
        elif action == "update_layers":
            self.source_image_layers = data.get("layers", None)
        elif action == "enable":
            self.generation_thread.enable_source_image(data.get("value"))
        elif action == "remove":
            try:
                self.generation_thread.remove_source_image_mask()
            except Exception:
                pass

            if self.source_image_mask_path is not None and self.directories.temp_path in self.source_image_mask_path:
                try:
                    if os.path.isfile(self.source_image_mask_path):
                        os.remove(self.source_image_mask_path)
                except Exception:
                    pass

            self.generation_thread.remove_source_image()
            self.source_image_path = None
            self.source_image_thumb_path = None
            self.source_image_layers = None

            self.source_image_mask_path = None
            self.source_image_mask_thumb_path = None
        elif action == "add_mask":
            self.source_image_mask_path = data.get("source_image_mask_path")
            self.source_image_mask_thumb_path = data.get("source_image_mask_thumb_path")
            self.generation_thread.add_source_image_mask(data.get("source_image_mask_final_path"))
        elif action == "update_mask":
            self.source_image_mask_path = data.get("source_image_mask_path")
            self.source_image_mask_thumb_path = data.get("source_image_mask_thumb_path")
            self.generation_thread.update_source_image_mask(data.get("source_image_mask_final_path"))
        elif action == "remove_mask":
            self.generation_thread.remove_source_image_mask()
            self.source_image_mask_path = None
            self.source_image_mask_thumb_path = None

    # TODO: refactor this method to make the logic better and cleaner (remove if elif chain)
    def on_controlnet_event(self, data: dict):
        action = data.get("action")

        def _validate_control_mode(value: object) -> str:
            mode = str(value or "").strip().lower() or "balanced"
            if mode not in {"balanced", "prompt", "controlnet"}:
                logger.warning("Unknown controlnet_control_mode '%s'; falling back to 'balanced'.", mode)
                return "balanced"
            return mode

        def _clamp_decay(value: object) -> float:
            try:
                v = float(value)
            except Exception:
                return 0.825
            if v != v:  # NaN
                return 0.825
            return max(0.0, min(1.0, v))

        if action in {"add", "update"}:
            self.controlnet_processed_image_path = data.get("control_image_path")
            self.controlnet_condition_thumb_path = data.get("control_image_thumb_path")

            conditioning_scale = data.get("conditioning_scale", 0.75)
            control_guidance_start_end = data.get("control_guidance_start_end", [0.0, 1.0])

            # Allow optional values from payload, but keep current if missing.
            if "controlnet_control_mode" in data:
                self.controlnet_control_mode = _validate_control_mode(data.get("controlnet_control_mode"))
            if "controlnet_prompt_decay" in data:
                self.controlnet_prompt_decay = _clamp_decay(data.get("controlnet_prompt_decay"))

            if self.controlnet_model_path and self.controlnet_processed_image_path:
                self.generation_thread.add_controlnet(
                    controlnet_path=self.controlnet_model_path,
                    control_image_path=self.controlnet_processed_image_path,
                    conditioning_scale=conditioning_scale,
                    control_mode=self.controlnet_control_mode,
                    prompt_decay=self.controlnet_prompt_decay,
                )
                self.generation_thread.update_controlnet(
                    controlnet_path=self.controlnet_model_path,
                    control_image_path=self.controlnet_processed_image_path,
                )
                self.generation_thread.update_controlnet_conditioning_scale(conditioning_scale)
                self.generation_thread.update_node("control_guidance_start_end", control_guidance_start_end)

                # Enable ControlNet after nodes are created (for "add" action)
                if action == "add":
                    self.generation_thread.enable_controlnet(True)

                # Ensure graph has current values (even if ControlNet existed already).
                self.generation_thread.update_node("controlnet_control_mode", self.controlnet_control_mode)
                self.generation_thread.update_node("controlnet_prompt_decay", self.controlnet_prompt_decay)
                self.generation_thread.enable_controlnet(True)

                # If a ControlNet inpaint mask already exists, ensure it is wired.
                if self.controlnet_mask_final_path:
                    self.generation_thread.add_controlnet_mask_image(self.controlnet_mask_final_path)
                    self.generation_thread.update_controlnet_mask_image(self.controlnet_mask_final_path)

        elif action == "update_conditioning_scale":
            conditioning_scale = data.get("conditioning_scale")
            if conditioning_scale is not None:
                self.generation_thread.update_controlnet_conditioning_scale(conditioning_scale)

        elif action == "update_control_guidance_start_end":
            control_guidance_start_end = data.get("control_guidance_start_end")
            if control_guidance_start_end is not None:
                self.generation_thread.update_node("control_guidance_start_end", control_guidance_start_end)

        elif action == "update_control_mode":
            mode = _validate_control_mode(data.get("controlnet_control_mode"))
            self.controlnet_control_mode = mode
            self.generation_thread.update_node("controlnet_control_mode", mode)

        elif action == "update_prompt_decay":
            decay = _clamp_decay(data.get("controlnet_prompt_decay"))
            self.controlnet_prompt_decay = decay
            self.generation_thread.update_node("controlnet_prompt_decay", decay)

        elif action == "update_layers":
            self.controlnet_processed_image_layers = data.get("layers", None)

        elif action == "add_mask":
            self.controlnet_mask_path = data.get("controlnet_mask_path")
            self.controlnet_mask_final_path = data.get("controlnet_mask_final_path")
            self.controlnet_mask_thumb_path = data.get("controlnet_mask_thumb_path")
            if self.controlnet_mask_final_path:
                self.generation_thread.add_controlnet_mask_image(self.controlnet_mask_final_path)

        elif action == "update_mask":
            self.controlnet_mask_path = data.get("controlnet_mask_path")
            self.controlnet_mask_final_path = data.get("controlnet_mask_final_path")
            self.controlnet_mask_thumb_path = data.get("controlnet_mask_thumb_path")
            if self.controlnet_mask_final_path:
                self.generation_thread.update_controlnet_mask_image(self.controlnet_mask_final_path)

        elif action == "remove_mask":
            self.generation_thread.remove_controlnet_mask_image()
            self.controlnet_mask_path = None
            self.controlnet_mask_final_path = None
            self.controlnet_mask_thumb_path = None

        elif action == "add_init_image":
            self.controlnet_init_image_path = data.get("controlnet_init_image_path")
            self.controlnet_init_image_layers = data.get("controlnet_init_image_layers")
            init_image_final_path = data.get("controlnet_init_image_final_path")
            if init_image_final_path:
                self.generation_thread.add_controlnet_init_image(
                    init_image_final_path, controlnet_model_path=self.controlnet_model_path
                )
                # Apply current UI parameter values to the graph
                self.generation_thread.update_node("controlnet_control_mode", self.controlnet_control_mode)
                self.generation_thread.update_node("controlnet_prompt_decay", self.controlnet_prompt_decay)
                # Enable ControlNet after nodes are created
                self.generation_thread.enable_controlnet(True)

        elif action == "update_init_image":
            self.controlnet_init_image_path = data.get("controlnet_init_image_path")
            self.controlnet_init_image_layers = data.get("controlnet_init_image_layers")
            init_image_final_path = data.get("controlnet_init_image_final_path")
            if init_image_final_path:
                self.generation_thread.update_controlnet_init_image(init_image_final_path)

        elif action == "update_init_image_layers":
            self.controlnet_init_image_layers = data.get("controlnet_init_image_layers")

        elif action == "remove_init_image":
            self.generation_thread.remove_controlnet_init_image()
            self.controlnet_init_image_path = None
            self.controlnet_init_image_layers = None

        elif action == "enable":
            self.generation_thread.enable_controlnet(True)

        elif action == "disable":
            self.generation_thread.enable_controlnet(False)

        elif action == "update_model":
            model_name = data.get("controlnet_model_path")
            if model_name:
                self.controlnet_model_path = os.path.join(self.directories.models_controlnets, model_name)
                self.generation_thread.update_controlnet(controlnet_path=self.controlnet_model_path)

        elif action == "remove":
            self.generation_thread.remove_controlnet()
            self.controlnet_model_path = None
            self.controlnet_processed_image_path = None
            self.controlnet_processed_image_layers = None
            self.controlnet_mask_path = None
            self.controlnet_mask_final_path = None
            self.controlnet_mask_thumb_path = None
            self.controlnet_init_image_path = None
            self.controlnet_init_image_layers = None
            self.controlnet_condition_thumb_path = None

            # Reset behavior controls to defaults.
            self.controlnet_control_mode = "balanced"
            self.controlnet_prompt_decay = 0.825
