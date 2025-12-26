import logging

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.app.directories import DirectoriesObject
from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode


class NodeGraphThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    generation_finished = pyqtSignal(np.ndarray)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(
        self,
        directories: DirectoriesObject,
        node_graph: ImageArtisanZNodeGraph,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()

        self.logger = logging.getLogger(__name__)

        self.node_graph = node_graph
        self.dtype = dtype
        self.device = device
        self.directories = directories

        self.force_new_run = False
        self.node_graph.set_abort_function(self.on_aborted)

    def run(self):
        self.node_graph.dtype = self.dtype
        self.node_graph.device = self.device
        self.status_changed.emit("Generating image...")

        if self.force_new_run:
            node = self.node_graph.get_node_by_name("num_inference_steps")
            node.update_value(9)

            node = self.node_graph.get_node_by_name("guidance_scale")
            node.update_value(1.0)

            node = self.node_graph.get_node_by_name("model")
            node.update_model(
                path="/home/ozzy/Documents/Image Artisan Z/models/diffusers/Z-Image-Turbo/",
                model_name="Z-Image-Turbo",
                version="1.0",
                model_type="Turbo",
            )

            node = self.node_graph.get_node_by_name("scheduler")
            node.update_value(SchedulerDataObject())

            node = self.node_graph.get_node_by_name("denoise")
            node.callback = self.step_progress_update

            node = self.node_graph.get_node_by_name("image_send")
            node.image_callback = self.preview_image

            self.force_new_run = False

        try:
            self.node_graph()
        except IArtisanZNodeError as e:
            self.logger.debug(f"Error in node: '{e.node_name}': {e}")
            self.generation_error.emit(f"Error in node '{e.node_name}': {e}", False)

    def update_node(self, node_name: str, value):
        node = self.node_graph.get_node_by_name(node_name)
        node.update_value(value)

    def add_lora(self, lora_data):
        lora_node = self.node_graph.get_node_by_name(f"{lora_data.name}_{lora_data.version}_lora")

        if lora_node is not None:
            lora_node.update_lora(
                enabled=lora_data.enabled,
                transformer_weight=lora_data.transformer_weight,
                is_slider=lora_data.is_slider,
            )
            return

        lora_node = LoraNode(
            path=lora_data.path,
            adapter_name=lora_data.name,
            lora_name=lora_data.name,
            version=lora_data.version,
            transformer_weight=1.0,
            is_slider=False,
        )
        lora_node.connect("transformer", self.node_graph.get_node_by_name("model"), "transformer")
        self.node_graph.add_node(lora_node, f"{lora_data.name}_{lora_data.version}_lora")

        denoise = self.node_graph.get_node_by_name("denoise")
        denoise.connect("lora", lora_node, "lora")

    def remove_lora(self, lora_data):
        lora_node = None
        for node in self.node_graph.nodes:
            if isinstance(node, LoraNode):
                if node.lora_name == lora_data.name and node.version == lora_data.version:
                    lora_node = node
                    break

        if lora_node is not None:
            self.node_graph.delete_node(lora_node)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self.generation_finished.emit(image)

    def abort_graph(self):
        self.node_graph.abort_graph()

    def on_aborted(self):
        self.generation_aborted.emit()
