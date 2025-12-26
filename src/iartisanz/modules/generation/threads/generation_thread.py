import logging

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.app.directories import DirectoriesObject
from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph


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
        self.seed = None
        self.positive_prompt = None
        self.negative_prompt = None
        self.image_width = 1024
        self.image_height = 1024

        self.node_graph.set_abort_function(self.on_aborted)

    def run(self):
        self.node_graph.dtype = self.dtype
        self.node_graph.device = self.device
        self.status_changed.emit("Generating image...")

        if self.force_new_run:
            node = self.node_graph.get_node_by_name("seed")
            node.update_value(self.seed)

            node = self.node_graph.get_node_by_name("image_width")
            node.update_value(self.image_width)

            node = self.node_graph.get_node_by_name("image_height")
            node.update_value(self.image_height)

            node = self.node_graph.get_node_by_name("num_inference_steps")
            node.update_value(9)

            node = self.node_graph.get_node_by_name("guidance_scale")
            node.update_value(1.0)

            node = self.node_graph.get_node_by_name("positive_prompt")
            node.update_value(self.positive_prompt)

            node = self.node_graph.get_node_by_name("negative_prompt")
            node.update_value(self.negative_prompt)

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
        else:
            node = self.node_graph.get_node_by_name("seed")
            node.update_value(self.seed)

        try:
            self.node_graph()
        except IArtisanZNodeError as e:
            self.logger.debug(f"Error in node: '{e.node_name}': {e}")
            self.generation_error.emit(f"Error in node '{e.node_name}': {e}", False)

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self.generation_finished.emit(image)

    def abort_graph(self):
        self.node_graph.abort_graph()

    def on_aborted(self):
        self.generation_aborted.emit()
