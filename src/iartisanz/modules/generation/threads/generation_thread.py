from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes import NODE_CLASSES
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode
from iartisanz.modules.generation.graph.nodes.number_node import NumberNode


if TYPE_CHECKING:
    from iartisanz.app.directories import DirectoriesObject
    from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph

logger = logging.getLogger(__name__)


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

        self.node_graph = node_graph
        self.dtype = dtype
        self.device = device
        self.directories = directories

        self.force_new_run = False
        self.node_graph.set_abort_function(self.on_aborted)
        self.loaded_node_graph = None

    def run(self):
        self.node_graph.dtype = self.dtype
        self.node_graph.device = self.device
        self.status_changed.emit("Generating image...")

        if self.force_new_run:
            node = self.node_graph.get_node_by_name("model")
            node.update_model(
                path="/home/ozzy/ImageArtisanZ/models/diffusers/Z-Image-Turbo/",
                model_name="Z-Image-Turbo",
                version="1.0",
                model_type="Turbo",
            )

            node = self.node_graph.get_node_by_name("denoise")
            node.callback = self.step_progress_update

            node = self.node_graph.get_node_by_name("image_send")
            node.image_callback = self.preview_image

            self.force_new_run = False

        try:
            self.node_graph()
        except IArtisanZNodeError as e:
            logger.debug(f"Error in node: '{e.node_name}': {e}")
            self.generation_error.emit(f"Error in node '{e.node_name}': {e}", False)

        if not self.node_graph.updated:
            self.generation_error.emit("Nothing was changed", False)

    def update_node(self, node_name: str, value) -> bool:
        node = self.node_graph.get_node_by_name(node_name)
        if node is None:
            logger.debug("update_node: unknown node '%s' (ignored)", node_name)
            return False

        node.update_value(value)
        return True

    def update_nodes(self, values: dict) -> dict:
        return {name: self.update_node(name, value) for name, value in values.items()}

    def load_json_graph(self, json_graph: str, callbacks: dict | None = None):
        if callbacks is None:
            callbacks = {"preview_image": self.preview_image}

        self.node_graph.from_json(json_graph, node_classes=NODE_CLASSES, callbacks=callbacks)

        # Wire denoise step callback
        node = self.node_graph.get_node_by_name("denoise")
        if node is not None:
            node.callback = self.step_progress_update

        # Wire image_send node to preview images
        image_send = self.node_graph.get_node_by_name("image_send")
        image_send.image_callback = self.preview_image

    def add_lora(self, lora_data):
        lora_node = self.node_graph.get_node_by_name(f"{lora_data.name}_{lora_data.version}_lora")

        if lora_node is not None:
            return

        lora_node = LoraNode(
            path=lora_data.path,
            adapter_name=lora_data.name,
            lora_name=lora_data.name,
            version=lora_data.version,
            transformer_weight=1.0,
            is_slider=False,
            database_id=lora_data.id,
        )
        lora_node.connect("transformer", self.node_graph.get_node_by_name("model"), "transformer")
        self.node_graph.add_node(lora_node, f"{lora_data.name}_{lora_data.version}_lora")

        denoise = self.node_graph.get_node_by_name("denoise")
        denoise.connect("lora", lora_node, "lora")

    def update_lora_weight(self, lora_data, weight: float):
        lora_node = self.node_graph.get_node_by_name(f"{lora_data.name}_{lora_data.version}_lora")
        if lora_node is not None:
            lora_node.update_lora_weight(weight)

    def update_lora_enabled(self, lora_node_name: str, enabled: bool):
        lora_node = self.node_graph.get_node_by_name(lora_node_name)
        if lora_node is not None:
            lora_node.update_lora_enabled(enabled)

    def remove_lora(self, lora_data):
        lora_node = None
        for node in self.node_graph.nodes:
            if isinstance(node, LoraNode):
                if node.lora_name == lora_data.name and node.version == lora_data.version:
                    lora_node = node
                    break

        if lora_node is not None:
            self.node_graph.delete_node(lora_node)

    def add_source_image(self, source_image_path: str, strength: float):
        source_image_node = ImageLoadNode(path=source_image_path)
        self.node_graph.add_node(source_image_node, "source_image")

        strength = NumberNode(number=strength)
        self.node_graph.add_node(strength, "strength")

        latents_node = self.node_graph.get_node_by_name("latents")
        models_node = self.node_graph.get_node_by_name("model")
        denoise_node = self.node_graph.get_node_by_name("denoise")

        latents_node.connect("image", source_image_node, "image")
        latents_node.connect("vae", models_node, "vae")
        denoise_node.connect("noise", latents_node, "noise")
        denoise_node.connect("strength", strength, "value")

    def update_source_image(self, source_image_path: str):
        source_image_node = self.node_graph.get_node_by_name("source_image")
        source_image_node.update_value(source_image_path)

    def update_strength(self, strength: float):
        strength_node = self.node_graph.get_node_by_name("strength")
        if strength_node is not None:
            strength_node.update_value(strength)

    def enable_source_image(self, enabled: bool):
        source_image_node = self.node_graph.get_node_by_name("source_image")
        strength_node = self.node_graph.get_node_by_name("strength")
        latents_node = self.node_graph.get_node_by_name("latents")
        models_node = self.node_graph.get_node_by_name("model")
        denoise_node = self.node_graph.get_node_by_name("denoise")

        if enabled:
            source_image_node.enabled = True
            strength_node.enabled = True
            latents_node.connect("image", source_image_node, "image")
            latents_node.connect("vae", models_node, "vae")
            denoise_node.connect("noise", latents_node, "noise")
            denoise_node.connect("strength", strength_node, "value")
        else:
            source_image_node.enabled = False
            strength_node.enabled = False
            latents_node.disconnect("image", source_image_node, "image")
            latents_node.disconnect("vae", models_node, "vae")
            denoise_node.disconnect("noise", latents_node, "noise")
            denoise_node.disconnect("strength", strength_node, "value")

    def remove_source_image(self):
        latents_node = self.node_graph.get_node_by_name("latents")
        models_node = self.node_graph.get_node_by_name("model")
        denoise_node = self.node_graph.get_node_by_name("denoise")

        latents_node.disconnect("vae", models_node, "vae")
        denoise_node.disconnect("noise", latents_node, "noise")

        self.node_graph.delete_node_by_name("source_image")
        self.node_graph.delete_node_by_name("strength")

    def add_source_image_mask(self, mask_image_path: str):
        mask_image_node = ImageLoadNode(path=mask_image_path, grayscale=True)
        self.node_graph.add_node(mask_image_node, "source_image_mask")

        denoise_node = self.node_graph.get_node_by_name("denoise")
        denoise_node.connect("image_mask", mask_image_node, "image")

    def update_source_image_mask(self, mask_image_path: str):
        mask_image_node = self.node_graph.get_node_by_name("source_image_mask")
        mask_image_node.update_value(mask_image_path)

    def remove_source_image_mask(self):
        self.node_graph.delete_node_by_name("source_image_mask")

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        self.generation_finished.emit(image)

    def abort_graph(self):
        self.node_graph.abort_graph()

    def on_aborted(self):
        self.generation_aborted.emit()
