from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from PyQt6.QtCore import QThread, pyqtSignal

from iartisanz.app.model_manager import get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph
from iartisanz.modules.generation.graph.nodes import NODE_CLASSES
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode
from iartisanz.modules.generation.graph.nodes.number_node import NumberNode


if TYPE_CHECKING:
    from iartisanz.app.directories import DirectoriesObject
    from iartisanz.modules.generation.data_objects.lora_data_object import LoraDataObject
    from iartisanz.modules.generation.data_objects.model_data_object import ModelDataObject
    from iartisanz.modules.generation.graph.iartisanz_node_graph import ImageArtisanZNodeGraph

logger = logging.getLogger(__name__)


class NodeGraphThread(QThread):
    status_changed = pyqtSignal(str)
    progress_update = pyqtSignal(int, torch.Tensor)
    # Emits (image: np.ndarray, duration_seconds: float | None)
    generation_finished = pyqtSignal(object, object)
    generation_error = pyqtSignal(str, bool)
    generation_aborted = pyqtSignal()

    def __init__(
        self,
        directories: DirectoriesObject,
        node_graph: ImageArtisanZNodeGraph,
        dtype: torch.dtype,
        device: torch.device,
        *,
        graph_factory: Callable[[], ImageArtisanZNodeGraph] | None = None,
        node_classes: dict | None = None,
    ):
        super().__init__()

        self.node_graph = node_graph
        self.dtype = dtype
        self.device = device
        self.directories = directories

        # Staged graph is edited by the UI thread. Each generation run executes
        # a fresh graph instance created from a JSON snapshot.
        self.node_graph.set_abort_function(self.on_aborted)
        self._job_json_graph: str | None = None
        self._active_graph: ImageArtisanZNodeGraph | None = None

        self._graph_factory = graph_factory or ImageArtisanZNodeGraph
        self._node_classes = node_classes or NODE_CLASSES

    def get_staged_json_graph(self) -> str:
        return self.node_graph.to_json()

    def start_generation(self, json_graph: str) -> None:
        # Snapshot is captured in the UI thread and treated as immutable for this run.
        self._job_json_graph = json_graph
        self.start()

    def _wire_callbacks(self, graph: ImageArtisanZNodeGraph) -> None:
        node = graph.get_node_by_name("denoise")
        if node is not None:
            node.callback = self.step_progress_update

        image_send = graph.get_node_by_name("image_send")
        if image_send is not None:
            image_send.image_callback = self.preview_image

    def _create_run_graph_from_json(self, json_graph: str) -> ImageArtisanZNodeGraph:
        run_graph = self._graph_factory()
        run_graph.set_abort_function(self.on_aborted)
        run_graph.from_json(
            json_graph, node_classes=self._node_classes, callbacks={"preview_image": self.preview_image}
        )
        self._wire_callbacks(run_graph)
        return run_graph

    def _extract_required_loras(self, json_graph: str) -> dict[str, str | None]:
        """Return adapter_name -> path for LoRAs that should be active for this run."""
        try:
            payload = json.loads(json_graph)
        except Exception:
            return {}

        required: dict[str, str | None] = {}
        for node_dict in payload.get("nodes", []) or []:
            if node_dict.get("class") != "LoraNode":
                continue

            if node_dict.get("enabled", True) is False:
                continue

            state = node_dict.get("state", {}) or {}
            if state.get("lora_enabled", True) is False:
                continue

            adapter_name = state.get("adapter_name") or node_dict.get("adapter_name")
            if not adapter_name:
                continue

            path = state.get("path") or node_dict.get("path")
            required[str(adapter_name)] = path

        return required

    def _prune_transformer_loras(self, required: dict[str, str | None]) -> None:
        mm = get_model_manager()
        if not mm.has("transformer"):
            return

        try:
            transformer = mm.get_raw("transformer")
        except Exception:
            return

        loaded = set(getattr(transformer, "peft_config", {}) or {})
        required_names = set(required.keys())

        # Remove adapters that are no longer referenced by the current snapshot.
        stale = sorted(loaded - required_names)
        if stale:
            try:
                transformer.delete_adapters(stale)
            except Exception:
                pass
            for name in stale:
                mm.clear_lora_source(name)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # If an adapter_name is reused for a different file, force reload.
        for name in sorted(loaded & required_names):
            desired_path = required.get(name)
            if desired_path is None:
                continue
            current_path = mm.get_lora_source(name)
            if current_path is not None and current_path != desired_path:
                try:
                    transformer.delete_adapters([name])
                except Exception:
                    pass
                mm.clear_lora_source(name)
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass

    def run(self):
        self.status_changed.emit("Generating image...")

        json_graph = self._job_json_graph or self.get_staged_json_graph()

        # Keep LoRA adapter set in sync with the active snapshot to avoid
        # accumulating adapters in VRAM across runs.
        self._prune_transformer_loras(self._extract_required_loras(json_graph))

        run_graph = self._create_run_graph_from_json(json_graph)
        run_graph.dtype = self.dtype
        run_graph.device = self.device
        self._active_graph = run_graph

        try:
            run_graph()
        except IArtisanZNodeError as e:
            logger.debug(f"Error in node: '{e.node_name}': {e}")
            self.generation_error.emit(f"Error in node '{e.node_name}': {e}", False)
        finally:
            self._active_graph = None

    def clean_up(self):
        if self._active_graph is not None:
            model_node = self._active_graph.get_node_by_name("model")
            if model_node is not None:
                model_node.clear_models()

        if self.node_graph is not None:
            model_node = self.node_graph.get_node_by_name("model")
            if model_node is not None:
                model_node.clear_models()

        # Safety: ensure global manager is cleared even if model node isn't present.
        get_model_manager().clear()

        self.node_graph = None
        self.dtype = None
        self.device = None

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
        # TODO: maybe check if the models are different than the loaded ones
        models_node = self.node_graph.get_node_by_name("model")
        incoming_model_sig = None
        try:
            payload = json.loads(json_graph)
            for node_dict in payload.get("nodes", []) or []:
                if node_dict.get("name") == "model":
                    incoming_model_sig = (
                        node_dict.get("model_name"),
                        node_dict.get("path"),
                        node_dict.get("version"),
                        node_dict.get("model_type"),
                    )
                    break
        except Exception:
            incoming_model_sig = None

        current_model_sig = None
        if models_node is not None:
            current_model_sig = (
                getattr(models_node, "model_name", None),
                getattr(models_node, "path", None),
                getattr(models_node, "version", None),
                getattr(models_node, "model_type", None),
            )

        if models_node is not None and incoming_model_sig is not None and incoming_model_sig != current_model_sig:
            models_node.clear_models()

        if callbacks is None:
            callbacks = {"preview_image": self.preview_image}

        self.node_graph.from_json(json_graph, node_classes=NODE_CLASSES, callbacks=callbacks)

        self._wire_callbacks(self.node_graph)

    def update_model(self, model_data_object: ModelDataObject):
        node = self.node_graph.get_node_by_name("model")

        node.update_model(
            path=model_data_object.filepath,
            model_name=model_data_object.name,
            version=model_data_object.version,
            model_type=model_data_object.model_type,
        )

    def add_lora(self, lora_data: LoraDataObject):
        lora_node = self.node_graph.get_node_by_name(lora_data.lora_node_name)

        if lora_node is not None:
            return

        lora_node = LoraNode(
            path=lora_data.path,
            adapter_name=lora_data.name,
            lora_name=lora_data.name,
            version=lora_data.version,
            transformer_weight=lora_data.transformer_weight,
            granular_transformer_weights_enabled=lora_data.granular_transformer_weights_enabled,
            transformer_granular_weights=lora_data.granular_transformer_weights,
            is_slider=lora_data.is_slider,
            database_id=lora_data.id,
        )
        lora_node.lora_enabled = lora_data.enabled
        lora_node.connect("transformer", self.node_graph.get_node_by_name("model"), "transformer")
        self.node_graph.add_node(lora_node, lora_data.lora_node_name)

        denoise = self.node_graph.get_node_by_name("denoise")
        denoise.connect("lora", lora_node, "lora")

    def update_lora_weights(self, lora_data: LoraDataObject):
        lora_node = self.node_graph.get_node_by_name(lora_data.lora_node_name)
        if lora_node is not None:
            lora_node.update_lora_weights(lora_data.transformer_weight, lora_data.granular_transformer_weights)

    def update_lora_enabled(self, lora_data: LoraDataObject):
        lora_node = self.node_graph.get_node_by_name(lora_data.lora_node_name)
        if lora_node is not None:
            lora_node.update_lora_enabled(lora_data.enabled)

    def update_lora_transformer_granular_enabled(self, lora_data: LoraDataObject):
        lora_node = self.node_graph.get_node_by_name(lora_data.lora_node_name)
        if lora_node is not None:
            lora_node.update_lora_transformer_granular_enabled(lora_data.granular_transformer_weights_enabled)

    def update_lora_slider_enabled(self, lora_data: LoraDataObject):
        lora_node = self.node_graph.get_node_by_name(lora_data.lora_node_name)
        if lora_node is not None:
            lora_node.update_slider_enabled(lora_data.is_slider)

    def remove_lora(self, lora_data: LoraDataObject):
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
        duration = None
        if self._active_graph is not None:
            denoise_node = self._active_graph.get_node_by_name("denoise")
            duration = getattr(denoise_node, "elapsed_time", None) if denoise_node is not None else None
        self.generation_finished.emit(image, duration)

    def abort_graph(self):
        if self._active_graph is not None:
            self._active_graph.abort_graph()
        elif self.node_graph is not None:
            self.node_graph.abort_graph()

    def on_aborted(self):
        self.generation_aborted.emit()
