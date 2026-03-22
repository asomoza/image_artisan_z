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
from iartisanz.modules.generation.graph.nodes.choice_node import ChoiceNode
from iartisanz.modules.generation.graph.nodes.controlnet_conditioning_node import ControlNetConditioningNode
from iartisanz.modules.generation.graph.nodes.controlnet_model_node import ControlNetModelNode
from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode
from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode
from iartisanz.modules.generation.graph.nodes.number_node import NumberNode
from iartisanz.modules.generation.graph.nodes.number_range_node import NumberRangeNode


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
        # against a persistent run graph that is updated from a JSON snapshot.
        # Using update_from_json() (smart diffing) means only nodes whose inputs
        # actually changed are re-executed, avoiding unnecessary work like
        # re-encoding an unchanged prompt (which would load the text encoder to GPU).
        self.node_graph.set_abort_function(self.on_aborted)
        self._job_json_graph: str | None = None
        self._active_graph: ImageArtisanZNodeGraph | None = None
        self._persistent_run_graph: ImageArtisanZNodeGraph | None = None
        self._completion_emitted: bool = False

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
        callbacks = {"preview_image": self.preview_image}

        if self._persistent_run_graph is not None:
            self._persistent_run_graph.update_from_json(
                json_graph, node_classes=self._node_classes, callbacks=callbacks
            )
        else:
            self._persistent_run_graph = self._graph_factory()
            self._persistent_run_graph.set_abort_function(self.on_aborted)
            self._persistent_run_graph.from_json(
                json_graph, node_classes=self._node_classes, callbacks=callbacks
            )

        self._wire_callbacks(self._persistent_run_graph)
        return self._persistent_run_graph

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
        self._completion_emitted = False

        json_graph = self._job_json_graph or self.get_staged_json_graph()

        # Keep LoRA adapter set in sync with the active snapshot to avoid
        # accumulating adapters in VRAM across runs.
        self._prune_transformer_loras(self._extract_required_loras(json_graph))

        run_graph = self._create_run_graph_from_json(json_graph)
        run_graph.dtype = self.dtype
        run_graph.device = self.device
        self._active_graph = run_graph

        try:
            # Validate controlnet inpainting configuration before execution
            run_graph.validate_controlnet_inpainting()
            run_graph()
        except ValueError as e:
            # Configuration validation errors
            logger.debug(f"Configuration error: {e}")
            self._completion_emitted = True
            self.generation_error.emit(str(e), False)
        except IArtisanZNodeError as e:
            logger.debug(f"Error in node: '{e.node_name}': {e}")
            self._completion_emitted = True
            self.generation_error.emit(f"Error in node '{e.node_name}': {e}", False)
        finally:
            self._active_graph = None

        # If the graph ran but produced no output (e.g. all nodes were no-ops due
        # to unchanged inputs), release the UI — treat it as an abort.
        if not self._completion_emitted:
            self.generation_aborted.emit()

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

        self._persistent_run_graph = None
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
        # Clean up persistent run graph — nodes may hold state (e.g. LoKr weight
        # merges) that must be reverted via before_delete() before dropping.
        if self._persistent_run_graph is not None:
            self._persistent_run_graph.clean_up()
            self._persistent_run_graph = None

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
            db_model_id=model_data_object.id if model_data_object.id else None,
        )

    def _ensure_freefuse_wiring(self):
        """Ensure positive_prompt_text is wired to denoise for FreeFuse.

        Old graphs created before FreeFuse was added lack this connection.
        This is a no-op if the connection already exists.
        """
        denoise = self.node_graph.get_node_by_name("denoise")
        if denoise is None:
            return
        if "positive_prompt_text" not in denoise.OPTIONAL_INPUTS:
            return
        if denoise.connections.get("positive_prompt_text"):
            return
        positive_prompt = self.node_graph.get_node_by_name("positive_prompt")
        if positive_prompt is None:
            return
        denoise.connect("positive_prompt_text", positive_prompt, "value")

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
            spatial_mask_enabled=lora_data.spatial_mask_enabled,
            spatial_mask_path=lora_data.spatial_mask_path,
            trigger_words=lora_data.trigger_words,
        )
        lora_node.lora_enabled = lora_data.enabled
        lora_node.connect("transformer", self.node_graph.get_node_by_name("model"), "transformer")
        self.node_graph.add_node(lora_node, lora_data.lora_node_name)

        denoise = self.node_graph.get_node_by_name("denoise")
        denoise.connect("lora", lora_node, "lora")

        self._ensure_freefuse_wiring()

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

    def update_lora_spatial_mask(self, lora_data: LoraDataObject):
        """Update the spatial mask settings for a LoRA.

        Args:
            lora_data: LoRA data object with updated mask settings
        """
        lora_node = self.node_graph.get_node_by_name(lora_data.lora_node_name)
        if lora_node is not None:
            lora_node.update_spatial_mask(
                enabled=lora_data.spatial_mask_enabled,
                path=lora_data.spatial_mask_path,
            )

    def update_lora_trigger_words(self, lora_data: LoraDataObject):
        lora_node = self.node_graph.get_node_by_name(lora_data.lora_node_name)
        if lora_node is not None:
            lora_node.update_trigger_words(lora_data.trigger_words)
            if lora_data.trigger_words:
                self._ensure_freefuse_wiring()

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
        denoise_node.connect("source_mask", mask_image_node, "image")

        # Set differential_diffusion_active flag to True when mask is added
        diff_diff_flag = self.node_graph.get_node_by_name("differential_diffusion_active")
        if diff_diff_flag is not None:
            diff_diff_flag.update_value(True)

    def update_source_image_mask(self, mask_image_path: str):
        mask_image_node = self.node_graph.get_node_by_name("source_image_mask")
        mask_image_node.update_value(mask_image_path)

    def remove_source_image_mask(self):
        self.node_graph.delete_node_by_name("source_image_mask")

        # Set differential_diffusion_active flag to False when mask is removed
        diff_diff_flag = self.node_graph.get_node_by_name("differential_diffusion_active")
        if diff_diff_flag is not None:
            diff_diff_flag.update_value(False)

    def add_controlnet_mask_image(self, mask_image_path: str, controlnet_path: str | None = None):
        """Add a separate mask image for ControlNet inpaint conditioning.

        This is intentionally independent from the regular differential diffusion
        mask used by the denoise node.

        Args:
            mask_image_path: Path to the mask image
            controlnet_path: Optional path to ControlNet model. If provided and infrastructure
                           doesn't exist, it will be created to enable inpainting-only mode.
        """

        conditioning_node = self.node_graph.get_node_by_name("controlnet_conditioning")

        # If conditioning node doesn't exist and we have a controlnet_path,
        # create the infrastructure to support inpainting-only mode
        if conditioning_node is None and controlnet_path is not None:
            self._ensure_controlnet_infrastructure(
                controlnet_path=controlnet_path,
                conditioning_scale=0.75,
                control_mode="balanced",
                prompt_decay=0.825,
            )
            conditioning_node = self.node_graph.get_node_by_name("controlnet_conditioning")

        # If still no conditioning node (no controlnet_path provided), return
        if conditioning_node is None:
            return

        if self.node_graph.get_node_by_name("control_mask_image") is not None:
            return

        mask_node = ImageLoadNode(path=mask_image_path, grayscale=True)
        self.node_graph.add_node(mask_node, "control_mask_image")

        try:
            conditioning_node.connect("mask_image", mask_node, "image")
        except Exception:
            pass

    def update_controlnet_mask_image(self, mask_image_path: str):
        mask_node = self.node_graph.get_node_by_name("control_mask_image")
        if mask_node is not None:
            mask_node.update_value(mask_image_path)

    def remove_controlnet_mask_image(self):
        conditioning_node = self.node_graph.get_node_by_name("controlnet_conditioning")
        mask_node = self.node_graph.get_node_by_name("control_mask_image")
        if conditioning_node is not None and mask_node is not None:
            try:
                conditioning_node.disconnect("mask_image", mask_node, "image")
            except Exception:
                pass
        self.node_graph.delete_node_by_name("control_mask_image")

    def _ensure_controlnet_infrastructure(
        self,
        controlnet_path: str,
        conditioning_scale: float = 0.75,
        control_mode: str = "balanced",
        prompt_decay: float = 0.825,
    ):
        """Create ControlNet infrastructure nodes if they don't exist.

        This creates all the common nodes needed for ControlNet (model, conditioning,
        scale, guidance, etc.) but NOT the control_image node, allowing for
        inpainting-only usage.
        """

        if self.node_graph.get_node_by_name("controlnet_model") is not None:
            return

        controlnet_model = ControlNetModelNode(path=controlnet_path)
        controlnet_model.enabled = True
        self.node_graph.add_node(controlnet_model, "controlnet_model")

        scale_node = NumberNode(number=float(conditioning_scale))
        scale_node.enabled = True
        self.node_graph.add_node(scale_node, "controlnet_conditioning_scale")

        control_mode_node = ChoiceNode(
            value=str(control_mode),
            choices=["balanced", "prompt", "controlnet"],
            default="balanced",
        )
        control_mode_node.enabled = True
        self.node_graph.add_node(control_mode_node, "controlnet_control_mode")

        prompt_decay_node = NumberNode(number=float(prompt_decay))
        prompt_decay_node.enabled = True
        self.node_graph.add_node(prompt_decay_node, "controlnet_prompt_decay")

        conditioning = ControlNetConditioningNode()
        conditioning.enabled = True

        control_guidance = NumberRangeNode(value=[0.0, 1.0])
        control_guidance.enabled = True
        self.node_graph.add_node(control_guidance, "control_guidance_start_end")

        models_node = self.node_graph.get_node_by_name("model")
        width_node = self.node_graph.get_node_by_name("image_width")
        height_node = self.node_graph.get_node_by_name("image_height")
        denoise_node = self.node_graph.get_node_by_name("denoise")
        latents_node = self.node_graph.get_node_by_name("latents")

        # ControlNet weights depend on the transformer for shared embeddings.
        controlnet_model.connect("transformer", models_node, "transformer")

        conditioning.connect("vae", models_node, "vae")
        conditioning.connect("vae_scale_factor", models_node, "vae_scale_factor")
        conditioning.connect("width", width_node, "value")
        conditioning.connect("height", height_node, "value")

        # NEW: Connect source_image from LatentsNode to ControlNet as init_image
        # This allows ControlNet to use the source_image for inpainting
        if latents_node is not None:
            conditioning.connect("init_image", latents_node, "source_image")

        # NEW: Create or reuse differential_diffusion_active flag
        # This will be set to True when source_image_mask exists
        from iartisanz.modules.generation.graph.nodes.boolean_node import BooleanNode

        diff_diff_flag = self.node_graph.get_node_by_name("differential_diffusion_active")
        if diff_diff_flag is None:
            diff_diff_flag = BooleanNode(value=False)
            diff_diff_flag.enabled = True
            self.node_graph.add_node(diff_diff_flag, "differential_diffusion_active")
        else:
            diff_diff_flag.enabled = True

        # Initialize flag based on whether a source image mask is already present.
        has_source_mask = self.node_graph.get_node_by_name("source_image_mask") is not None
        diff_diff_flag.update_value(bool(has_source_mask))

        conditioning.connect("differential_diffusion_active", diff_diff_flag, "value")

        self.node_graph.add_node(conditioning, "controlnet_conditioning")

        denoise_node.connect("controlnet", controlnet_model, "controlnet")
        denoise_node.connect("control_image_latents", conditioning, "control_image_latents")
        denoise_node.connect("controlnet_conditioning_scale", scale_node, "value")
        denoise_node.connect("control_guidance_start_end", control_guidance, "value")

        # NEW: Connect spatial_mask from ControlNet conditioning to DenoiseNode
        denoise_node.connect("controlnet_spatial_mask", conditioning, "spatial_mask")

        # ControlNet runtime behavior controls (consumed by DenoiseNode).
        denoise_node.connect("control_mode", control_mode_node, "value")
        denoise_node.connect("prompt_mode_decay", prompt_decay_node, "value")

    def add_controlnet(
        self,
        *,
        controlnet_path: str,
        control_image_path: str,
        conditioning_scale: float = 0.75,
        control_mode: str = "balanced",
        prompt_decay: float = 0.825,
    ):
        """Add ControlNet nodes to the staged graph and wire them into denoising.

        Not called by default. The GUI will call this later when the user opts in.
        """

        # Create infrastructure if it doesn't exist
        self._ensure_controlnet_infrastructure(controlnet_path, conditioning_scale, control_mode, prompt_decay)

        # Add control image node if it doesn't exist
        if self.node_graph.get_node_by_name("control_image") is not None:
            return

        from iartisanz.modules.generation.graph.nodes.image_load_node import ImageLoadNode

        control_image = ImageLoadNode(path=control_image_path)
        self.node_graph.add_node(control_image, "control_image")

        # Connect control image to conditioning node
        conditioning_node = self.node_graph.get_node_by_name("controlnet_conditioning")
        if conditioning_node is not None:
            conditioning_node.connect("control_image", control_image, "image")

    def update_controlnet(self, *, controlnet_path: str | None = None, control_image_path: str | None = None):
        model_node = self.node_graph.get_node_by_name("controlnet_model")
        if model_node is not None and controlnet_path is not None:
            model_node.update_value(controlnet_path)

        image_node = self.node_graph.get_node_by_name("control_image")
        if image_node is not None and control_image_path is not None:
            image_node.update_value(control_image_path)

    def update_controlnet_conditioning_scale(self, conditioning_scale: float):
        scale_node = self.node_graph.get_node_by_name("controlnet_conditioning_scale")
        if scale_node is not None:
            scale_node.update_value(float(conditioning_scale))

    def enable_controlnet(self, enabled: bool):
        model_node = self.node_graph.get_node_by_name("controlnet_model")
        image_node = self.node_graph.get_node_by_name("control_image")
        conditioning_node = self.node_graph.get_node_by_name("controlnet_conditioning")
        scale_node = self.node_graph.get_node_by_name("controlnet_conditioning_scale")
        guidance_node = self.node_graph.get_node_by_name("control_guidance_start_end")
        control_mode_node = self.node_graph.get_node_by_name("controlnet_control_mode")
        prompt_decay_node = self.node_graph.get_node_by_name("controlnet_prompt_decay")
        denoise_node = self.node_graph.get_node_by_name("denoise")

        # Control image is optional (for inpainting-only mode)
        if None in (
            model_node,
            conditioning_node,
            scale_node,
            guidance_node,
            control_mode_node,
            prompt_decay_node,
            denoise_node,
        ):
            return

        model_node.enabled = bool(enabled)
        if image_node is not None:
            image_node.enabled = bool(enabled)
        conditioning_node.enabled = bool(enabled)
        scale_node.enabled = bool(enabled)
        guidance_node.enabled = bool(enabled)
        control_mode_node.enabled = bool(enabled)
        prompt_decay_node.enabled = bool(enabled)

        if enabled:
            if not self._has_connection(denoise_node, "controlnet", model_node, "controlnet"):
                denoise_node.connect("controlnet", model_node, "controlnet")
            if not self._has_connection(
                denoise_node, "control_image_latents", conditioning_node, "control_image_latents"
            ):
                denoise_node.connect("control_image_latents", conditioning_node, "control_image_latents")
            if not self._has_connection(denoise_node, "controlnet_conditioning_scale", scale_node, "value"):
                denoise_node.connect("controlnet_conditioning_scale", scale_node, "value")
            if not self._has_connection(denoise_node, "control_guidance_start_end", guidance_node, "value"):
                denoise_node.connect("control_guidance_start_end", guidance_node, "value")

            if not self._has_connection(denoise_node, "control_mode", control_mode_node, "value"):
                denoise_node.connect("control_mode", control_mode_node, "value")
            if not self._has_connection(denoise_node, "prompt_mode_decay", prompt_decay_node, "value"):
                denoise_node.connect("prompt_mode_decay", prompt_decay_node, "value")
            if not self._has_connection(denoise_node, "controlnet_spatial_mask", conditioning_node, "spatial_mask"):
                denoise_node.connect("controlnet_spatial_mask", conditioning_node, "spatial_mask")
        else:
            denoise_node.disconnect("controlnet", model_node, "controlnet")
            denoise_node.disconnect("control_image_latents", conditioning_node, "control_image_latents")
            denoise_node.disconnect("controlnet_conditioning_scale", scale_node, "value")
            denoise_node.disconnect("control_guidance_start_end", guidance_node, "value")
            denoise_node.disconnect("control_mode", control_mode_node, "value")
            denoise_node.disconnect("prompt_mode_decay", prompt_decay_node, "value")
            denoise_node.disconnect("controlnet_spatial_mask", conditioning_node, "spatial_mask")

    @staticmethod
    def _has_connection(node, input_name: str, source_node, output_name: str) -> bool:
        for connected_node, connected_output in node.connections.get(input_name, []):
            if connected_node is source_node and connected_output == output_name:
                return True
        return False

    # ──────────────────────────────────────────────────────────
    # Edit images (Flux2 Klein edit/inpaint conditioning)
    # ──────────────────────────────────────────────────────────

    def add_edit_image(self, image_index: int, image_path: str):
        """Add an edit image node and wire it to the encode node."""
        node_name = f"edit_image_{image_index}"
        if self.node_graph.get_node_by_name(node_name) is not None:
            return

        image_node = ImageLoadNode(path=image_path)
        self.node_graph.add_node(image_node, node_name)

        self._ensure_edit_image_encoder()

        encode_node = self.node_graph.get_node_by_name("edit_image_encode")
        encode_node.connect(f"image_{image_index}", image_node, "image")

    def update_edit_image(self, image_index: int, image_path: str):
        node_name = f"edit_image_{image_index}"
        node = self.node_graph.get_node_by_name(node_name)
        if node is not None:
            node.update_value(image_path)

    def remove_edit_image(self, image_index: int):
        node_name = f"edit_image_{image_index}"
        encode_node = self.node_graph.get_node_by_name("edit_image_encode")
        image_node = self.node_graph.get_node_by_name(node_name)
        if encode_node is not None and image_node is not None:
            encode_node.disconnect(f"image_{image_index}", image_node, "image")
        self.node_graph.delete_node_by_name(node_name)

        if not self._has_any_edit_images():
            self._remove_edit_image_encoder()

    def remove_all_edit_images(self):
        for i in range(4):
            self.remove_edit_image(i)

    def _ensure_edit_image_encoder(self):
        if self.node_graph.get_node_by_name("edit_image_encode") is not None:
            return

        from iartisanz.modules.generation.graph.nodes.flux2_edit_image_encode_node import Flux2EditImageEncodeNode

        encode_node = Flux2EditImageEncodeNode()
        models_node = self.node_graph.get_node_by_name("model")
        encode_node.connect("vae", models_node, "vae")
        encode_node.connect("vae_scale_factor", models_node, "vae_scale_factor")
        self.node_graph.add_node(encode_node, "edit_image_encode")

        denoise = self.node_graph.get_node_by_name("denoise")
        denoise.connect("edit_image_latents", encode_node, "image_latents")
        denoise.connect("edit_image_latent_ids", encode_node, "image_latent_ids")

    def _remove_edit_image_encoder(self):
        denoise = self.node_graph.get_node_by_name("denoise")
        encode_node = self.node_graph.get_node_by_name("edit_image_encode")
        if denoise is not None and encode_node is not None:
            denoise.disconnect("edit_image_latents", encode_node, "image_latents")
            denoise.disconnect("edit_image_latent_ids", encode_node, "image_latent_ids")
        self.node_graph.delete_node_by_name("edit_image_encode")

    def _has_any_edit_images(self) -> bool:
        return any(
            self.node_graph.get_node_by_name(f"edit_image_{i}") is not None
            for i in range(4)
        )

    def remove_controlnet(self):
        denoise_node = self.node_graph.get_node_by_name("denoise")
        model_node = self.node_graph.get_node_by_name("controlnet_model")
        conditioning_node = self.node_graph.get_node_by_name("controlnet_conditioning")
        scale_node = self.node_graph.get_node_by_name("controlnet_conditioning_scale")
        guidance_node = self.node_graph.get_node_by_name("control_guidance_start_end")
        control_mode_node = self.node_graph.get_node_by_name("controlnet_control_mode")
        prompt_decay_node = self.node_graph.get_node_by_name("controlnet_prompt_decay")

        if denoise_node is not None and model_node is not None:
            denoise_node.disconnect("controlnet", model_node, "controlnet")
        if denoise_node is not None and conditioning_node is not None:
            denoise_node.disconnect("control_image_latents", conditioning_node, "control_image_latents")
            denoise_node.disconnect("controlnet_spatial_mask", conditioning_node, "spatial_mask")
        if denoise_node is not None and scale_node is not None:
            denoise_node.disconnect("controlnet_conditioning_scale", scale_node, "value")
        if denoise_node is not None and guidance_node is not None:
            denoise_node.disconnect("control_guidance_start_end", guidance_node, "value")
        if denoise_node is not None and control_mode_node is not None:
            denoise_node.disconnect("control_mode", control_mode_node, "value")
        if denoise_node is not None and prompt_decay_node is not None:
            denoise_node.disconnect("prompt_mode_decay", prompt_decay_node, "value")

        control_init_image_node = self.node_graph.get_node_by_name("control_init_image")
        if conditioning_node is not None and control_init_image_node is not None:
            try:
                conditioning_node.disconnect("init_image", control_init_image_node, "image")
            except Exception:
                pass

        control_mask_image_node = self.node_graph.get_node_by_name("control_mask_image")
        if conditioning_node is not None and control_mask_image_node is not None:
            try:
                conditioning_node.disconnect("mask_image", control_mask_image_node, "image")
            except Exception:
                pass

        self.node_graph.delete_node_by_name("controlnet_conditioning")
        self.node_graph.delete_node_by_name("controlnet_conditioning_scale")
        self.node_graph.delete_node_by_name("control_guidance_start_end")
        self.node_graph.delete_node_by_name("controlnet_control_mode")
        self.node_graph.delete_node_by_name("controlnet_prompt_decay")
        self.node_graph.delete_node_by_name("control_image")
        self.node_graph.delete_node_by_name("controlnet_model")

        # Optional inpaint inputs for ControlNet.
        self.node_graph.delete_node_by_name("control_init_image")
        self.node_graph.delete_node_by_name("control_mask_image")

        # Cleanup differential diffusion flag created for ControlNet.
        self.node_graph.delete_node_by_name("differential_diffusion_active")

    def step_progress_update(self, step, _timestep, latents):
        self.progress_update.emit(step, latents)

    def preview_image(self, image):
        duration = None
        if self._active_graph is not None:
            denoise_node = self._active_graph.get_node_by_name("denoise")
            duration = getattr(denoise_node, "elapsed_time", None) if denoise_node is not None else None
        self._completion_emitted = True
        self.generation_finished.emit(image, duration)

    def abort_graph(self):
        if self._active_graph is not None:
            self._active_graph.abort_graph()
        elif self.node_graph is not None:
            self.node_graph.abort_graph()

    def on_aborted(self):
        self._completion_emitted = True
        self.generation_aborted.emit()
