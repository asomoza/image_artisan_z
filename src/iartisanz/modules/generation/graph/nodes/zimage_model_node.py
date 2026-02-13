import gc
import logging
import os

import torch
from diffusers import AutoencoderKL, ZImageTransformer2DModel
from transformers import AutoTokenizer, Qwen2Tokenizer, Qwen3Model

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


logger = logging.getLogger(__name__)


class ZImageModelNode(Node):
    OUTPUTS = ["text_encoder", "tokenizer", "transformer", "vae", "num_channels_latents", "vae_scale_factor"]
    SERIALIZE_EXCLUDE = {"path", "model_name", "version", "model_type", "db_model_id"}

    def __init__(
        self,
        path: str = None,
        model_name: str = None,
        version: str = None,
        model_type: str = None,
        db_model_id: int = None,
    ):
        super().__init__()
        self.path = path
        self.model_name = model_name
        self.version = version
        self.model_type = model_type
        self.db_model_id = db_model_id

    def update_model(self, path: str, model_name: str, version: str, model_type: str, db_model_id: int = None):
        current_sig = (self.path, self.model_name, self.version, self.model_type)
        new_sig = (path, model_name, version, model_type)
        if current_sig != new_sig:
            self.clear_models()
        self.path = path
        self.model_name = model_name
        self.version = version
        self.model_type = model_type
        self.db_model_id = db_model_id
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        node_dict["model_name"] = self.model_name
        node_dict["version"] = self.version
        node_dict["model_type"] = self.model_type
        if self.db_model_id is not None:
            node_dict["db_model_id"] = self.db_model_id
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(ZImageModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        node.model_name = node_dict["model_name"]
        node.version = node_dict["version"]
        node.model_type = node_dict["model_type"]
        node.db_model_id = node_dict.get("db_model_id")
        return node

    def update_inputs(self, node_dict, callbacks=None):
        new_sig = (
            node_dict["path"],
            node_dict["model_name"],
            node_dict["version"],
            node_dict["model_type"],
        )
        current_sig = (self.path, self.model_name, self.version, self.model_type)
        if current_sig != new_sig:
            self.clear_models()
        self.path, self.model_name, self.version, self.model_type = new_sig
        self.db_model_id = node_dict.get("db_model_id", self.db_model_id)

    def _resolve_component_paths(self) -> dict[str, str] | None:
        """Resolve component paths from registry if db_model_id is available."""
        if self.db_model_id is None:
            return None

        try:
            from iartisanz.app.component_registry import ComponentRegistry
            from iartisanz.app.app import get_app_database_path, get_app_directories

            db_path = get_app_database_path()
            directories = get_app_directories()
            if db_path is None or directories is None:
                return None

            registry = ComponentRegistry(
                db_path,
                os.path.join(directories.models_diffusers, "_components"),
            )
            components = registry.get_model_components(self.db_model_id)
            if not components:
                return None

            return {comp_type: info.storage_path for comp_type, info in components.items()}
        except Exception:
            return None

    def _get_target_component_hashes(self) -> dict[str, str] | None:
        """Get target component hashes from registry for smart switching."""
        if self.db_model_id is None:
            return None

        try:
            from iartisanz.app.component_registry import ComponentRegistry
            from iartisanz.app.app import get_app_database_path, get_app_directories

            db_path = get_app_database_path()
            directories = get_app_directories()
            if db_path is None or directories is None:
                return None

            registry = ComponentRegistry(
                db_path,
                os.path.join(directories.models_diffusers, "_components"),
            )
            components = registry.get_model_components(self.db_model_id)
            if not components:
                return None

            return {comp_type: info.content_hash for comp_type, info in components.items()}
        except Exception:
            return None

    def __call__(self):
        mm = get_model_manager()
        model_id = self.model_name or self.path

        # Try smart switching via registry hashes
        target_hashes = self._get_target_component_hashes()
        registry_paths = self._resolve_component_paths()

        if target_hashes:
            return self._smart_load(mm, model_id, target_hashes, registry_paths)
        else:
            return self._legacy_load(mm, model_id)

    def _smart_load(self, mm, model_id, target_hashes, registry_paths):
        """Load model components, skipping any whose hash matches what's already loaded."""
        # Fast path: model already fully loaded (e.g. consecutive runs, or
        # transition from _legacy_load which doesn't set component hashes).
        if mm.is_active_model(model_id) and all(
            mm.has(c) for c in ("tokenizer", "text_encoder", "transformer", "vae")
        ):
            self._set_output_handles(mm)
            return self.values

        components_to_load: list[str] = []
        transformer_changed = False

        for comp_type in ("tokenizer", "text_encoder", "transformer", "vae"):
            target_hash = target_hashes.get(comp_type)
            if target_hash is None:
                components_to_load.append(comp_type)
                logger.debug("Smart switch: reloading '%s' (no target hash in registry)", comp_type)
                if comp_type == "transformer":
                    transformer_changed = True
                continue

            current_hash = mm.get_component_hash(comp_type)
            if current_hash == target_hash and mm.has(comp_type):
                logger.debug("Smart switch: keeping '%s' (hash match: %s)", comp_type, target_hash[:12])
            else:
                components_to_load.append(comp_type)
                logger.debug(
                    "Smart switch: reloading '%s' (current=%s, target=%s, loaded=%s)",
                    comp_type,
                    current_hash[:12] if current_hash else None,
                    target_hash[:12],
                    mm.has(comp_type),
                )
                if comp_type == "transformer":
                    transformer_changed = True

        if not components_to_load:
            # All components match — just update model_id and set handles
            mm._model_id = model_id
            self._set_output_handles(mm)
            return self.values

        # Clear components that need reloading
        for comp_type in components_to_load:
            mm.clear_component(comp_type)

        # If transformer changed, clear LoRA sources and compiled components
        if transformer_changed:
            with mm._lock:
                mm._lora_sources.clear()
                mm._compiled_components.clear()

        # Force garbage collection to free GPU memory from cleared components
        # before loading new ones. Without this, old models may still occupy
        # VRAM (due to reference cycles in torch.nn.Module) causing OOM.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Resolve paths: prefer registry, fall back to self.path
        paths = {}
        for comp_type in ("tokenizer", "text_encoder", "transformer", "vae"):
            if registry_paths and comp_type in registry_paths:
                paths[comp_type] = registry_paths[comp_type]
            else:
                paths[comp_type] = os.path.join(self.path, comp_type)

        # Load only the components that need loading
        if "tokenizer" in components_to_load:
            self._load_tokenizer(paths["tokenizer"])

        if "text_encoder" in components_to_load:
            self._load_text_encoder(paths["text_encoder"])

        if self.abort:
            return

        if "transformer" in components_to_load:
            self._load_transformer(paths["transformer"])

        if self.abort:
            return

        if "vae" in components_to_load:
            self._load_vae(paths["vae"])

        if self.abort:
            return

        # Update hashes for newly loaded components
        for comp_type in components_to_load:
            if comp_type in target_hashes:
                mm.set_component_hash(comp_type, target_hashes[comp_type])

        mm._model_id = model_id
        self._set_output_handles(mm)
        return self.values

    def _legacy_load(self, mm, model_id):
        """Original loading path for models without registry data."""
        if mm.is_active_model(model_id) and all(
            mm.has(c) for c in ("tokenizer", "text_encoder", "transformer", "vae")
        ):
            self._set_output_handles(mm)
            return self.values

        # Clear everything before a full reload to avoid double memory usage
        mm.clear()

        self._load_tokenizer(os.path.join(self.path, "tokenizer"))
        self._load_text_encoder(os.path.join(self.path, "text_encoder"))

        if self.abort:
            return

        self._load_transformer(os.path.join(self.path, "transformer"))

        if self.abort:
            return

        self._load_vae(os.path.join(self.path, "vae"))

        if self.abort:
            return

        mm.register_active_model(
            model_id=model_id,
            tokenizer=mm.get_raw("tokenizer"),
            text_encoder=mm.get_raw("text_encoder"),
            transformer=mm.get_raw("transformer"),
            vae=mm.get_raw("vae"),
        )

        self._set_output_handles(mm)
        return self.values

    def _load_tokenizer(self, tokenizer_path: str):
        mm = get_model_manager()
        try:
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                    use_fast=True,
                    extra_special_tokens={},
                )
            except Exception:
                tokenizer = Qwen2Tokenizer.from_pretrained(
                    tokenizer_path,
                    local_files_only=True,
                )
            if tokenizer is None:
                raise IArtisanZNodeError(
                    "Error trying to load the tokenizer, probably the file doesn't exists.", self.name
                )
            mm.register_component("tokenizer", tokenizer)
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the tokenizer: {e}", self.name) from e

    def _load_text_encoder(self, text_encoder_path: str):
        mm = get_model_manager()
        try:
            text_encoder = Qwen3Model.from_pretrained(
                text_encoder_path,
                use_safetensors=True,
                dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            mm.register_component("text_encoder", text_encoder)
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the text encoder: {e}", self.name) from e

    def _load_transformer(self, transformer_path: str):
        mm = get_model_manager()
        try:
            transformer = ZImageTransformer2DModel.from_pretrained(
                transformer_path,
                use_safetensors=True,
                torch_dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            mm.register_component("transformer", transformer)
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the transformer: {e}", self.name) from e

    def _load_vae(self, vae_path: str):
        mm = get_model_manager()
        try:
            vae = AutoencoderKL.from_pretrained(
                vae_path,
                use_safetensors=True,
                torch_dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            mm.register_component("vae", vae)
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the VAE: {e}", self.name) from e

    def _set_output_handles(self, mm):
        self.values["tokenizer"] = ModelHandle("tokenizer")
        self.values["text_encoder"] = ModelHandle("text_encoder")
        self.values["transformer"] = ModelHandle("transformer")
        self.values["vae"] = ModelHandle("vae")

        transformer = mm.get_raw("transformer")
        vae = mm.get_raw("vae")
        self.values["num_channels_latents"] = transformer.in_channels
        self.values["vae_scale_factor"] = 2 ** (len(vae.config.block_out_channels) - 1)

    def delete(self):
        self.clear_models()
        super().delete()

    def clear_models(self):
        self.values["transformer"] = None
        self.values["tokenizer"] = None
        self.values["text_encoder"] = None
        self.values["vae"] = None

        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
