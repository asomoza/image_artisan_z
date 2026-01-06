import os

import torch
from diffusers import AutoencoderKL, ZImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen3Model

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node


class ZImageModelNode(Node):
    OUTPUTS = ["text_encoder", "tokenizer", "transformer", "vae", "num_channels_latents", "vae_scale_factor"]

    def __init__(
        self,
        path: str = None,
        model_name: str = None,
        version: str = None,
        model_type: str = None,
    ):
        super().__init__()
        self.path = path
        self.model_name = model_name
        self.version = version
        self.model_type = model_type

    def update_model(self, path: str, model_name: str, version: str, model_type: str):
        self.clear_models()
        self.path = path
        self.model_name = model_name
        self.version = version
        self.model_type = model_type
        self.set_updated()

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict["path"] = self.path
        node_dict["model_name"] = self.model_name
        node_dict["version"] = self.version
        node_dict["model_type"] = self.model_type
        return node_dict

    @classmethod
    def from_dict(cls, node_dict, _callbacks=None):
        node = super(ZImageModelNode, cls).from_dict(node_dict)
        node.path = node_dict["path"]
        node.model_name = node_dict["model_name"]
        node.version = node_dict["version"]
        node.model_type = node_dict["model_type"]
        return node

    def update_inputs(self, node_dict):
        self.clear_models()
        self.path = node_dict["path"]
        self.model_name = node_dict["model_name"]
        self.version = node_dict["version"]
        self.model_type = node_dict["model_type"]

    def __call__(self):
        mm = get_model_manager()

        try:
            tokenizer = Qwen2Tokenizer.from_pretrained(
                os.path.join(self.path, "tokenizer"),
                local_files_only=True,
            )
            if tokenizer is None:
                raise IArtisanZNodeError(
                    "Error trying to load the tokenizer, probably the file doesn't exists.", self.name
                )
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the tokenizer: {e}", self.name) from e

        try:
            text_encoder = Qwen3Model.from_pretrained(
                os.path.join(self.path, "text_encoder"),
                use_safetensors=True,
                dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map=self.device,
            )
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the text encoder: {e}", self.name) from e

        if self.abort:
            return

        try:
            transformer = ZImageTransformer2DModel.from_pretrained(
                os.path.join(self.path, "transformer"),
                use_safetensors=True,
                torch_dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map=self.device,
            )
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the transformer: {e}", self.name) from e

        if self.abort:
            return

        try:
            vae = AutoencoderKL.from_pretrained(
                os.path.join(self.path, "vae"),
                use_safetensors=True,
                torch_dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
                device_map=self.device,
            )
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the VAE: {e}", self.name) from e

        if self.abort:
            return

        mm.register_active_model(
            model_id=self.model_name or self.path,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            transformer=transformer,
            vae=vae,
        )

        # Pass lightweight handles through the graph; heavy objects live in ModelManager.
        self.values["tokenizer"] = ModelHandle("tokenizer")
        self.values["text_encoder"] = ModelHandle("text_encoder")
        self.values["transformer"] = ModelHandle("transformer")
        self.values["vae"] = ModelHandle("vae")

        self.values["num_channels_latents"] = transformer.in_channels
        self.values["vae_scale_factor"] = 2 ** (len(vae.config.block_out_channels) - 1)

        return self.values

    def delete(self):
        self.clear_models()
        super().delete()

    def clear_models(self):
        get_model_manager().clear()
        self.values["transformer"] = None
        self.values["tokenizer"] = None
        self.values["text_encoder"] = None
        self.values["vae"] = None

        if torch.cuda.is_available():
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
