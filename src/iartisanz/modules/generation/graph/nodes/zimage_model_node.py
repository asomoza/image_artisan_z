import os

import torch
from diffusers import AutoencoderKL, ZImageTransformer2DModel
from transformers import Qwen2Tokenizer, Qwen3Model

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
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
        self.values["tokenizer"] = Qwen2Tokenizer.from_pretrained(os.path.join(self.path, "tokenizer"))
        if "tokenizer" not in self.values or self.values["tokenizer"] is None:
            raise IArtisanZNodeError(
                "Error trying to load the tokenizer, probably the file doesn't exists.", self.name
            )

        self.values["text_encoder"] = Qwen3Model.from_pretrained(
            os.path.join(self.path, "text_encoder"),
            use_safetensors=True,
            dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )
        if self.abort:
            return
        if "text_encoder" not in self.values or self.values["text_encoder"] is None:
            raise IArtisanZNodeError(
                "Error trying to load the text encoder, probably the file doesn't exists.", self.name
            )

        self.values["transformer"] = ZImageTransformer2DModel.from_pretrained(
            os.path.join(self.path, "transformer"),
            use_safetensors=True,
            torch_dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )
        if self.abort:
            return
        if "transformer" not in self.values or self.values["transformer"] is None:
            raise IArtisanZNodeError(
                "Error trying to load the transformer, probably the file doesn't exists.", self.name
            )

        self.values["vae"] = AutoencoderKL.from_pretrained(
            os.path.join(self.path, "vae"),
            use_safetensors=True,
            torch_dtype=self.dtype,
            local_files_only=True,
            low_cpu_mem_usage=True,
            device_map=self.device,
        )
        if self.abort:
            return
        if "vae" not in self.values or self.values["vae"] is None:
            raise IArtisanZNodeError("Error trying to load the VAE, probably the file doesn't exists.", self.name)

        self.values["num_channels_latents"] = self.values["transformer"].in_channels
        self.values["vae_scale_factor"] = 2 ** (len(self.values["vae"].config.block_out_channels) - 1)
        return self.values

    def delete(self):
        self.clear_models()
        super().delete()

    def clear_models(self):
        self.values["transformer"] = None
        self.values["text_encoder"] = None
        self.values["tokenizer"] = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
