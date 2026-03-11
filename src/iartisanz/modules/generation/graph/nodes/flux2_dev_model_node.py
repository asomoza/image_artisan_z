import logging

from diffusers import AutoencoderKLFlux2, Flux2Transformer2DModel
from transformers import AutoProcessor, Mistral3ForConditionalGeneration

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.zimage_model_node import ZImageModelNode


logger = logging.getLogger(__name__)


class Flux2DevModelNode(ZImageModelNode):
    """Model node for loading Flux.2 Dev components.

    Inherits smart-load / registry-paths logic from ZImageModelNode.
    Uses Mistral3ForConditionalGeneration + AutoProcessor (PixtralProcessor)
    instead of the Qwen3 text encoder used by Flux2 Klein.
    """

    def _load_tokenizer(self, tokenizer_path: str):
        mm = get_model_manager()
        try:
            tokenizer = AutoProcessor.from_pretrained(
                tokenizer_path,
                local_files_only=True,
            )
            if tokenizer is None:
                raise IArtisanZNodeError(
                    "Error trying to load the tokenizer.", self.name
                )
            mm.register_component("tokenizer", tokenizer)
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the tokenizer: {e}", self.name) from e

    def _load_text_encoder(self, text_encoder_path: str):
        mm = get_model_manager()
        try:
            self._prepare_quantization(text_encoder_path)

            text_encoder = Mistral3ForConditionalGeneration.from_pretrained(
                text_encoder_path,
                use_safetensors=True,
                torch_dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )

            self._apply_quantization_options(text_encoder, text_encoder_path)
            mm.register_component("text_encoder", text_encoder)
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the text encoder: {e}", self.name) from e

    def _load_transformer(self, transformer_path: str):
        mm = get_model_manager()
        try:
            self._prepare_quantization(transformer_path)

            transformer = Flux2Transformer2DModel.from_pretrained(
                transformer_path,
                use_safetensors=True,
                torch_dtype=self.dtype,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )

            self._apply_quantization_options(transformer, transformer_path)
            mm.register_component("transformer", transformer)
        except OSError as e:
            raise IArtisanZNodeError(f"Error trying to load the transformer: {e}", self.name) from e

    def _load_vae(self, vae_path: str):
        mm = get_model_manager()
        try:
            vae = AutoencoderKLFlux2.from_pretrained(
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
        # Flux2 patchifies 2x2, so real latent channels = in_channels // 4
        self.values["num_channels_latents"] = transformer.config.in_channels // 4
        self.values["vae_scale_factor"] = 2 ** (len(vae.config.block_out_channels) - 1)

        # Register nn.Module components for offload lifecycle management
        text_encoder = mm.get_raw("text_encoder")
        mm.register_managed_component("text_encoder", text_encoder)
        mm.register_managed_component("transformer", transformer)
        mm.register_managed_component("vae", vae)

        # Apply offload strategy (no-op for "auto")
        mm.apply_offload_strategy(self.device)
