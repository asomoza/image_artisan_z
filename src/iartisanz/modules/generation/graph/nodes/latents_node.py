import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node
from iartisanz.utils.image_converters import normalize, numpy_to_pt


class LatentsNode(Node):
    REQUIRED_INPUTS = ["vae_scale_factor", "width", "height", "num_channels_latents", "seed"]
    OPTIONAL_INPUTS = ["image", "vae"]
    OUTPUTS = ["latents", "noise"]
    SERIALIZE_EXCLUDE = {"latents"}

    def __init__(self, latents=None):
        super().__init__()
        self.latents = latents

    def __call__(self):
        mm = get_model_manager()
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        height = 2 * (int(self.height) // (self.vae_scale_factor * 2))
        width = 2 * (int(self.width) // (self.vae_scale_factor * 2))
        shape = (1, self.num_channels_latents, height, width)

        image = getattr(self, "image", None)
        noise = None

        if image is not None:
            try:
                # Optional input "vae" may be wired (or omitted); either way resolve from ModelManager.
                vae_input = getattr(self, "vae", None)
                if vae_input is None:
                    vae_input = ModelHandle("vae")
                vae = mm.resolve(vae_input, device=self.device)

                if isinstance(image, np.ndarray):
                    np_image = image
                else:
                    np_image = np.array(image)

                if np_image.ndim == 2:
                    np_image = np.stack([np_image, np_image, np_image], axis=-1)

                if np_image.dtype == np.uint8:
                    np_image = np_image.astype(np.float32) / 255.0
                else:
                    np_image = np_image.astype(np.float32)

                image = numpy_to_pt(np_image[None, ...]).to(device=self.device, dtype=self.dtype)
                image = normalize(image)
                latents = vae.encode(image).latent_dist.sample(generator)
                latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
                noise = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)
            except Exception as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)
        else:
            latents = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

        self.values["latents"] = latents
        self.values["noise"] = noise

        return self.values
