from typing import Union

import numpy as np
import PIL
import torch
from diffusers.utils.torch_utils import randn_tensor

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


class LatentsNode(Node):
    REQUIRED_INPUTS = ["vae_scale_factor", "width", "height", "num_channels_latents", "seed"]
    OPTIONAL_INPUTS = ["image", "vae"]
    OUTPUTS = ["latents", "noise"]
    SERIALIZE_EXCLUDE = {"latents"}

    def __init__(self, latents=None):
        super().__init__()
        self.latents = latents

    def __call__(self):
        generator = torch.Generator(device=self.device).manual_seed(self.seed)

        height = 2 * (int(self.height) // (self.vae_scale_factor * 2))
        width = 2 * (int(self.width) // (self.vae_scale_factor * 2))
        shape = (1, self.num_channels_latents, height, width)

        image = getattr(self, "image", None)
        noise = None

        if image is not None:
            try:
                image = self.pil_to_numpy(image)
                image = self.numpy_to_pt(image).to(device=self.device, dtype=self.dtype)
                image = self.normalize(image)
                latents = self.vae.encode(image).latent_dist.sample(generator)
                latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
                noise = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)
            except Exception as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)
        else:
            latents = randn_tensor(shape, generator=generator, device=self.device, dtype=self.dtype)

        self.values["latents"] = latents
        self.values["noise"] = noise

        return self.values

    @staticmethod
    def pil_to_numpy(images: Union[list[PIL.Image.Image], PIL.Image.Image]) -> np.ndarray:
        r"""
        Convert a PIL image or a list of PIL images to NumPy arrays.

        Args:
            images (`PIL.Image.Image` or `List[PIL.Image.Image]`):
                The PIL image or list of images to convert to NumPy format.

        Returns:
            `np.ndarray`:
                A NumPy array representation of the images.
        """
        if not isinstance(images, list):
            images = [images]
        images = [np.array(image).astype(np.float32) / 255.0 for image in images]
        images = np.stack(images, axis=0)

        return images

    @staticmethod
    def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
        r"""
        Convert a NumPy image to a PyTorch tensor.

        Args:
            images (`np.ndarray`):
                The NumPy image array to convert to PyTorch format.

        Returns:
            `torch.Tensor`:
                A PyTorch tensor representation of the images.
        """
        if images.ndim == 3:
            images = images[..., None]

        images = torch.from_numpy(images.transpose(0, 3, 1, 2))
        return images

    @staticmethod
    def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        r"""
        Normalize an image array to [-1,1].

        Args:
            images (`np.ndarray` or `torch.Tensor`):
                The image array to normalize.

        Returns:
            `np.ndarray` or `torch.Tensor`:
                The normalized image array.
        """
        return 2.0 * images - 1.0
