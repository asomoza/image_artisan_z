from typing import Union

import numpy as np
import torch
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap


def convert_latents_to_rgb(latents, latent_rgb_factors) -> np.ndarray:
    latent_rgb_factors = torch.tensor(latent_rgb_factors, dtype=latents.dtype).to(device=latents.device)
    latent_image = latents.squeeze(0).permute(1, 2, 0) @ latent_rgb_factors
    latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
    denoised_image = latents_ubyte.byte().cpu().numpy()

    return denoised_image


def convert_numpy_to_pixmap(numpy_image: np.array) -> QPixmap:
    qimage = QImage(numpy_image.tobytes(), numpy_image.shape[1], numpy_image.shape[0], QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)

    return pixmap


def pil_to_numpy(images: Union[list[Image.Image], Image.Image]) -> np.ndarray:
    if not isinstance(images, list):
        images = [images]
    images = [np.array(image).astype(np.float32) / 255.0 for image in images]
    images = np.stack(images, axis=0)

    return images


def numpy_to_pt(images: np.ndarray) -> torch.Tensor:
    if images.ndim == 3:
        images = images[..., None]

    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images


def normalize(images: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    return 2.0 * images - 1.0
