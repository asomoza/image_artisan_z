import numpy as np
import torch
from PyQt6.QtGui import QImage, QPixmap


def convert_latents_to_rgb(latents, latent_rgb_factors):
    latent_rgb_factors = torch.tensor(latent_rgb_factors, dtype=latents.dtype).to(device=latents.device)
    latent_image = latents.squeeze(0).permute(1, 2, 0) @ latent_rgb_factors
    latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
    denoised_image = latents_ubyte.byte().cpu().numpy()

    return denoised_image


def convert_numpy_to_pixmap(numpy_image: np.array):
    qimage = QImage(numpy_image.tobytes(), numpy_image.shape[1], numpy_image.shape[0], QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)

    return pixmap
