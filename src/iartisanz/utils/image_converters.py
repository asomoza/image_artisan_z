from typing import Union

import numpy as np
import torch
from PIL import Image
from PyQt6.QtGui import QImage, QPixmap


def convert_latents_to_rgb(latents, latent_rgb_factors) -> np.ndarray:
    latent_rgb_factors = torch.tensor(latent_rgb_factors, dtype=latents.dtype).to(device=latents.device)
    num_factor_channels = latent_rgb_factors.shape[0]

    latent_image = latents.squeeze(0)

    if latent_image.ndim == 2:
        # Flux2 packed format: (seq_len, C) where C = num_channels * 4 (patchified)
        seq_len, channels = latent_image.shape

        # Infer spatial dims from sequence length
        h = int(seq_len**0.5)
        while h > 0 and seq_len % h != 0:
            h -= 1
        w = seq_len // h

        latent_image = latent_image.reshape(h, w, channels)

        # Reduce packed channels to match the factor matrix if needed
        # e.g. 64 packed channels → average pairs → 32 channels for Flux2 factors
        if channels != num_factor_channels and channels % num_factor_channels == 0:
            groups = channels // num_factor_channels
            latent_image = latent_image.reshape(h, w, num_factor_channels, groups).mean(dim=3)

        latent_image = latent_image @ latent_rgb_factors
    else:
        # Z-Image format: (C, H, W) → (H, W, C)
        latent_image = latent_image.permute(1, 2, 0) @ latent_rgb_factors

    latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
    denoised_image = latents_ubyte.byte().cpu().numpy()

    return denoised_image


def convert_numpy_to_pixmap(numpy_image: np.array) -> QPixmap:
    qimage = QImage(numpy_image.tobytes(), numpy_image.shape[1], numpy_image.shape[0], QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimage)

    return pixmap


def pixmap_to_numpy_rgb(pixmap: QPixmap) -> np.ndarray:
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 4)
    return arr[:, :, :3].copy()


def numpy_to_pixmap(numpy_image: np.ndarray) -> QPixmap:
    if numpy_image.ndim == 2:
        data = numpy_image.astype(np.uint8, copy=False)
        height, width = data.shape
        qimage = QImage(data.data, width, height, width, QImage.Format.Format_Grayscale8).copy()
        return QPixmap.fromImage(qimage)

    if numpy_image.shape[2] == 4:
        data = numpy_image.astype(np.uint8, copy=False)
        height, width, _ = data.shape
        qimage = QImage(
            data.data,
            width,
            height,
            data.strides[0],
            QImage.Format.Format_RGBA8888,
        ).copy()
        return QPixmap.fromImage(qimage)

    data = numpy_image.astype(np.uint8, copy=False)
    height, width, _ = data.shape
    qimage = QImage(data.data, width, height, data.strides[0], QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(qimage)


def pixmap_to_pil(pixmap: QPixmap) -> Image.Image:
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(image.sizeInBytes())
    arr = np.frombuffer(ptr, np.uint8).reshape(height, width, 4)
    return Image.fromarray(arr, mode="RGBA").convert("RGB")


def pil_to_pixmap(image: Image.Image) -> QPixmap:
    return numpy_to_pixmap(np.array(image))


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
