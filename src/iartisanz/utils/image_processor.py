import io
import json
import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from PyQt6.QtCore import QBuffer, QByteArray, QIODevice
from PyQt6.QtGui import QImage, QPixmap


class ImageProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.image_data = None
        self.serialized_data = None

    def open_image(self, path: str):
        pil_image = Image.open(path)

        metadata = pil_image.info
        data = metadata.get("iartisanz_data")

        if data is not None:
            self.serialized_data = data

        self.set_pillow_image(pil_image)

    def set_pillow_image(self, pillow_image: Image):
        byte_arr = io.BytesIO()
        pillow_image.save(byte_arr, format="PNG")
        self.image_data = byte_arr.getvalue()

    def set_pixmap(self, pixmap: QPixmap):
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        self.image_data = byte_array.data()

    def set_qimage(self, qimage: QImage):
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        qimage.save(buffer, "PNG")
        self.image_data = byte_array.data()

    def set_numpy_array(self, np_array: np.ndarray):
        self.image_data = np_array.tobytes()

    def set_serialized_data(self, serialized_data: json):
        self.serialized_data = serialized_data

    def get_qimage(self) -> QImage:
        byte_array = QByteArray(self.image_data)
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)

        qimage = QImage()
        qimage.loadFromData(buffer.data(), "PNG")
        return qimage

    def get_qpixmap(self) -> QPixmap:
        qimage = self.get_qimage()
        qpixmap = QPixmap.fromImage(qimage)
        return qpixmap

    def get_pillow_image(self) -> Image:
        return Image.open(io.BytesIO(self.image_data))

    def save_to_png(self, output_path: str):
        if os.path.isdir(output_path):
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}.png"
            output_filepath = os.path.join(output_path, filename)
        else:
            output_filepath = output_path

        image = Image.open(io.BytesIO(self.image_data))

        if output_filepath is not None and len(output_filepath) > 0:
            try:
                if self.serialized_data is None:
                    image.save(output_filepath)
                    return

                metadata = PngInfo()
                metadata.add_text("iartisanz_data", self.serialized_data)

                image.save(output_filepath, pnginfo=metadata)
            except ValueError:
                pass

    def get_pillow_thumbnail(self, target_height: Optional[int] = None, target_width: Optional[int] = None) -> Image:
        image = self.get_pillow_image()

        if target_height is not None and target_width is not None:
            image.thumbnail((target_width, target_height))
        elif target_height is not None:
            ratio = target_height / image.height
            target_width = int(image.width * ratio)
            image.thumbnail((target_width, target_height))
        elif target_width is not None:
            ratio = target_width / image.width
            target_height = int(image.height * ratio)
            image.thumbnail((target_width, target_height))

        return image
