from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
from image_gen_aux import DepthPreprocessor, LineArtPreprocessor, LineArtStandardPreprocessor, TeedPreprocessor
from PIL.Image import Image as PILImage
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from iartisanz.app.model_manager import get_model_manager
from iartisanz.utils.image_converters import numpy_to_pixmap, pixmap_to_pil


class ControlnetPreprocessThread(QThread):
    error = pyqtSignal(str)
    preprocessor_finished = pyqtSignal(QPixmap)

    def __init__(
        self,
        pixmap: QPixmap,
        preprocessor_type: str,
        preprocessor_name: str = "",
        preprocessor_model: str = "",
        resolution_scale: float = 1.0,
    ):
        super().__init__()

        self.pixmap = pixmap
        self.preprocessor_type = preprocessor_type
        self.preprocessor_name = preprocessor_name
        self.preprocessor_model = preprocessor_model
        self.resolution_scale = resolution_scale

        self.logger = logging.getLogger(__name__)

    def _normalize_output(self, output: Any) -> np.ndarray:
        if hasattr(output, "cpu"):
            output = output.detach().cpu().numpy()

        if isinstance(output, np.ndarray):
            if output.dtype != np.uint8:
                output = np.clip(output, 0, 255).astype(np.uint8)
            return output

        if isinstance(output, PILImage):
            return np.array(output)

        if isinstance(output, (list, tuple)) and output:
            first = output[0]
            if isinstance(first, PILImage):
                return np.array(first)

        if hasattr(output, "numpy"):
            output = output.numpy()
            if output.dtype != np.uint8:
                output = np.clip(output, 0, 255).astype(np.uint8)
            return output

        raise ValueError("Unsupported preprocessor output type")

    def _run_preprocessor(self, preprocessor: Any, image: Any, resolution_scale: float | None):
        try:
            if resolution_scale is not None:
                return preprocessor(image, resolution_scale)
            return preprocessor(image)
        except TypeError:
            return preprocessor(image)

    def run(self):
        if self.pixmap is None or self.pixmap.isNull():
            self.error.emit("No source image to preprocess.")
            return

        try:
            mm = get_model_manager()
            preprocessor_id = f"{self.preprocessor_type}:{self.preprocessor_name}:{self.preprocessor_model}"

            if mm.has("preprocessor"):
                try:
                    existing = mm.get_raw("preprocessor")
                    if getattr(existing, "_iartisanz_preprocessor_id", None) != preprocessor_id:
                        mm.clear_component("preprocessor")
                except Exception:
                    mm.clear_component("preprocessor")

            if not mm.has("preprocessor"):
                if self.preprocessor_type == "depth":
                    if not self.preprocessor_model:
                        raise ValueError("Depth preprocessor requires a model repo id.")
                    preprocessor = DepthPreprocessor.from_pretrained(self.preprocessor_model)
                elif self.preprocessor_type == "lines":
                    if self.preprocessor_model:
                        try:
                            preprocessor = LineArtPreprocessor.from_pretrained(self.preprocessor_model)
                        except Exception:
                            preprocessor = LineArtPreprocessor()
                    else:
                        preprocessor = LineArtPreprocessor()
                elif self.preprocessor_type == "lineart_standard":
                    preprocessor = LineArtStandardPreprocessor()
                elif self.preprocessor_type == "teed":
                    if not self.preprocessor_model:
                        raise ValueError("Edge preprocessor requires a model repo id.")
                    preprocessor = TeedPreprocessor.from_pretrained(self.preprocessor_model)
                else:
                    self.error.emit(f"Unsupported preprocessor: {self.preprocessor_type}")
                    return

                setattr(preprocessor, "_iartisanz_preprocessor_id", preprocessor_id)
                mm.register_component("preprocessor", preprocessor)

            preprocessor = mm.get("preprocessor")
            if torch.cuda.is_available() and hasattr(preprocessor, "to"):
                try:
                    preprocessor = preprocessor.to("cuda")
                    mm.register_component("preprocessor", preprocessor)
                except Exception:
                    pass

            image_input = pixmap_to_pil(self.pixmap)

            resolution_scale = float(self.resolution_scale) if self.resolution_scale is not None else None

            output = self._run_preprocessor(preprocessor, image_input, resolution_scale)
            output_array = self._normalize_output(output)

            processed_pixmap = numpy_to_pixmap(output_array)
            self.preprocessor_finished.emit(processed_pixmap)
        except Exception as e:
            self.logger.exception("Preprocessor failed")
            self.error.emit(str(e))
