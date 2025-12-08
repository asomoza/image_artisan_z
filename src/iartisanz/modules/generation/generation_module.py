import logging

import torch
from PIL import Image
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QProgressBar, QSizePolicy, QSpacerItem, QVBoxLayout

from iartisanz.modules.base_module import BaseModule
from iartisanz.modules.generation.constants import LATENT_RGB_FACTORS
from iartisanz.modules.generation.generation_thread import DiffusersThread
from iartisanz.modules.generation.image_viewer_simple import ImageViewerSimple
from iartisanz.modules.generation.prompts_widget import PromptsWidget
from iartisanz.utils.image_processor import ImageProcessor


class GenerationModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.logger = logging.getLogger(__name__)

        self.thread = DiffusersThread()
        self.thread.progress_update.connect(self.step_progress_update)
        self.thread.generation_finished.connect(self.generation_finished)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        self.image_viewer = ImageViewerSimple(self.directories.outputs_images)
        self.image_viewer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.image_viewer)

        spacer = QSpacerItem(5, 5, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        main_layout.addSpacerItem(spacer)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.prompts_widget = PromptsWidget()
        self.prompts_widget.generate_signal.connect(self.on_generate)
        main_layout.addWidget(self.prompts_widget)

        main_layout.setStretch(0, 16)
        main_layout.setStretch(1, 0)
        main_layout.setStretch(2, 0)
        main_layout.setStretch(3, 4)

        self.setLayout(main_layout)

    def on_generate(self, seed: int, positive_prompt: str, negative_prompt: str):
        self.thread.seed = seed
        self.thread.positive_prompt = positive_prompt
        self.thread.negative_prompt = negative_prompt
        self.thread.start()

    def step_progress_update(self, step: int, latents: torch.Tensor):
        latent_rgb_factors = torch.tensor(LATENT_RGB_FACTORS, dtype=latents.dtype).to(device=latents.device)

        latent_image = latents.squeeze(0).permute(1, 2, 0) @ latent_rgb_factors
        latents_ubyte = ((latent_image + 1.0) / 2.0).clamp(0, 1).mul(0xFF)
        image = Image.fromarray(latents_ubyte.byte().cpu().numpy())

        self.show_preview(image)

    def generation_finished(self, image: Image):
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(100)

        image_processor = ImageProcessor()
        image_processor.set_pillow_image(image)
        image = None

        self.image_viewer.set_pixmap(image_processor.get_qpixmap())
        self.image_viewer.reset_view()

        self.show_preview(image)

    def show_preview(self, image: Image):
        if image is None:
            return

        qimage = QImage(image.tobytes(), image.width, image.height, QImage.Format.Format_RGB888)
        qpixmap = QPixmap.fromImage(qimage)

        label_size = self.image_viewer.size()

        scaled_pixmap = qpixmap.scaled(
            label_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

        self.image_viewer.set_pixmap(scaled_pixmap)
