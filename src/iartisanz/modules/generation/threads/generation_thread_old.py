import random

import torch
from diffusers import ZImagePipeline
from PIL import Image
from PyQt6.QtCore import QThread, pyqtSignal


class DiffusersThread(QThread):
    generation_finished = pyqtSignal(Image.Image)
    progress_update = pyqtSignal(int, torch.Tensor)

    def __init__(self):
        super().__init__()

        self.pipe = None
        self.positive_prompt = None
        self.negative_prompt = None
        self.seed = None

    def run(self):
        if self.pipe is None:
            self.pipe = ZImagePipeline.from_pretrained(
                "Tongyi-MAI/Z-Image-Turbo",
                torch_dtype=torch.bfloat16,
                device_map="cuda",
            )

        seed = self.seed if self.seed is not None and self.seed >= 0 else random.randint(0, 2**32 - 1)
        generator = torch.Generator(device="cuda").manual_seed(seed)

        image = self.pipe(
            self.positive_prompt,
            self.negative_prompt,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=generator,
            callback_on_step_end=self.step_progress_update,
            callback_on_step_end_tensor_inputs=["latents"],
        ).images[0]

        self.preview_image(image)

    def step_progress_update(self, _pipe, step, _timestep, callback_kwargs):
        latents = callback_kwargs["latents"]
        self.progress_update.emit(step, latents)
        return callback_kwargs

    def preview_image(self, image):
        self.generation_finished.emit(image)
