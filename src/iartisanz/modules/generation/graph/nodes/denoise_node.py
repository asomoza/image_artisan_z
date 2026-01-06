import inspect
import logging
from typing import Optional, Union

import numpy as np
import torch

from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.model_manager import get_model_manager
from iartisanz.modules.generation.graph.nodes.node import Node
from iartisanz.utils.image_converters import numpy_to_pt


logger = logging.getLogger(__name__)


class DenoiseNode(Node):
    REQUIRED_INPUTS = [
        "transformer",
        "num_inference_steps",
        "latents",
        "scheduler",
        "prompt_embeds",
        "negative_prompt_embeds",
        "guidance_scale",
    ]
    OPTIONAL_INPUTS = [
        "cfg_normalization",
        "sigmas",
        "lora",
        "guidance_start_end",
        "noise",
        "strength",
        "image_mask",
    ]
    OUTPUTS = ["latents"]
    SERIALIZE_EXCLUDE = {"callback"}

    def __init__(self, callback: callable = None):
        super().__init__()

        self.callback = callback

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        transformer = mm.resolve(self.transformer)

        do_classifier_free_guidance = True if self.guidance_scale > 1 else False

        guidance_start = float(self.guidance_start_end[0] if hasattr(self, "guidance_start_end") else 0.0)
        guidance_end = float(self.guidance_start_end[1] if hasattr(self, "guidance_start_end") else 1.0)

        if hasattr(self, "strength") and self.strength is not None:
            num_inference_steps = int(self.num_inference_steps / self.strength)
        else:
            num_inference_steps = self.num_inference_steps

        if guidance_start > guidance_end:
            logger.warning(
                "guidance_start (%s) is higher than guidance_end (%s); ignoring guidance window and using full range [0.0, 1.0].",
                guidance_start,
                guidance_end,
            )
            guidance_start, guidance_end = 0.0, 1.0

        cfg_normalization = self.cfg_normalization if self.cfg_normalization is not None else False

        if self.lora:
            try:
                if isinstance(self.lora, list):
                    keys = [item[0] for item in self.lora]
                    transformer_values = [item[1]["transformer"] for item in self.lora]
                    transformer.set_adapters(keys, transformer_values)
                else:
                    transformer.set_adapters([self.lora[0]], self.lora[1]["transformer"])
            except RuntimeError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)

        prompt_embeds = self.prompt_embeds.to(self.device)
        negative_prompt_embeds = self.negative_prompt_embeds.to(self.device)

        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            sigmas=self.sigmas,
        )
        total_time_steps = num_inference_steps

        if hasattr(self, "strength") and self.strength is not None:
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, self.strength)

        original_image_latents = self.latents
        latents = self.latents
        masks = None

        if hasattr(self, "noise") and self.noise is not None:
            latents = self.scheduler.scale_noise(latents, timesteps[:1], self.noise.to(self.device))

            # begin differential diffusion
            if hasattr(self, "image_mask") and self.image_mask is not None:
                original_mask = np.expand_dims(self.image_mask, axis=0)
                original_mask = np.concatenate(original_mask, axis=0)
                original_mask = numpy_to_pt(original_mask[None, ...]).to(device=self.device, dtype=self.dtype)
                original_mask = torch.nn.functional.interpolate(
                    original_mask, size=(latents.shape[2], latents.shape[3])
                )

                mask_thresholds = torch.arange(total_time_steps, dtype=original_mask.dtype) / total_time_steps
                mask_thresholds = mask_thresholds.reshape(-1, 1, 1, 1).to(self.device)
                masks = original_mask > mask_thresholds
            # end differential diffusion

        order = getattr(self.scheduler, "order", 1)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * order, 0)
        step_norm_den = float(max(num_inference_steps - 1, 1))

        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            # Per-inference-step index/progress (0 at start, 1 at end)
            step_idx = i // order
            step_norm = float(step_idx) / step_norm_den

            current_guidance_scale = self.guidance_scale
            if not (guidance_start <= step_norm <= guidance_end):
                current_guidance_scale = 0.0

            apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

            if apply_cfg:
                latents_typed = latents.to(transformer.dtype)
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = [
                    prompt_embeds.to(dtype=transformer.dtype),
                    negative_prompt_embeds.to(dtype=transformer.dtype),
                ]
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(transformer.dtype)
                prompt_embeds_model_input = [prompt_embeds.to(dtype=transformer.dtype)]
                timestep_model_input = timestep

            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            model_out_list = transformer(
                latent_model_input_list, timestep_model_input, prompt_embeds_model_input, return_dict=False
            )[0]

            if apply_cfg:
                # Perform CFG
                pos_out = model_out_list[:1]
                neg_out = model_out_list[1:]

                noise_pred = []
                for j in range(1):
                    pos = pos_out[j].float()
                    neg = neg_out[j].float()

                    pred = pos + current_guidance_scale * (pos - neg)

                    # Renormalization
                    if cfg_normalization and float(cfg_normalization) > 0.0:
                        ori_pos_norm = torch.linalg.vector_norm(pos)
                        new_pos_norm = torch.linalg.vector_norm(pred)
                        max_new_norm = ori_pos_norm * float(cfg_normalization)
                        if new_pos_norm > max_new_norm:
                            pred = pred * (max_new_norm / new_pos_norm)

                    noise_pred.append(pred)

                noise_pred = torch.stack(noise_pred, dim=0)
            else:
                noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

            if self.abort:
                return

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

            # begin differential diffusion
            if masks is not None:
                image_latent = original_image_latents

                if i < len(timesteps) - 1:
                    noise_timestep = timesteps[i + 1]
                    image_latent = self.scheduler.scale_noise(self.latents, torch.tensor([noise_timestep]), self.noise)

                    mask = masks[i].to(latents.dtype)
                    latents = image_latent * mask + latents * (1 - mask)
            # end differential diffusion

            is_last = i == len(timesteps) - 1
            is_step_boundary = (i + 1) > num_warmup_steps and (i + 1) % order == 0
            if (is_last or is_step_boundary) and self.callback is not None:
                self.callback(step_idx, t, latents)

        self.values["latents"] = latents.to("cpu").detach().clone()

        return self.values

    def retrieve_timesteps(
        self,
        scheduler,
        num_inference_steps: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        timesteps: Optional[list[int]] = None,
        sigmas: Optional[list[float]] = None,
        **kwargs,
    ):
        if timesteps is not None and sigmas is not None:
            raise ValueError(
                "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
            )
        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" timestep schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    f" sigmas schedules. Please check whether you are using the correct scheduler."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps

    def get_timesteps(self, num_inference_steps, strength):
        # get the original timestep using init_timestep
        init_timestep = min(num_inference_steps * strength, num_inference_steps)

        t_start = int(max(num_inference_steps - init_timestep, 0))
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps, num_inference_steps - t_start
