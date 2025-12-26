import inspect
from typing import Optional, Union

import torch

from iartisanz.modules.generation.graph.nodes.node import Node


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
    OPTIONAL_INPUTS = ["cfg_truncation", "cfg_normalization", "sigmas"]
    OUTPUTS = ["latents"]

    def __init__(self, callback: callable = None):
        super().__init__()

        self.callback = callback

    @torch.inference_mode()
    def __call__(self):
        do_classifier_free_guidance = True if self.guidance_scale > 1 else False
        cfg_truncation = self.cfg_truncation if self.cfg_truncation is not None else 1.0
        cfg_normalization = self.cfg_normalization if self.cfg_normalization is not None else False

        self.prompt_embeds = self.prompt_embeds.to(self.device)
        self.negative_prompt_embeds = self.negative_prompt_embeds.to(self.device)
        latents = self.latents

        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler,
            self.num_inference_steps,
            self.device,
            sigmas=self.sigmas,
            # **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        for i, t in enumerate(timesteps):
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0])
            timestep = (1000 - timestep) / 1000

            # Normalized time for time-aware config (0 at start, 1 at end)
            t_norm = timestep[0].item()

            # Handle cfg truncation
            current_guidance_scale = self.guidance_scale
            if do_classifier_free_guidance and cfg_truncation is not None and float(cfg_truncation) <= 1:
                if t_norm > cfg_truncation:
                    current_guidance_scale = 0.0

            # Run CFG only if configured AND scale is non-zero
            apply_cfg = do_classifier_free_guidance and current_guidance_scale > 0

            if apply_cfg:
                latents_typed = latents.to(self.transformer.dtype)
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = self.prompt_embeds + self.negative_prompt_embeds
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(self.transformer.dtype)
                prompt_embeds_model_input = self.prompt_embeds
                timestep_model_input = timestep

            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = list(latent_model_input.unbind(dim=0))

            model_out_list = self.transformer(
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

            noise_pred = noise_pred.squeeze(2)
            noise_pred = -noise_pred

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]

            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if self.callback is not None:
                    step_idx = i // getattr(self.scheduler, "order", 1)
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
