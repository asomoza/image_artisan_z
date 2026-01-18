import inspect
import logging
from typing import Optional, Union

import numpy as np
import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
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
        "use_torch_compile",
        "noise",
        "strength",
        "image_mask",
        "controlnet",
        "control_image_latents",
        "controlnet_conditioning_scale",
        "control_guidance_start_end",
        "control_mode",
        "prompt_mode_decay",
    ]
    OUTPUTS = ["latents"]
    SERIALIZE_EXCLUDE = {"callback"}

    def __init__(self, callback: callable = None):
        super().__init__()

        self.callback = callback

    @staticmethod
    def _apply_prompt_mode_decay(block_samples: dict, decay: float) -> dict:
        # Apply in deterministic order (matches "injection count" concept).
        try:
            keys = sorted(block_samples.keys())
        except Exception:
            return block_samples

        for injection_index, layer_idx in enumerate(keys):
            try:
                block_samples[layer_idx] = block_samples[layer_idx] * (float(decay) ** injection_index)
            except Exception:
                # If a value is not directly scalable, keep it unchanged.
                pass
        return block_samples

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        transformer_input = self.transformer
        transformer_raw = mm.resolve(transformer_input)

        use_torch_compile = bool(getattr(self, "use_torch_compile", False) or False)
        transformer = transformer_raw

        # If the transformer was previously region-compiled in-place (Diffusers) and the
        # user toggled compilation off, restore eager behavior.
        if (
            (not use_torch_compile)
            and isinstance(transformer_input, ModelHandle)
            and transformer_input.component == "transformer"
        ):
            try:
                mm.disable_compiled("transformer")
            except Exception:
                pass

        # Avoid repeated dtype casts and small allocations inside the denoise loop.
        transformer_dtype = getattr(transformer_raw, "dtype", None) or getattr(transformer, "dtype", None)
        if transformer_dtype is None:
            try:
                for p in transformer_raw.parameters(recurse=True):
                    transformer_dtype = p.dtype
                    break
                if transformer_dtype is None:
                    for b in transformer_raw.buffers(recurse=True):
                        transformer_dtype = b.dtype
                        break
            except Exception:
                transformer_dtype = None
        if transformer_dtype is None:
            transformer_dtype = self.dtype or torch.float32

        guidance_scale = float(self.guidance_scale)
        do_classifier_free_guidance = abs(guidance_scale - 1.0) > 1e-6

        guidance_start_end = getattr(self, "guidance_start_end", None)
        if guidance_start_end is None:
            guidance_start, guidance_end = 0.0, 1.0
        else:
            guidance_start = float(guidance_start_end[0])
            guidance_end = float(guidance_start_end[1])

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
                    transformer_raw.set_adapters(keys, transformer_values)
                else:
                    transformer_raw.set_adapters([self.lora[0]], self.lora[1]["transformer"])
            except RuntimeError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)

        # Compile after adapters are set so the compiled graph matches runtime behavior.
        if (
            use_torch_compile
            and isinstance(transformer_input, ModelHandle)
            and transformer_input.component == "transformer"
        ):
            # If LoRAs are involved, be conservative: clear any cached compiled wrapper
            # to avoid reusing a graph compiled under a different adapter configuration.
            if self.lora:
                try:
                    mm.clear_compiled("transformer")
                except Exception:
                    pass

            try:
                transformer = mm.get_compiled(
                    "transformer",
                    device=self.device,
                    dtype=self.dtype,
                    compile_kwargs={"fullgraph": True},
                )
            except Exception:
                transformer = transformer_raw

        prompt_embeds = self.prompt_embeds.to(self.device)
        negative_prompt_embeds = self.negative_prompt_embeds.to(self.device)

        # Pre-cast conditioning once per invocation.
        prompt_embeds_typed = prompt_embeds.to(dtype=transformer_dtype)
        negative_prompt_embeds_typed = negative_prompt_embeds.to(dtype=transformer_dtype)

        controlnet_input = getattr(self, "controlnet", None)
        control_image_latents = getattr(self, "control_image_latents", None)
        has_controlnet = controlnet_input is not None and control_image_latents is not None

        control_mode = getattr(self, "control_mode", None)
        if control_mode is None:
            control_mode = "balanced"
        control_mode = str(control_mode).strip().lower()
        if control_mode not in {"balanced", "prompt", "controlnet"}:
            logger.warning(
                "Unknown control_mode '%s'; falling back to 'balanced'.",
                control_mode,
            )
            control_mode = "balanced"

        use_prompt_mode = control_mode == "prompt"
        use_guess_mode = control_mode == "controlnet"
        prompt_mode_decay = getattr(self, "prompt_mode_decay", None)
        prompt_mode_decay = float(prompt_mode_decay) if prompt_mode_decay is not None else 0.825

        control_guidance_start_end = getattr(self, "control_guidance_start_end", None)
        if control_guidance_start_end is None:
            control_guidance_start, control_guidance_end = 0.0, 1.0
        else:
            control_guidance_start = float(control_guidance_start_end[0])
            control_guidance_end = float(control_guidance_start_end[1])

        if control_guidance_start > control_guidance_end:
            logger.warning(
                "control_guidance_start (%s) is higher than control_guidance_end (%s); ignoring control guidance window and using full range [0.0, 1.0].",
                control_guidance_start,
                control_guidance_end,
            )
            control_guidance_start, control_guidance_end = 0.0, 1.0

        controlnet = None
        control_image_latents_typed = None
        controlnet_conditioning_scale = getattr(self, "controlnet_conditioning_scale", None)
        controlnet_conditioning_scale = (
            float(controlnet_conditioning_scale) if controlnet_conditioning_scale is not None else 1.0
        )

        control_context_latents_cfg = None

        if has_controlnet:
            controlnet = mm.resolve(controlnet_input)
            control_image_latents_typed = control_image_latents.to(self.device, dtype=transformer_dtype)

            # For some ControlNet checkpoints (e.g. 2.0), control_in_dim can differ from transformer in_channels.
            try:
                control_in_dim = int(getattr(controlnet, "config", {}).get("control_in_dim"))
            except Exception:
                control_in_dim = int(getattr(getattr(controlnet, "config", None), "control_in_dim", 0) or 0)

            if control_in_dim and control_image_latents_typed.shape[1] != control_in_dim:
                if control_image_latents_typed.shape[1] > control_in_dim:
                    raise IArtisanZNodeError(
                        f"Control image latents have {control_image_latents_typed.shape[1]} channels but controlnet expects {control_in_dim}.",
                        self.__class__.__name__,
                    )
                pad = torch.zeros(
                    control_image_latents_typed.shape[0],
                    control_in_dim - control_image_latents_typed.shape[1],
                    *control_image_latents_typed.shape[2:],
                    device=control_image_latents_typed.device,
                    dtype=control_image_latents_typed.dtype,
                )
                control_image_latents_typed = torch.cat([control_image_latents_typed, pad], dim=1)

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

        noise = getattr(self, "noise", None)

        if noise is not None:
            noise = noise.to(self.device)
            latents = self.scheduler.scale_noise(latents, timesteps[:1], noise)

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

            current_guidance_scale = guidance_scale
            if not (guidance_start <= step_norm <= guidance_end):
                # Outside the guidance window, behave like "no guidance".
                current_guidance_scale = 1.0

            apply_cfg = do_classifier_free_guidance and (abs(float(current_guidance_scale) - 1.0) > 1e-6)

            if apply_cfg:
                latents_typed = latents.to(transformer_dtype)
                latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                prompt_embeds_model_input = (prompt_embeds_typed, negative_prompt_embeds_typed)
                timestep_model_input = timestep.repeat(2)
            else:
                latent_model_input = latents.to(transformer_dtype)
                prompt_embeds_model_input = (prompt_embeds_typed,)
                timestep_model_input = timestep

            latent_model_input = latent_model_input.unsqueeze(2)
            latent_model_input_list = latent_model_input.unbind(dim=0)

            # If compilation uses CUDA graphs, PyTorch requires marking step boundaries to
            # avoid reading outputs that get overwritten by subsequent runs.
            if use_torch_compile and self.device.type == "cuda" and hasattr(torch, "compiler"):
                mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
                if callable(mark):
                    mark()

            controlnet_block_samples = None
            if has_controlnet:
                # Per-step keep schedule (1 -> apply, 0 -> skip).
                step_progress_start = float(step_idx) / float(max(num_inference_steps, 1))
                step_progress_end = float(step_idx + 1) / float(max(num_inference_steps, 1))
                keep = 1.0 - float(
                    step_progress_start < float(control_guidance_start)
                    or step_progress_end > float(control_guidance_end)
                )
                cond_scale = float(keep) * float(controlnet_conditioning_scale)

                if cond_scale <= 0.0:
                    controlnet_block_samples = None
                elif apply_cfg and use_guess_mode:
                    # In "ControlNet more important" mode, only condition the positive branch.
                    base_batch = int(latents.shape[0])

                    pos_latent_model_input_list = latent_model_input_list[:base_batch]
                    pos_timestep_model_input = timestep_model_input[:base_batch]

                    # Match the non-CFG prompt input shape. Prompt tensor is already batched.
                    pos_prompt_embeds_model_input = (prompt_embeds_typed,)

                    pos_controlnet_block_samples = controlnet(
                        pos_latent_model_input_list,
                        pos_timestep_model_input,
                        pos_prompt_embeds_model_input,
                        control_image_latents_typed,
                        conditioning_scale=cond_scale,
                    )

                    # Expand to (pos, neg) batch with zeros for the negative branch.
                    expanded: dict = {}
                    for layer_idx, hint in (pos_controlnet_block_samples or {}).items():
                        try:
                            expanded[layer_idx] = torch.cat([hint, torch.zeros_like(hint)], dim=0)
                        except Exception:
                            # If concatenation fails, fall back to passing pos-only (best-effort).
                            expanded[layer_idx] = hint
                    controlnet_block_samples = expanded
                else:
                    # Balanced/prompt modes: apply ControlNet to all branches.
                    control_context = control_image_latents_typed
                    if apply_cfg:
                        if control_context_latents_cfg is None:
                            control_context_latents_cfg = torch.cat([control_context] * 2, dim=0)
                        control_context = control_context_latents_cfg

                    controlnet_block_samples = controlnet(
                        latent_model_input_list,
                        timestep_model_input,
                        prompt_embeds_model_input,
                        control_context,
                        conditioning_scale=cond_scale,
                    )

                    if use_prompt_mode and controlnet_block_samples is not None:
                        controlnet_block_samples = self._apply_prompt_mode_decay(
                            controlnet_block_samples, prompt_mode_decay
                        )

            model_out_list = transformer(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
                controlnet_block_samples=controlnet_block_samples,
                return_dict=False,
            )[0]

            if apply_cfg:
                # Perform CFG. Graph currently uses batch=1, so keep this fast-path.
                # Conditioning order here is (pos, neg) based on prompt_embeds_model_input.
                pos = model_out_list[0]
                neg = model_out_list[1]
                pred = neg + float(current_guidance_scale) * (pos - neg)

                # Optional CFG renormalization (do math in fp32 for stability).
                if cfg_normalization and float(cfg_normalization) > 0.0:
                    pos_f = pos.float()
                    pred_f = pred.float()
                    ori_pos_norm = torch.linalg.vector_norm(pos_f)
                    new_pos_norm = torch.linalg.vector_norm(pred_f)
                    max_new_norm = ori_pos_norm * float(cfg_normalization)
                    if new_pos_norm > max_new_norm:
                        pred_f = pred_f * (max_new_norm / new_pos_norm)
                    pred = pred_f.to(dtype=pred.dtype)

                noise_pred = pred.unsqueeze(0)
            else:
                noise_pred = torch.stack([out.float() for out in model_out_list], dim=0)

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
                    # Keep timestep tensor on-device; avoid creating a new CPU tensor each step.
                    image_latent = self.scheduler.scale_noise(self.latents, noise_timestep[None], noise)

                    mask = masks[i].to(latents.dtype)
                    latents = image_latent * mask + latents * (1 - mask)
            # end differential diffusion

            is_last = i == len(timesteps) - 1
            is_step_boundary = (i + 1) > num_warmup_steps and (i + 1) % order == 0
            if (is_last or is_step_boundary) and self.callback is not None:
                self.callback(step_idx, t, latents)

        # Return a detached CPU copy; avoid an extra clone.
        self.values["latents"] = latents.detach().to("cpu")

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
