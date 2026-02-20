import inspect
import logging
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node
from iartisanz.utils.image_converters import numpy_to_pt


logger = logging.getLogger(__name__)


def compute_empirical_mu(image_seq_len: int, num_steps: int) -> float:
    """Compute mu for sigma scheduling based on image sequence length and step count.

    Uses piecewise linear interpolation with empirically derived constants.
    """
    a1, b1 = 8.73809524e-05, 1.89833333
    a2, b2 = 0.00016927, 0.45666666

    if image_seq_len > 4300:
        return float(a2 * image_seq_len + b2)

    m_200 = a2 * image_seq_len + b2
    m_10 = a1 * image_seq_len + b1

    a = (m_200 - m_10) / 190.0
    b = m_200 - 200.0 * a
    mu = a * num_steps + b

    return float(mu)


class Flux2DenoiseNode(Node):
    """Denoise node for Flux.2 Klein models.

    Key differences from ZImageDenoiseNode:
    - Latents are in packed sequence format (B, seq_len, C)
    - Position IDs (img_ids, txt_ids) are required
    - Timestep normalization: t / 1000
    - CFG uses separate forward passes (not batch doubling)
    - Sigma scheduling with empirical mu
    - No ControlNet support
    - No noise negation
    """

    def _extract_lora_masks(self) -> dict[str, torch.Tensor]:
        """Extract spatial masks from LoRA inputs.

        LoRA tuples can be 2-tuples (legacy) or 3-tuples (with mask).
        - 2-tuple: (adapter_name, scale_dict)
        - 3-tuple: (adapter_name, scale_dict, spatial_mask)

        Returns:
            Dict mapping adapter_name -> mask_tensor
            Empty dict if no masks present
        """
        if not self.lora:
            return {}

        masks = {}

        if isinstance(self.lora, list):
            # Multiple LoRAs
            for lora_tuple in self.lora:
                if len(lora_tuple) >= 3 and lora_tuple[0] is not None and lora_tuple[2] is not None:
                    adapter_name = lora_tuple[0]
                    mask = lora_tuple[2]
                    masks[adapter_name] = mask
        else:
            # Single LoRA
            if len(self.lora) >= 3 and self.lora[0] is not None and self.lora[2] is not None:
                adapter_name = self.lora[0]
                mask = self.lora[2]
                masks[adapter_name] = mask

        return masks

    REQUIRED_INPUTS = [
        "transformer",
        "num_inference_steps",
        "latents",
        "latent_ids",
        "scheduler",
        "prompt_embeds",
        "text_ids",
        "guidance_scale",
    ]
    OPTIONAL_INPUTS = [
        "negative_prompt_embeds",
        "negative_text_ids",
        "guidance_start_end",
        "lora",
        "edit_image_latents",
        "edit_image_latent_ids",
        "edit_image_base_latents",
        "edit_image_mask",
        "edit_image_mask_strength",
    ]
    OUTPUTS = ["latents", "latent_ids"]
    SERIALIZE_EXCLUDE = {"callback"}

    def __init__(self, callback: callable = None):
        super().__init__()
        self.callback = callback

    @staticmethod
    def _run_with_oom_retry(
        mm,
        forward_fn: callable,
        preserve: tuple = (),
        description: str = "forward pass",
    ):
        """Run a forward pass with single OOM retry after offloading models."""
        try:
            return forward_fn()
        except Exception as e:
            if not mm.offload_on_cuda_oom or not mm.is_cuda_oom(e):
                raise

            logger.warning(f"[Flux2DenoiseNode] CUDA OOM during {description}. Offloading models and retrying...")
            offloaded = mm.free_vram_for_forward_pass(preserve=preserve)
            if offloaded == 0:
                logger.error("[Flux2DenoiseNode] No models available to offload for OOM recovery.")
                raise

            return forward_fn()

    @torch.inference_mode()
    def __call__(self):
        mm = get_model_manager()
        transformer_input = self.transformer
        transformer_raw = mm.resolve(transformer_input)

        use_torch_compile = mm.use_torch_compile
        transformer = transformer_raw

        mm.apply_attention_backend(transformer_raw)

        if use_torch_compile and not mm.is_attention_backend_compile_compatible(transformer_raw):
            logger.warning(
                f"Attention backend '{mm.attention_backend}' is not compatible with torch.compile. "
                "Disabling compilation for this generation."
            )
            use_torch_compile = False

        if (
            (not use_torch_compile)
            and isinstance(transformer_input, ModelHandle)
            and transformer_input.component == "transformer"
        ):
            try:
                mm.disable_compiled("transformer")
            except Exception:
                pass

        transformer_dtype = getattr(transformer_raw, "dtype", None) or getattr(transformer, "dtype", None)
        if transformer_dtype is None:
            try:
                for p in transformer_raw.parameters(recurse=True):
                    transformer_dtype = p.dtype
                    break
            except Exception:
                pass
        if transformer_dtype is None:
            transformer_dtype = self.dtype or torch.float32

        guidance_scale = float(self.guidance_scale)
        do_cfg = guidance_scale > 1.0

        guidance_start_end = getattr(self, "guidance_start_end", None)
        if guidance_start_end is None:
            guidance_start, guidance_end = 0.0, 1.0
        else:
            guidance_start = float(guidance_start_end[0])
            guidance_end = float(guidance_start_end[1])

        if guidance_start > guidance_end:
            logger.warning(
                "guidance_start (%s) > guidance_end (%s); using full range [0.0, 1.0].",
                guidance_start,
                guidance_end,
            )
            guidance_start, guidance_end = 0.0, 1.0

        num_inference_steps = int(self.num_inference_steps)

        lora_masks_applied = False

        if self.lora:
            try:
                if isinstance(self.lora, list):
                    # Filter LoKr entries (adapter_name=None) — weight-merged, not PEFT
                    peft_items = [item for item in self.lora if item[0] is not None]
                    if peft_items:
                        keys = [item[0] for item in peft_items]
                        transformer_values = [item[1]["transformer"] for item in peft_items]
                        transformer_raw.set_adapters(keys, transformer_values)
                else:
                    if self.lora[0] is not None:
                        transformer_raw.set_adapters([self.lora[0]], self.lora[1]["transformer"])
            except RuntimeError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)

            # LoRA spatial masking (Phase A — manual masks)
            lora_masks = self._extract_lora_masks()
            if lora_masks:
                from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import patch_lora_adapter_with_spatial_mask

                # Derive patch-space dims from latent_ids (T, H, W, L) — still on CPU, fine for .max()
                patch_h = int(self.latent_ids[0, :, 1].max().item()) + 1
                patch_w = int(self.latent_ids[0, :, 2].max().item()) + 1
                latent_spatial_dims = (patch_h, patch_w)

                for adapter_name, mask in lora_masks.items():
                    patch_lora_adapter_with_spatial_mask(
                        transformer_raw, adapter_name, mask, latent_spatial_dims
                    )
                    logger.debug(
                        "[Flux2DenoiseNode] Applied spatial mask for LoRA '%s' at patch dims %s",
                        adapter_name,
                        latent_spatial_dims,
                    )
                lora_masks_applied = True

        if (
            use_torch_compile
            and isinstance(transformer_input, ModelHandle)
            and transformer_input.component == "transformer"
        ):
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

        prompt_embeds = self.prompt_embeds.to(self.device, dtype=transformer_dtype)
        text_ids = self.text_ids.to(self.device)

        negative_prompt_embeds = None
        negative_text_ids = None
        if do_cfg:
            neg_embeds_input = getattr(self, "negative_prompt_embeds", None)
            neg_ids_input = getattr(self, "negative_text_ids", None)
            if neg_embeds_input is not None:
                negative_prompt_embeds = neg_embeds_input.to(self.device, dtype=transformer_dtype)
            if neg_ids_input is not None:
                negative_text_ids = neg_ids_input.to(self.device)

        latents = self.latents.to(self.device)
        latent_ids = self.latent_ids.to(self.device)

        # Edit images (optional): pre-encoded latents from Flux2EditImageEncodeNode
        edit_image_latents = getattr(self, "edit_image_latents", None)
        edit_image_latent_ids = getattr(self, "edit_image_latent_ids", None)
        has_edit_images = edit_image_latents is not None and edit_image_latent_ids is not None
        if has_edit_images:
            edit_image_latents = edit_image_latents.to(self.device, dtype=transformer_dtype)
            edit_image_latent_ids = edit_image_latent_ids.to(self.device)

        # Compute sigmas and empirical mu for Flux2 scheduling
        sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps).tolist()
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len, num_inference_steps)

        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            sigmas=sigmas,
            mu=mu,
        )

        # Differential diffusion setup (edit image inpainting mask)
        edit_base_latents = getattr(self, "edit_image_base_latents", None)
        edit_mask_np = getattr(self, "edit_image_mask", None)
        has_dd = edit_base_latents is not None and edit_mask_np is not None
        dd_masks = None
        initial_noise = None
        original_base = None

        if has_dd:
            original_base = edit_base_latents.to(self.device, dtype=latents.dtype)
            initial_noise = latents.clone()

            # Mask from dialog: painted dark (0) = change, unpainted white (1) = preserve.
            # This matches Z-Image DD convention — use directly as preserve_mask.
            mask_t = numpy_to_pt(edit_mask_np[None, ...]).to(self.device, dtype=latents.dtype)
            preserve_mask = mask_t  # (1, 1, H, W)

            # Apply strength: 1.0 = full DD, 0.0 = ignore mask (all change)
            mask_strength = getattr(self, "edit_image_mask_strength", None)
            if mask_strength is not None:
                preserve_mask = preserve_mask * float(mask_strength)

            # Derive patchified spatial dims from latent_ids (T, H, W, L)
            patch_h = int(latent_ids[0, :, 1].max().item()) + 1
            patch_w = int(latent_ids[0, :, 2].max().item()) + 1

            preserve_mask = F.interpolate(
                preserve_mask, size=(patch_h, patch_w), mode="bilinear", align_corners=False
            )
            preserve_mask = preserve_mask.reshape(1, -1)  # (1, seq_len)

            # Per-step threshold schedule
            mask_thresholds = (
                torch.arange(num_inference_steps, dtype=preserve_mask.dtype) / num_inference_steps
            )
            mask_thresholds = mask_thresholds.reshape(-1, 1).to(self.device)
            dd_masks = (preserve_mask > mask_thresholds).float()  # (steps, seq_len)

            # Start from noised edit image (like img2img at full strength)
            latents = self.scheduler.scale_noise(original_base, timesteps[:1], initial_noise)

        order = getattr(self.scheduler, "order", 1)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * order, 0)
        step_norm_den = float(max(num_inference_steps - 1, 1))

        # Mark step boundaries for CUDA graphs
        if use_torch_compile and self.device.type == "cuda" and hasattr(torch, "compiler"):
            mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        else:
            mark = None

        self.scheduler.set_begin_index(0)

        for i, t in enumerate(timesteps):
            if self.abort:
                return

            step_idx = i // order
            step_norm = float(step_idx) / step_norm_den

            current_guidance_scale = guidance_scale
            if not (guidance_start <= step_norm <= guidance_end):
                current_guidance_scale = 1.0

            apply_cfg = do_cfg and negative_prompt_embeds is not None and (abs(current_guidance_scale - 1.0) > 1e-6)

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if has_edit_images:
                latent_model_input = torch.cat([latents, edit_image_latents], dim=1).to(transformer_dtype)
                loop_latent_ids = torch.cat([latent_ids, edit_image_latent_ids], dim=1)
            else:
                latent_model_input = latents.to(transformer_dtype)
                loop_latent_ids = latent_ids

            if mark is not None and callable(mark):
                mark()

            # Conditional forward pass
            def _run_cond():
                ctx = getattr(transformer, "cache_context", None)
                if ctx is not None:
                    with ctx("cond"):
                        return transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=None,
                            encoder_hidden_states=prompt_embeds,
                            txt_ids=text_ids,
                            img_ids=loop_latent_ids,
                            return_dict=False,
                        )[0]
                return transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=loop_latent_ids,
                    return_dict=False,
                )[0]

            noise_pred = self._run_with_oom_retry(
                mm,
                _run_cond,
                preserve=("transformer",),
                description="transformer (cond)",
            )

            # Only keep the generated latent tokens
            noise_pred = noise_pred[:, : latents.size(1)]

            # Classifier-free guidance (gated by guidance window)
            if apply_cfg:
                def _run_uncond():
                    ctx = getattr(transformer, "cache_context", None)
                    if ctx is not None:
                        with ctx("uncond"):
                            return transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=None,
                                encoder_hidden_states=negative_prompt_embeds,
                                txt_ids=negative_text_ids,
                                img_ids=loop_latent_ids,
                                return_dict=False,
                            )[0]
                    return transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=loop_latent_ids,
                        return_dict=False,
                    )[0]

                neg_noise_pred = self._run_with_oom_retry(
                    mm,
                    _run_uncond,
                    preserve=("transformer",),
                    description="transformer (uncond)",
                )
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                noise_pred = neg_noise_pred + current_guidance_scale * (noise_pred - neg_noise_pred)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Differential diffusion blending
            if has_dd and dd_masks is not None:
                step_mask = dd_masks[i].unsqueeze(0).unsqueeze(-1)  # (1, seq_len, 1)
                if i < len(timesteps) - 1:
                    image_latent = self.scheduler.scale_noise(
                        original_base, timesteps[i + 1 : i + 2], initial_noise
                    )
                else:
                    image_latent = original_base
                # mask=1 -> preserve original, mask=0 -> use denoised
                latents = image_latent * step_mask + latents * (1 - step_mask)

            # Step callback
            is_last = i == len(timesteps) - 1
            is_step_boundary = (i + 1) > num_warmup_steps and (i + 1) % order == 0
            if (is_last or is_step_boundary) and self.callback is not None:
                self.callback(step_idx, t, latents)

        if lora_masks_applied:
            from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import unpatch_all_lora_layers
            unpatch_count = unpatch_all_lora_layers()
            if unpatch_count > 0:
                logger.debug(f"[Flux2DenoiseNode] Cleaned up {unpatch_count} LoRA spatial mask patches")

        self.values["latents"] = latents.detach().to("cpu")
        self.values["latent_ids"] = self.latent_ids.detach().to("cpu")

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
            raise ValueError("Only one of `timesteps` or `sigmas` can be passed.")

        if timesteps is not None:
            accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accepts_timesteps:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    " timestep schedules."
                )
            scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        elif sigmas is not None:
            accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
            if not accept_sigmas:
                raise ValueError(
                    f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                    " sigmas schedules."
                )
            scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
            timesteps = scheduler.timesteps
            num_inference_steps = len(timesteps)
        else:
            scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
            timesteps = scheduler.timesteps
        return timesteps, num_inference_steps
