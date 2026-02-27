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

    def _extract_freefuse_lora_data(self) -> dict[str, tuple[torch.Tensor, str]]:
        """Extract LoRAs that have both spatial mask and trigger words."""
        if not self.lora:
            return {}

        result = {}
        loras = self.lora if isinstance(self.lora, list) else [self.lora]
        for lora_tuple in loras:
            if len(lora_tuple) >= 4:
                adapter_name, _, spatial_mask, trigger_words = lora_tuple[:4]
                if adapter_name is not None and spatial_mask is not None and trigger_words:
                    result[adapter_name] = (spatial_mask, trigger_words)
        return result

    def _extract_auto_mask_lora_data(self) -> dict[str, str]:
        """Extract LoRAs with trigger words but NO spatial mask (Phase B candidates)."""
        if not self.lora:
            return {}

        result = {}
        loras = self.lora if isinstance(self.lora, list) else [self.lora]
        for lora_tuple in loras:
            if len(lora_tuple) >= 4:
                adapter_name, _, spatial_mask, trigger_words = lora_tuple[:4]
                if adapter_name is not None and spatial_mask is None and trigger_words:
                    result[adapter_name] = trigger_words
        return result

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
        "positive_prompt_text",
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
                raise IArtisanZNodeError(
                    f"CUDA out of memory during {description}. No models available to offload.",
                    "Flux2DenoiseNode",
                ) from e

            try:
                return forward_fn()
            except Exception as retry_exc:
                if mm.is_cuda_oom(retry_exc):
                    raise IArtisanZNodeError(
                        f"CUDA out of memory during {description} even after offloading.",
                        "Flux2DenoiseNode",
                    ) from retry_exc
                raise

    @torch.no_grad()
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

        # Phase B: Auto-mask extraction for LoRAs with trigger words but no spatial mask
        auto_mask_loras = self._extract_auto_mask_lora_data()
        auto_derived_masks = {}

        if auto_mask_loras and self.lora:
            try:
                from iartisanz.modules.generation.graph.nodes.freefuse_flux2_auto_mask import (
                    install_flux2_sim_map_collector,
                    remove_flux2_sim_map_collector,
                )
                from iartisanz.modules.generation.graph.nodes.freefuse_flux2_attention_bias import (
                    find_trigger_word_positions_flux2,
                )
                from iartisanz.modules.generation.graph.nodes.freefuse_auto_mask import (
                    process_sim_maps,
                )

                ff_tokenizer = mm.get_raw("tokenizer")
                ff_prompt_text = getattr(self, "positive_prompt_text", None) or ""

                if not ff_prompt_text:
                    logger.warning(
                        "[Flux2DenoiseNode] Phase B skipped: positive_prompt_text not wired. "
                        "Rebuild graph (switch model type and back) to enable FreeFuse auto-masks."
                    )
                elif ff_tokenizer:
                    # Derive spatial dims from latent_ids
                    patch_h = int(self.latent_ids[0, :, 1].max().item()) + 1
                    patch_w = int(self.latent_ids[0, :, 2].max().item()) + 1
                    img_seq_len = patch_h * patch_w
                    txt_seq_len = self.prompt_embeds.shape[1]

                    auto_token_pos_maps = {}
                    for adapter_name, trigger_words in auto_mask_loras.items():
                        positions = find_trigger_word_positions_flux2(
                            ff_tokenizer, ff_prompt_text, trigger_words
                        )
                        if positions:
                            auto_token_pos_maps[adapter_name] = positions

                    if auto_token_pos_maps:
                        # Install collector on last double-stream block
                        double_blocks = getattr(transformer_raw, "transformer_blocks", [])
                        block_idx = max(0, len(double_blocks) - 1)

                        collector_state = install_flux2_sim_map_collector(
                            transformer_raw, block_idx, auto_token_pos_maps,
                            txt_seq_len=txt_seq_len,
                        )

                        # Disable LoRA for Phase 1
                        transformer_raw.disable_adapters()

                        try:
                            phase1_steps = 3
                            # Compute sigmas and timesteps for Phase 1
                            p1_sigmas = np.linspace(1.0, 1.0 / num_inference_steps, num_inference_steps).tolist()
                            p1_mu = compute_empirical_mu(img_seq_len, num_inference_steps)
                            p1_timesteps, _ = self.retrieve_timesteps(
                                self.scheduler, num_inference_steps, self.device,
                                sigmas=p1_sigmas, mu=p1_mu,
                            )
                            phase1_steps = min(phase1_steps, len(p1_timesteps))
                            phase1_latents = self.latents.to(self.device)
                            p1_latent_ids = self.latent_ids.to(self.device)
                            p1_prompt_embeds = self.prompt_embeds.to(
                                device=self.device, dtype=transformer_dtype,
                            )
                            p1_text_ids = self.text_ids.to(self.device)

                            logger.info(
                                "[Flux2DenoiseNode] Phase B: collecting sim maps (%d steps, double block %d)",
                                phase1_steps, block_idx,
                            )

                            collector = collector_state["collector"]
                            self.scheduler.set_begin_index(0)

                            for p1_i, p1_t in enumerate(p1_timesteps[:phase1_steps]):
                                collector.cal_concept_sim_map = (p1_i == phase1_steps - 1)
                                p1_timestep = p1_t.expand(phase1_latents.shape[0]).to(phase1_latents.dtype)

                                p1_latent_input = phase1_latents.to(transformer_dtype)

                                p1_noise_pred = transformer_raw(
                                    hidden_states=p1_latent_input,
                                    timestep=p1_timestep / 1000,
                                    guidance=None,
                                    encoder_hidden_states=p1_prompt_embeds,
                                    txt_ids=p1_text_ids,
                                    img_ids=p1_latent_ids,
                                    return_dict=False,
                                )[0]
                                p1_noise_pred = p1_noise_pred[:, :phase1_latents.size(1)]

                                phase1_latents = self.scheduler.step(
                                    p1_noise_pred, p1_t, phase1_latents, return_dict=False
                                )[0]

                            concept_sim_maps = collector.concept_sim_maps
                        finally:
                            remove_flux2_sim_map_collector(collector_state)
                            transformer_raw.enable_adapters()

                        if concept_sim_maps:
                            logger.debug(
                                "[Flux2DenoiseNode] Phase B sim maps collected: %s",
                                {k: v.shape for k, v in concept_sim_maps.items()},
                            )
                            auto_derived_masks = process_sim_maps(
                                concept_sim_maps, patch_h, patch_w, img_seq_len,
                            )
                            if auto_derived_masks:
                                for mn, mv in auto_derived_masks.items():
                                    logger.info(
                                        "[Flux2DenoiseNode] Phase B mask '%s': "
                                        "coverage=%.1f%% (%d/%d tokens)",
                                        mn, mv.sum().item() / mv.numel() * 100,
                                        int(mv.sum().item()), mv.numel(),
                                    )
                            else:
                                logger.warning(
                                    "[Flux2DenoiseNode] Phase B: process_sim_maps returned empty masks"
                                )
                        else:
                            logger.warning(
                                "[Flux2DenoiseNode] Phase B: no concept sim maps collected"
                            )

            except Exception:
                logger.exception("[Flux2DenoiseNode] Failed Phase B auto-mask extraction")
                auto_derived_masks = {}

        # Apply Phase B auto-derived masks as LoRA spatial masks
        if auto_derived_masks:
            from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
                patch_lora_adapter_with_spatial_mask,
            )

            patch_h = int(self.latent_ids[0, :, 1].max().item()) + 1
            patch_w = int(self.latent_ids[0, :, 2].max().item()) + 1
            latent_spatial_dims = (patch_h, patch_w)
            for adapter_name, flat_mask in auto_derived_masks.items():
                spatial_mask_2d = flat_mask.view(1, 1, patch_h, patch_w)
                patch_lora_adapter_with_spatial_mask(
                    transformer_raw, adapter_name, spatial_mask_2d,
                    latent_spatial_dims=latent_spatial_dims,
                )
            lora_masks_applied = True
            logger.info(
                "[Flux2DenoiseNode] Phase B: applied auto-derived spatial masks: %s",
                list(auto_derived_masks.keys()),
            )

        # Build FreeFuse attention bias (Phase A manual + Phase B auto masks)
        freefuse_patched = None
        if self.lora:
            freefuse_loras = self._extract_freefuse_lora_data()
            has_freefuse_data = bool(freefuse_loras) or bool(auto_derived_masks)

            if has_freefuse_data:
                try:
                    from iartisanz.modules.generation.graph.nodes.freefuse_flux2_attention_bias import (
                        construct_flux2_attention_bias,
                        find_trigger_word_positions_flux2,
                        patch_flux2_attention_with_bias,
                    )

                    tokenizer = mm.get_raw("tokenizer")
                    prompt_text = getattr(self, "positive_prompt_text", None) or ""

                    if not prompt_text:
                        logger.warning(
                            "[Flux2DenoiseNode] FreeFuse skipped: positive_prompt_text not wired. "
                            "Rebuild graph (switch model type and back) to enable FreeFuse."
                        )
                    elif tokenizer:
                        patch_h = int(self.latent_ids[0, :, 1].max().item()) + 1
                        patch_w = int(self.latent_ids[0, :, 2].max().item()) + 1
                        img_seq_len = patch_h * patch_w
                        txt_seq_len = self.prompt_embeds.shape[1]

                        token_pos_maps = {}
                        flat_masks = {}

                        # Phase A: manual masks with trigger words
                        for adapter_name, (mask, trigger_words) in freefuse_loras.items():
                            positions = find_trigger_word_positions_flux2(
                                tokenizer, prompt_text, trigger_words
                            )
                            if positions:
                                token_pos_maps[adapter_name] = positions
                                resized = F.interpolate(
                                    mask.float().unsqueeze(0).unsqueeze(0) if mask.dim() == 2
                                    else mask.float(),
                                    size=(patch_h, patch_w),
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                flat_masks[adapter_name] = resized.flatten(1)

                        # Phase B: auto-derived masks (don't override manual)
                        for adapter_name, flat_mask in auto_derived_masks.items():
                            if adapter_name not in flat_masks:
                                trigger_words = auto_mask_loras.get(adapter_name, "")
                                positions = find_trigger_word_positions_flux2(
                                    tokenizer, prompt_text, trigger_words
                                )
                                if positions:
                                    token_pos_maps[adapter_name] = positions
                                    flat_masks[adapter_name] = flat_mask

                        if token_pos_maps and flat_masks:
                            attention_bias = construct_flux2_attention_bias(
                                lora_masks=flat_masks,
                                token_pos_maps=token_pos_maps,
                                txt_seq_len=txt_seq_len,
                                img_seq_len=img_seq_len,
                                device=self.device,
                                dtype=self.latents.dtype,
                            )
                            freefuse_patched = patch_flux2_attention_with_bias(
                                transformer_raw, attention_bias
                            )
                            logger.info(
                                "[Flux2DenoiseNode] FreeFuse attention bias applied: "
                                "img_seq=%d, txt_seq=%d, loras=%s",
                                img_seq_len, txt_seq_len, list(token_pos_maps.keys()),
                            )
                except Exception:
                    logger.exception("[Flux2DenoiseNode] Failed to build FreeFuse attention bias")
                    freefuse_patched = None

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

        # Clean up FreeFuse attention bias patches
        if freefuse_patched:
            from iartisanz.modules.generation.graph.nodes.freefuse_flux2_attention_bias import (
                unpatch_flux2_attention_bias,
            )
            unpatch_flux2_attention_bias(freefuse_patched)
            logger.debug("[Flux2DenoiseNode] Cleaned up FreeFuse attention bias patches")

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
