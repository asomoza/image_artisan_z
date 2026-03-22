import inspect
import logging
from typing import Dict, Optional, Union

import torch
import torch.nn.functional as F

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node
from iartisanz.utils.image_converters import numpy_to_pt


logger = logging.getLogger(__name__)


class ZImageDenoiseNode(Node):
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
        "source_mask",
        "controlnet",
        "control_image_latents",
        "controlnet_conditioning_scale",
        "controlnet_spatial_mask",
        "control_guidance_start_end",
        "control_mode",
        "prompt_mode_decay",
        "positive_prompt_text",
    ]
    OUTPUTS = ["latents"]
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
        """Run a forward pass with single OOM retry after offloading models.

        Args:
            mm: ModelManager instance.
            forward_fn: Callable that executes the forward pass.
            preserve: Model components that should NOT be offloaded on retry.
            description: Description for logging purposes.

        Returns:
            Result of forward_fn().

        Raises:
            Original exception if not OOM or if retry also fails.
        """
        try:
            return forward_fn()
        except Exception as e:
            if not mm.offload_on_cuda_oom or not mm.is_cuda_oom(e):
                raise

            logger.warning(f"[DenoiseNode] CUDA OOM during {description}. Offloading models and retrying...")
            offloaded = mm.free_vram_for_forward_pass(preserve=preserve)
            if offloaded == 0:
                logger.error("[DenoiseNode] No models available to offload for OOM recovery.")
                raise IArtisanZNodeError(
                    f"CUDA out of memory during {description}. No models available to offload.",
                    "ZImageDenoiseNode",
                ) from e

            try:
                return forward_fn()
            except Exception as retry_exc:
                if mm.is_cuda_oom(retry_exc):
                    raise IArtisanZNodeError(
                        f"CUDA out of memory during {description} even after offloading.",
                        "ZImageDenoiseNode",
                    ) from retry_exc
                raise

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

    @staticmethod
    def _apply_spatial_mask(block_samples: dict, spatial_mask: torch.Tensor, latent_shape: tuple) -> dict:
        """Apply spatial mask to ControlNet block samples to restrict where they affect generation.

        Args:
            block_samples: Dict of layer_idx -> block tensor
            spatial_mask: Mask tensor (H, W) or (1, 1, H, W) in pixel space
            latent_shape: Shape of latents (B, C, H, W)

        Returns:
            Modified block_samples dict with spatial mask applied
        """
        import logging

        logger = logging.getLogger(__name__)

        try:
            # Ensure mask is in the right format (B, 1, H, W) - MUST be 4D
            if spatial_mask.ndim == 2:
                spatial_mask = spatial_mask.unsqueeze(0).unsqueeze(0)
            elif spatial_mask.ndim == 3:
                spatial_mask = spatial_mask.unsqueeze(1)
            elif spatial_mask.ndim > 4:
                logger.error(
                    f"[DenoiseNode._apply_spatial_mask] Spatial mask has too many dimensions: "
                    f"{spatial_mask.ndim}D (shape={spatial_mask.shape}). Expected 2D, 3D, or 4D. "
                    f"Squeezing extra dimensions."
                )
                # Squeeze extra dimensions
                while spatial_mask.ndim > 4:
                    spatial_mask = spatial_mask.squeeze(0)

            # Apply mask to each block sample
            # NOTE: Each block may have different spatial dimensions, so we resize mask per-block
            for layer_idx, block_tensor in block_samples.items():
                try:
                    # Get the spatial dimensions of this block
                    block_h, block_w = block_tensor.shape[-2], block_tensor.shape[-1]

                    # Resize mask to match THIS block's spatial dimensions
                    mask_for_block = torch.nn.functional.interpolate(
                        spatial_mask, size=(block_h, block_w), mode="bilinear", align_corners=False
                    )

                    # Move mask to same device and dtype as block
                    mask_for_block = mask_for_block.to(device=block_tensor.device, dtype=block_tensor.dtype)

                    # CRITICAL: Match the dimensionality of the block tensor
                    # If block is 3D [B, H, W], squeeze mask from [1, 1, H, W] to [1, H, W]
                    # If block is 4D [B, C, H, W], keep mask as [1, 1, H, W]
                    while mask_for_block.ndim > block_tensor.ndim:
                        mask_for_block = mask_for_block.squeeze(1)

                    # Apply mask: keep ControlNet influence only where mask > 0
                    # Broadcasting will handle remaining dimensions automatically
                    masked_block = block_tensor * mask_for_block

                    # Convert to dense if sparse (sparse tensors can cause issues in transformer)
                    if masked_block.is_sparse:
                        logger.warning(
                            f"[DenoiseNode._apply_spatial_mask] Block {layer_idx} became sparse, converting to dense"
                        )
                        masked_block = masked_block.to_dense()

                    # Verify we didn't change dimensionality
                    if masked_block.ndim != block_tensor.ndim:
                        logger.error(
                            f"[DenoiseNode._apply_spatial_mask] Dimension mismatch! "
                            f"Original: {block_tensor.ndim}D ({block_tensor.shape}), "
                            f"After mask: {masked_block.ndim}D ({masked_block.shape})"
                        )
                        # Don't apply mask if dimensions changed
                        continue

                    block_samples[layer_idx] = masked_block

                except Exception as e:
                    # If masking fails for a particular layer, keep original
                    logger.warning(
                        f"[DenoiseNode._apply_spatial_mask] Failed to mask layer {layer_idx} "
                        f"(shape={block_tensor.shape}): {e}"
                    )

        except Exception as e:
            # If overall masking fails, return original block_samples
            logger.error(f"[DenoiseNode._apply_spatial_mask] Failed to apply spatial mask: {e}")

        return block_samples

    def _extract_lora_masks(self) -> Dict[str, torch.Tensor]:
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

    @torch.no_grad()
    def __call__(self):
        mm = get_model_manager()

        with mm.use_components("transformer", device=self.device):
            return self._run_denoise(mm)

    def _run_denoise(self, mm):
        transformer_input = self.transformer
        transformer_raw = mm.resolve(transformer_input)

        # Read use_torch_compile from ModelManager (runtime config, not saved in graph)
        use_torch_compile = mm.use_torch_compile
        transformer = transformer_raw

        # Apply attention backend (runtime config, not saved in graph).
        # This should be done BEFORE torch.compile to ensure the backend is set
        # on the raw module (compilation will preserve the backend setting).
        mm.apply_attention_backend(transformer_raw)

        # Some attention backends (e.g., sage_varlen) are not compatible with torch.compile
        # due to using operations like torch.cuda.set_device() that cause graph breaks.
        if use_torch_compile and not mm.is_attention_backend_compile_compatible(transformer_raw):
            logger.warning(
                f"Attention backend '{mm.attention_backend}' is not compatible with torch.compile. "
                "Disabling compilation for this generation."
            )
            use_torch_compile = False

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

        # Track if we applied LoRA spatial masks (for cleanup)
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

                # Extract and apply LoRA spatial masks
                lora_masks = self._extract_lora_masks()
                if lora_masks:
                    from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
                        patch_lora_adapter_with_spatial_mask,
                    )

                    # Get latent spatial dimensions for non-square aspect ratio support
                    latent_spatial_dims = (self.latents.shape[2], self.latents.shape[3])

                    for adapter_name, mask in lora_masks.items():
                        patch_lora_adapter_with_spatial_mask(
                            transformer_raw, adapter_name, mask, latent_spatial_dims=latent_spatial_dims
                        )
                    lora_masks_applied = True

            except RuntimeError as e:
                raise IArtisanZNodeError(e, self.__class__.__name__)

        # Phase B: Auto-mask extraction for LoRAs with trigger words but no spatial mask
        auto_mask_loras = self._extract_auto_mask_lora_data()
        auto_derived_masks = {}
        phase_b_total_seq = None

        if auto_mask_loras and self.lora:
            try:
                from iartisanz.modules.generation.graph.nodes.freefuse_auto_mask import (
                    install_sim_map_collector,
                    process_sim_maps,
                    remove_sim_map_collector,
                )
                from iartisanz.modules.generation.graph.nodes.freefuse_attention_bias import (
                    find_trigger_word_positions,
                )

                ff_tokenizer = mm.get_raw("tokenizer")
                ff_prompt_text = getattr(self, "positive_prompt_text", None) or ""

                if not ff_prompt_text:
                    logger.warning(
                        "[DenoiseNode] Phase B skipped: positive_prompt_text not wired. "
                        "Rebuild graph (switch model type and back) to enable FreeFuse auto-masks."
                    )
                elif ff_tokenizer:
                    patch_size = 2
                    img_h = self.latents.shape[2] // patch_size
                    img_w = self.latents.shape[3] // patch_size
                    img_ori_len = img_h * img_w
                    cap_ori_len = self.prompt_embeds.shape[1]

                    auto_token_pos_maps = {}
                    for adapter_name, trigger_words in auto_mask_loras.items():
                        positions = find_trigger_word_positions(
                            ff_tokenizer, ff_prompt_text, trigger_words
                        )
                        if positions:
                            auto_token_pos_maps[adapter_name] = positions

                    if auto_token_pos_maps:
                        block_idx = min(18, len(transformer_raw.layers) - 1)
                        collector_state = install_sim_map_collector(
                            transformer_raw, block_idx, auto_token_pos_maps,
                            img_seq_len=img_ori_len, cap_seq_len=cap_ori_len,
                        )

                        # Disable LoRA for Phase 1 — collect base model attention
                        transformer_raw.disable_adapters()

                        try:
                            phase1_steps = 3
                            p1_retrieve_kwargs = {}
                            if getattr(self.scheduler.config, "use_dynamic_shifting", False):
                                p1_image_seq_len = (self.latents.shape[2] // 2) * (self.latents.shape[3] // 2)
                                p1_retrieve_kwargs["mu"] = self._calculate_shift(p1_image_seq_len)
                            p1_timesteps, _ = self.retrieve_timesteps(
                                self.scheduler, num_inference_steps, self.device,
                                sigmas=self.sigmas, **p1_retrieve_kwargs,
                            )
                            phase1_steps = min(phase1_steps, len(p1_timesteps))
                            phase1_latents = self.latents.clone()
                            p1_prompt_embeds = self.prompt_embeds.to(
                                device=self.device, dtype=transformer_dtype,
                            )

                            logger.info(
                                "[DenoiseNode] Phase B: collecting sim maps (%d steps, block %d)",
                                phase1_steps, block_idx,
                            )

                            collector = collector_state["collector"]
                            for i, t in enumerate(p1_timesteps[:phase1_steps]):
                                collector.cal_concept_sim_map = (
                                    i == phase1_steps - 1
                                )
                                timestep = t.expand(phase1_latents.shape[0])
                                timestep = (1000 - timestep) / 1000

                                lat = phase1_latents.to(transformer_dtype).unsqueeze(2)
                                lat_list = lat.unbind(dim=0)

                                model_out = transformer_raw(
                                    lat_list, timestep, (p1_prompt_embeds,),
                                    return_dict=False,
                                )[0]
                                noise_pred_p1 = torch.stack(
                                    [o.float() for o in model_out], dim=0
                                )
                                noise_pred_p1 = noise_pred_p1.squeeze(2)
                                noise_pred_p1 = -noise_pred_p1
                                phase1_latents = self.scheduler.step(
                                    noise_pred_p1.to(torch.float32), t,
                                    phase1_latents, return_dict=False,
                                )[0]

                            concept_sim_maps = collector.concept_sim_maps
                            phase_b_total_seq = collector.total_seq
                        finally:
                            remove_sim_map_collector(collector_state)
                            transformer_raw.enable_adapters()

                        if concept_sim_maps:
                            logger.debug(
                                "[DenoiseNode] Phase B sim maps collected: %s",
                                {k: v.shape for k, v in concept_sim_maps.items()},
                            )
                            auto_derived_masks = process_sim_maps(
                                concept_sim_maps, img_h, img_w, img_ori_len,
                            )
                            if auto_derived_masks:
                                for mn, mv in auto_derived_masks.items():
                                    logger.info(
                                        "[DenoiseNode] Phase B mask '%s': "
                                        "coverage=%.1f%% (%d/%d tokens)",
                                        mn, mv.sum().item() / mv.numel() * 100,
                                        int(mv.sum().item()), mv.numel(),
                                    )
                            else:
                                logger.warning(
                                    "[DenoiseNode] Phase B: process_sim_maps returned empty masks"
                                )
                        else:
                            logger.warning(
                                "[DenoiseNode] Phase B: no concept sim maps collected"
                            )

            except Exception:
                logger.exception("[DenoiseNode] Failed Phase B auto-mask extraction")
                auto_derived_masks = {}

        # Apply Phase B auto-derived masks as LoRA spatial masks
        # (equivalent to FreeFuseLinear / set_freefuse_masks in the reference)
        if auto_derived_masks:
            from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
                patch_lora_adapter_with_spatial_mask,
            )

            _patch_size = 2
            _img_h = self.latents.shape[2] // _patch_size
            _img_w = self.latents.shape[3] // _patch_size
            _latent_dims = (self.latents.shape[2], self.latents.shape[3])
            for adapter_name, flat_mask in auto_derived_masks.items():
                spatial_mask_2d = flat_mask.view(1, 1, _img_h, _img_w)
                patch_lora_adapter_with_spatial_mask(
                    transformer_raw, adapter_name, spatial_mask_2d,
                    latent_spatial_dims=_latent_dims,
                )
            lora_masks_applied = True
            logger.info(
                "[DenoiseNode] Phase B: applied auto-derived spatial masks to LoRA layers: %s",
                list(auto_derived_masks.keys()),
            )

        # Build FreeFuse attention bias (Phase A manual + Phase B auto masks)
        freefuse_patched = None
        if self.lora:
            freefuse_loras = self._extract_freefuse_lora_data()
            has_freefuse_data = bool(freefuse_loras) or bool(auto_derived_masks)

            if has_freefuse_data:
                try:
                    from iartisanz.modules.generation.graph.nodes.freefuse_attention_bias import (
                        _SEQ_MULTI_OF,
                        _ceil_to_multiple,
                        construct_attention_bias,
                        find_trigger_word_positions,
                        patch_attention_with_bias,
                    )

                    tokenizer = mm.get_raw("tokenizer")
                    prompt_text = getattr(self, "positive_prompt_text", None) or ""

                    if not prompt_text:
                        logger.warning(
                            "[DenoiseNode] FreeFuse skipped: positive_prompt_text not wired. "
                            "Rebuild graph (switch model type and back) to enable FreeFuse."
                        )
                    elif tokenizer:
                        patch_size = 2
                        img_h = self.latents.shape[2] // patch_size
                        img_w = self.latents.shape[3] // patch_size
                        img_ori_len = img_h * img_w
                        img_seq_len = _ceil_to_multiple(img_ori_len, _SEQ_MULTI_OF)

                        prompt_embeds_for_len = self.prompt_embeds
                        cap_ori_len = prompt_embeds_for_len.shape[1]

                        # Use actual caption length from Phase B collector when available.
                        # The transformer may compress prompt_embeds (e.g. 2560 → 32 tokens),
                        # so prompt_embeds.shape[1] can be much larger than reality.
                        if phase_b_total_seq is not None:
                            txt_seq_len = phase_b_total_seq - img_seq_len
                        else:
                            txt_seq_len = _ceil_to_multiple(cap_ori_len, _SEQ_MULTI_OF)

                        token_pos_maps = {}
                        flat_masks = {}

                        # Phase A: manual masks with trigger words
                        for adapter_name, (mask, trigger_words) in freefuse_loras.items():
                            positions = find_trigger_word_positions(tokenizer, prompt_text, trigger_words)
                            if positions:
                                token_pos_maps[adapter_name] = positions
                                resized = F.interpolate(
                                    mask.float(), size=(img_h, img_w), mode="bilinear", align_corners=False
                                )
                                flat_masks[adapter_name] = resized.flatten(1)  # (1, N_img)

                        # Phase B: auto-derived masks (don't override manual)
                        for adapter_name, flat_mask in auto_derived_masks.items():
                            if adapter_name not in flat_masks:
                                trigger_words = auto_mask_loras.get(adapter_name, "")
                                positions = find_trigger_word_positions(
                                    tokenizer, prompt_text, trigger_words
                                )
                                if positions:
                                    token_pos_maps[adapter_name] = positions
                                    flat_masks[adapter_name] = flat_mask

                        if token_pos_maps and flat_masks:
                            attention_bias = construct_attention_bias(
                                lora_masks=flat_masks,
                                token_pos_maps=token_pos_maps,
                                img_seq_len=img_seq_len,
                                txt_seq_len=txt_seq_len,
                                device=self.latents.device,
                                dtype=self.latents.dtype,
                            )
                            freefuse_patched = patch_attention_with_bias(
                                transformer_raw, attention_bias
                            )
                            logger.info(
                                f"[DenoiseNode] FreeFuse attention bias applied: "
                                f"img_seq={img_seq_len}, txt_seq={txt_seq_len}, "
                                f"loras={list(token_pos_maps.keys())}"
                            )
                except Exception:
                    logger.exception("[DenoiseNode] Failed to build FreeFuse attention bias")
                    freefuse_patched = None

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
            controlnet = mm.resolve(controlnet_input, device=self.device, dtype=transformer_dtype)
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

        # Compute mu for dynamic shifting (resolution-dependent sigma schedule).
        # When use_dynamic_shifting=False the scheduler ignores mu.
        retrieve_kwargs = {}
        if getattr(self.scheduler.config, "use_dynamic_shifting", False):
            image_seq_len = (self.latents.shape[2] // 2) * (self.latents.shape[3] // 2)
            retrieve_kwargs["mu"] = self._calculate_shift(image_seq_len)

        timesteps, num_inference_steps = self.retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            self.device,
            sigmas=self.sigmas,
            **retrieve_kwargs,
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
            # Explicitly check for None to avoid numpy array truthiness issues
            source_mask = getattr(self, "source_mask", None)
            if source_mask is None:
                source_mask = getattr(self, "image_mask", None)
            if source_mask is not None:
                # Convert mask to torch tensor with batch dimension
                # source_mask shape: (H, W, C) -> (1, H, W, C)
                original_mask = numpy_to_pt(source_mask[None, ...]).to(device=self.device, dtype=self.dtype)
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

                    def _run_controlnet_guess_mode():
                        with mm.default_device_scope(control_image_latents_typed.device):
                            mm.ensure_module_device(
                                controlnet,
                                device=control_image_latents_typed.device,
                                dtype=transformer_dtype,
                            )
                            return controlnet(
                                pos_latent_model_input_list,
                                pos_timestep_model_input,
                                pos_prompt_embeds_model_input,
                                control_image_latents_typed,
                                conditioning_scale=cond_scale,
                            )

                    pos_controlnet_block_samples = self._run_with_oom_retry(
                        mm,
                        _run_controlnet_guess_mode,
                        preserve=("controlnet", "transformer"),
                        description="ControlNet (guess mode)",
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

                    def _run_controlnet_balanced():
                        with mm.default_device_scope(control_context.device):
                            mm.ensure_module_device(
                                controlnet,
                                device=control_context.device,
                                dtype=transformer_dtype,
                            )
                            return controlnet(
                                latent_model_input_list,
                                timestep_model_input,
                                prompt_embeds_model_input,
                                control_context,
                                conditioning_scale=cond_scale,
                            )

                    controlnet_block_samples = self._run_with_oom_retry(
                        mm,
                        _run_controlnet_balanced,
                        preserve=("controlnet", "transformer"),
                        description="ControlNet (balanced/prompt mode)",
                    )

                    if use_prompt_mode and controlnet_block_samples is not None:
                        controlnet_block_samples = self._apply_prompt_mode_decay(
                            controlnet_block_samples, prompt_mode_decay
                        )

            # Apply spatial mask to ControlNet block samples if provided
            spatial_mask_input = getattr(self, "controlnet_spatial_mask", None)

            if controlnet_block_samples is not None and spatial_mask_input is not None:
                controlnet_block_samples = self._apply_spatial_mask(
                    controlnet_block_samples, spatial_mask_input, latents.shape
                )

            def _run_transformer():
                return transformer(
                    latent_model_input_list,
                    timestep_model_input,
                    prompt_embeds_model_input,
                    controlnet_block_samples=controlnet_block_samples,
                    return_dict=False,
                )[0]

            # Determine which models to preserve during OOM recovery.
            # Always preserve transformer; also preserve controlnet if we have block samples.
            preserve_for_transformer = ("transformer",)
            if controlnet_block_samples is not None:
                preserve_for_transformer = ("transformer", "controlnet")

            model_out_list = self._run_with_oom_retry(
                mm,
                _run_transformer,
                preserve=preserve_for_transformer,
                description="transformer",
            )

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

        # Clean up FreeFuse attention bias patches
        if freefuse_patched:
            from iartisanz.modules.generation.graph.nodes.freefuse_attention_bias import (
                unpatch_attention_bias,
            )

            unpatch_attention_bias(freefuse_patched)
            logger.debug("[DenoiseNode] Cleaned up FreeFuse attention bias patches")

        # Clean up LoRA spatial mask patches
        if lora_masks_applied:
            from iartisanz.modules.generation.graph.nodes.lora_spatial_mask import (
                unpatch_all_lora_layers,
            )

            unpatch_count = unpatch_all_lora_layers()
            if unpatch_count > 0:
                logger.debug(f"[DenoiseNode] Cleaned up {unpatch_count} LoRA spatial mask patches")

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

    @staticmethod
    def _calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b
