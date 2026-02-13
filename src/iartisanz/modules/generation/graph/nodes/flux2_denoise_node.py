import inspect
import logging
from typing import Optional, Union

import numpy as np
import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.iartisanz_node_error import IArtisanZNodeError
from iartisanz.modules.generation.graph.nodes.node import Node


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
        "lora",
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

        num_inference_steps = int(self.num_inference_steps)

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

        order = getattr(self.scheduler, "order", 1)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * order, 0)

        # Mark step boundaries for CUDA graphs
        if use_torch_compile and self.device.type == "cuda" and hasattr(torch, "compiler"):
            mark = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
        else:
            mark = None

        self.scheduler.set_begin_index(0)

        for i, t in enumerate(timesteps):
            if self.abort:
                return

            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            latent_model_input = latents.to(transformer_dtype)

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
                            img_ids=latent_ids,
                            return_dict=False,
                        )[0]
                return transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=None,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
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

            # Classifier-free guidance
            if do_cfg and negative_prompt_embeds is not None:
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
                                img_ids=latent_ids,
                                return_dict=False,
                            )[0]
                    return transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=None,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]

                neg_noise_pred = self._run_with_oom_retry(
                    mm,
                    _run_uncond,
                    preserve=("transformer",),
                    description="transformer (uncond)",
                )
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Step callback
            step_idx = i // order
            is_last = i == len(timesteps) - 1
            is_step_boundary = (i + 1) > num_warmup_steps and (i + 1) % order == 0
            if (is_last or is_step_boundary) and self.callback is not None:
                self.callback(step_idx, t, latents)

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
