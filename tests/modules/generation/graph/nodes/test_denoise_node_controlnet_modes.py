import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.denoise_node import DenoiseNode


class DummyScheduler:
    order = 1

    def set_timesteps(self, num_inference_steps, device=None, **_kwargs):
        # Use a single timestep tensor on the requested device.
        self.timesteps = torch.tensor([999], device=device)

    def step(self, noise_pred, _t, latents, return_dict=False):
        # Identity step (keeps latents stable).
        return (latents + noise_pred.to(latents.dtype) * 0.0,)


class DummyTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.saw_controlnet = False
        self.last_controlnet_block_samples = None

    def forward(
        self,
        latent_model_input_list,
        _timestep_model_input,
        _prompt_embeds_model_input,
        *,
        controlnet_block_samples=None,
        return_dict=False,
    ):
        self.saw_controlnet = controlnet_block_samples is not None
        self.last_controlnet_block_samples = controlnet_block_samples

        outs = [torch.zeros_like(x) for x in latent_model_input_list]
        return (outs,)


class DummyControlNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.called = False
        self.last_scale = None
        self.last_batch = None
        self.last_control_context_batch = None

        class _Cfg:
            control_in_dim = 16

        self.config = _Cfg()

    def forward(
        self,
        latent_model_input_list,
        _timestep_model_input,
        _prompt_embeds_model_input,
        control_context,
        *,
        conditioning_scale: float = 1.0,
        **_kwargs,
    ):
        self.called = True
        self.last_scale = float(conditioning_scale)
        self.last_batch = len(latent_model_input_list)
        self.last_control_context_batch = int(control_context.shape[0])

        # Return a per-layer residual tensor with a batch dimension.
        hint = torch.ones(self.last_batch, 1, device=control_context.device, dtype=control_context.dtype)
        return {0: hint}


def _make_denoise_node(*, device: torch.device, guidance_scale: float = 2.0) -> DenoiseNode:
    node = DenoiseNode()
    node.device = device
    node.dtype = torch.float32

    node.transformer = ModelHandle("transformer")
    node.scheduler = DummyScheduler()
    node.num_inference_steps = 1

    node.latents = torch.randn(1, 16, 4, 4)

    # Prompt embeddings can be any tensor; dummy transformer/controlnet ignore them.
    node.prompt_embeds = torch.randn(1, 5, 8)
    node.negative_prompt_embeds = torch.randn(1, 5, 8)

    node.guidance_scale = float(guidance_scale)
    return node


def test_denoise_node_controlnet_schedule_skips_when_disabled_for_step():
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    controlnet = DummyControlNet()
    mm.register_active_model(model_id="test", transformer=transformer, controlnet=controlnet)

    node = _make_denoise_node(device=torch.device("cpu"))
    node.controlnet = ModelHandle("controlnet")
    node.control_image_latents = torch.zeros(1, 16, 1, 4, 4)
    node.controlnet_conditioning_scale = 0.7

    # With a single step, start=end=1.0 disables the only step.
    node.control_guidance_start_end = [1.0, 1.0]

    with mm.device_scope(device="cpu", dtype=torch.float32):
        node()

    assert controlnet.called is False
    assert transformer.saw_controlnet is False


def test_denoise_node_control_mode_controlnet_only_conditions_positive_branch():
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    controlnet = DummyControlNet()
    mm.register_active_model(model_id="test", transformer=transformer, controlnet=controlnet)

    node = _make_denoise_node(device=torch.device("cpu"), guidance_scale=2.0)
    node.controlnet = ModelHandle("controlnet")
    node.control_image_latents = torch.zeros(1, 16, 1, 4, 4)
    node.controlnet_conditioning_scale = 0.5
    node.control_mode = "controlnet"

    with mm.device_scope(device="cpu", dtype=torch.float32):
        node()

    assert controlnet.called is True
    assert controlnet.last_scale == 0.5
    assert controlnet.last_batch == 1
    assert controlnet.last_control_context_batch == 1

    assert transformer.saw_controlnet is True
    samples = transformer.last_controlnet_block_samples
    assert 0 in samples
    hint = samples[0]
    assert hint.shape[0] == 2
    assert float(hint[0].item()) == 1.0
    assert float(hint[1].item()) == 0.0


def test_denoise_node_control_mode_prompt_applies_decay():
    mm = get_model_manager()
    mm.clear()

    class ControlNetTwoLayers(DummyControlNet):
        def forward(
            self,
            latent_model_input_list,
            timestep_model_input,
            prompt_embeds_model_input,
            control_context,
            *,
            conditioning_scale: float = 1.0,
            **_kwargs,
        ):
            super().forward(
                latent_model_input_list,
                timestep_model_input,
                prompt_embeds_model_input,
                control_context,
                conditioning_scale=conditioning_scale,
            )
            batch = len(latent_model_input_list)
            t0 = torch.full((batch, 1), 2.0, device=control_context.device, dtype=control_context.dtype)
            t1 = torch.full((batch, 1), 2.0, device=control_context.device, dtype=control_context.dtype)
            return {0: t0, 1: t1}

    transformer = DummyTransformer()
    controlnet = ControlNetTwoLayers()
    mm.register_active_model(model_id="test", transformer=transformer, controlnet=controlnet)

    node = _make_denoise_node(device=torch.device("cpu"), guidance_scale=1.0)
    node.controlnet = ModelHandle("controlnet")
    node.control_image_latents = torch.zeros(1, 16, 1, 4, 4)
    node.controlnet_conditioning_scale = 1.0
    node.control_mode = "prompt"
    node.prompt_mode_decay = 0.5

    with mm.device_scope(device="cpu", dtype=torch.float32):
        node()

    samples = transformer.last_controlnet_block_samples
    assert float(samples[0][0].item()) == 2.0
    assert float(samples[1][0].item()) == 1.0
