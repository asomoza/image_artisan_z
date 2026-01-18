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

        class _Cfg:
            control_in_dim = 16

        self.config = _Cfg()

    def forward(
        self,
        _latent_model_input_list,
        _timestep_model_input,
        _prompt_embeds_model_input,
        _control_context,
        *,
        conditioning_scale: float = 1.0,
        **_kwargs,
    ):
        self.called = True
        self.last_scale = float(conditioning_scale)
        return {0: torch.tensor(1.0)}


def _make_denoise_node(*, device: torch.device) -> DenoiseNode:
    node = DenoiseNode()
    node.device = device
    node.dtype = torch.float32

    node.transformer = ModelHandle("transformer")
    node.scheduler = DummyScheduler()
    node.num_inference_steps = 1

    # Keep it tiny.
    node.latents = torch.randn(1, 16, 4, 4)

    # Prompt embeddings can be any tensor; dummy transformer ignores them.
    node.prompt_embeds = torch.randn(5, 8)
    node.negative_prompt_embeds = torch.randn(5, 8)

    # Enable CFG so we exercise the pos/neg batch path.
    node.guidance_scale = 2.0

    return node


def test_denoise_node_skips_controlnet_when_not_connected():
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    controlnet = DummyControlNet()

    mm.register_active_model(model_id="test", transformer=transformer, controlnet=controlnet)

    node = _make_denoise_node(device=torch.device("cpu"))

    with mm.device_scope(device="cpu", dtype=torch.float32):
        node()

    assert controlnet.called is False
    assert transformer.saw_controlnet is False


def test_denoise_node_uses_controlnet_when_connected():
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    controlnet = DummyControlNet()

    mm.register_active_model(model_id="test", transformer=transformer, controlnet=controlnet)

    node = _make_denoise_node(device=torch.device("cpu"))
    node.controlnet = ModelHandle("controlnet")
    node.control_image_latents = torch.zeros(1, 16, 1, 4, 4)
    node.controlnet_conditioning_scale = 0.7

    with mm.device_scope(device="cpu", dtype=torch.float32):
        node()

    assert controlnet.called is True
    assert controlnet.last_scale == 0.7
    assert transformer.saw_controlnet is True
    assert transformer.last_controlnet_block_samples == {0: torch.tensor(1.0)}
