import pytest
import torch

from iartisanz.app.model_manager import ModelHandle, get_model_manager
from iartisanz.modules.generation.graph.nodes.denoise_node import DenoiseNode


class DummyScheduler:
    order = 1

    def set_timesteps(self, num_inference_steps, device=None, **_kwargs):
        self.timesteps = torch.tensor([999], device=device)

    def step(self, noise_pred, _t, latents, return_dict=False):
        return (latents + noise_pred.to(latents.dtype) * 0.0,)


class DummyTransformer(torch.nn.Module):
    def forward(
        self,
        latent_model_input_list,
        _timestep_model_input,
        _prompt_embeds_model_input,
        *,
        controlnet_block_samples=None,
        return_dict=False,
    ):
        outs = [torch.zeros_like(x) for x in latent_model_input_list]
        return (outs,)


class LazyControlNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class _Cfg:
            control_in_dim = 16

        self.config = _Cfg()

    def forward(
        self,
        _latent_model_input_list,
        _timestep_model_input,
        _prompt_embeds_model_input,
        control_context,
        *,
        conditioning_scale: float = 1.0,
        **_kwargs,
    ):
        if not hasattr(self, "control_all_x_embedder"):
            self.control_all_x_embedder = torch.nn.ModuleDict()
            in_features = int(control_context.reshape(control_context.shape[0], -1).shape[1])
            self.control_all_x_embedder["2-1"] = torch.nn.Linear(
                in_features,
                1,
                bias=False,
            )

        embedder = self.control_all_x_embedder["2-1"]
        flat = control_context.reshape(control_context.shape[0], -1)
        _out = embedder(flat)
        return {0: _out}


def _make_denoise_node(*, device: torch.device) -> DenoiseNode:
    node = DenoiseNode()
    node.device = device
    node.dtype = torch.float32

    node.transformer = ModelHandle("transformer")
    node.scheduler = DummyScheduler()
    node.num_inference_steps = 1

    node.latents = torch.randn(1, 16, 4, 4, device=device)
    node.prompt_embeds = torch.randn(1, 5, 8)
    node.negative_prompt_embeds = torch.randn(1, 5, 8)
    node.guidance_scale = 1.0

    return node


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for device placement regression")
def test_controlnet_lazy_submodule_is_on_cuda():
    mm = get_model_manager()
    mm.clear()

    transformer = DummyTransformer()
    controlnet = LazyControlNet()
    mm.register_active_model(model_id="test", transformer=transformer, controlnet=controlnet)

    node = _make_denoise_node(device=torch.device("cuda"))
    node.controlnet = ModelHandle("controlnet")
    node.control_image_latents = torch.zeros(1, 16, 1, 4, 4)
    node.controlnet_conditioning_scale = 1.0

    with mm.device_scope(device="cuda", dtype=torch.float32):
        node()

    key = next(iter(controlnet.control_all_x_embedder.keys()))
    weight_device = next(controlnet.control_all_x_embedder[key].parameters()).device
    assert weight_device.type == "cuda"
