import contextlib
import types

import torch


class FakeMM:
    def __init__(self):
        self.offloaded = []

    def resolve(self, value):
        return value

    def has(self, component: str) -> bool:
        return component == "transformer"

    def offload_to_cpu(self, component: str) -> None:
        self.offloaded.append(component)

    def is_cuda_oom(self, exc: BaseException) -> bool:
        return isinstance(exc, torch.cuda.OutOfMemoryError)

    def free_vram_for_forward_pass(self, *, preserve=()):
        for comp in ("preprocessor", "text_encoder", "controlnet", "vae", "transformer"):
            if comp not in preserve and self.has(comp):
                self.offload_to_cpu(comp)
        return 1

    @contextlib.contextmanager
    def use_components(self, *names, device=None, strategy_override=None):
        yield


class FakeVae:
    def __init__(self):
        self.dtype = torch.float32
        self.config = types.SimpleNamespace(scaling_factor=1.0, shift_factor=0.0, block_out_channels=[1, 2])
        self.calls = 0

    def decode(self, _latents, return_dict=False):
        self.calls += 1
        if self.calls == 1:
            # Simulate CUDA OOM even in CPU-only test env.
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        decoded = torch.zeros((1, 3, 2, 2), dtype=torch.float32)
        return (decoded,)


def test_latents_decoder_retries_once_on_cuda_oom(monkeypatch):
    from iartisanz.modules.generation.graph.nodes.zimage_latents_decoder_node import ZImageLatentsDecoderNode

    mm = FakeMM()

    # Patch the module-level get_model_manager to return our fake manager.
    import iartisanz.modules.generation.graph.nodes.zimage_latents_decoder_node as mod

    monkeypatch.setattr(mod, "get_model_manager", lambda: mm)

    node = ZImageLatentsDecoderNode()
    node.device = torch.device("cpu")
    node.vae = FakeVae()
    node.latents = torch.zeros((1, 4, 2, 2), dtype=torch.float32)

    out = node()

    assert out["image"].shape == (2, 2, 3)
    assert mm.offloaded == ["transformer"]
    assert node.vae.calls == 2
