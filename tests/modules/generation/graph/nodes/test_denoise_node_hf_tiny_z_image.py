import os
import time
from pathlib import Path

import pytest


# Ensure deterministic cuBLAS behavior when CUDA is used.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch


REPO_ID = "tiny-random/z-image"
PROMPT = "a small red cube on a white table"
NEG_PROMPT = ""
SEED = 42
SIZE = 32
STEPS = 4
GUIDANCE_SCALE = 4.0


# exact float32 bits for out[0,0,0,:8] when running on CPU fp32..
CPU_SLICE_BITS_I32 = [
    1063861510,
    1066184247,
    1062861841,
    -1075318306,
    1063689315,
    -1075642206,
    -1105516285,
    -1071868637,
]

# exact float32 bits for out[0,0,0,:8] when running on CUDA fp32
# Note: change when either GPU, driver, CUDA or PyTorch version changes.
CUDA_SLICE_BITS_I32 = [
    1050383174,
    1081218628,
    -1093935312,
    1070311151,
    -1080280558,
    1059830088,
    -1080299736,
    -1091929961,
]


def _golden_slice_f32() -> torch.Tensor:
    return torch.tensor(CPU_SLICE_BITS_I32, dtype=torch.int32).view(torch.float32)


def _golden_cuda_slice_f32() -> torch.Tensor:
    return torch.tensor(CUDA_SLICE_BITS_I32, dtype=torch.int32).view(torch.float32)


def _slice_f32(latents: torch.Tensor) -> torch.Tensor:
    return latents[0, 0, 0, :8].contiguous().to(torch.float32)


def _slice_bits_i32(latents: torch.Tensor) -> list[int]:
    return _slice_f32(latents).view(torch.int32).tolist()


def _run_tiny_random_z_image(
    *,
    device: torch.device,
    dtype: torch.dtype,
    use_torch_compile: bool = False,
    clear_mm: bool = True,
) -> torch.Tensor:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    snapshot_download = huggingface_hub.snapshot_download

    model_path = Path(snapshot_download(repo_id=REPO_ID))
    for sub in ("tokenizer", "text_encoder", "transformer", "vae"):
        assert (model_path / sub).is_dir(), f"Missing expected folder: {sub}"

    from iartisanz.app.model_manager import get_model_manager
    from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
    from iartisanz.modules.generation.graph.nodes.boolean_node import BooleanNode
    from iartisanz.modules.generation.graph.nodes.denoise_node import DenoiseNode
    from iartisanz.modules.generation.graph.nodes.latents_node import LatentsNode
    from iartisanz.modules.generation.graph.nodes.number_node import NumberNode
    from iartisanz.modules.generation.graph.nodes.prompt_encode_node import PromptEncoderNode
    from iartisanz.modules.generation.graph.nodes.scheduler_node import SchedulerNode
    from iartisanz.modules.generation.graph.nodes.zimage_model_node import ZImageModelNode

    mm = get_model_manager()
    if clear_mm:
        mm.clear()

    # Keep text encoder on CPU for the CUDA test too, so prompt embeddings match CPU.
    if device.type == "cuda":
        mm.set_default_device("text_encoder", "cpu")

    model_node = ZImageModelNode(
        path=str(model_path),
        model_name=REPO_ID,
        version="hf-test",
        model_type="Z-Image Turbo",
    )

    prompt_node = PromptEncoderNode()
    prompt_node.positive_prompt = PROMPT
    prompt_node.negative_prompt = NEG_PROMPT

    latents_node = LatentsNode()
    latents_node.width = SIZE
    latents_node.height = SIZE
    latents_node.seed = SEED

    # set scheduler config explicitly
    sdo = SchedulerDataObject(
        name="Euler",
        scheduler_index=0,
        scheduler_class="FlowMatchEulerDiscreteScheduler",
        num_train_timesteps=1000,
        shift=3.0,
        use_dynamic_shifting=False,
        base_shift=0.5,
        max_shift=1.15,
        base_image_seq_len=256,
        max_image_seq_len=4096,
        invert_sigmas=False,
        shift_terminal=None,
        use_karras_sigmas=False,
        use_exponential_sigmas=False,
        use_beta_sigmas=False,
        time_shift_type="exponential",
        stochastic_sampling=False,
    )
    scheduler_node = SchedulerNode(scheduler_data_object=sdo)

    steps_node = NumberNode(number=STEPS)
    guidance_node = NumberNode(number=GUIDANCE_SCALE)
    compile_node = BooleanNode(value=bool(use_torch_compile))
    denoise_node = DenoiseNode()

    prompt_node.connect("tokenizer", model_node, "tokenizer")
    prompt_node.connect("text_encoder", model_node, "text_encoder")
    latents_node.connect("vae_scale_factor", model_node, "vae_scale_factor")
    latents_node.connect("num_channels_latents", model_node, "num_channels_latents")
    latents_node.connect("vae", model_node, "vae")
    denoise_node.connect("transformer", model_node, "transformer")
    denoise_node.connect("num_inference_steps", steps_node, "value")
    denoise_node.connect("latents", latents_node, "latents")
    denoise_node.connect("scheduler", scheduler_node, "scheduler")
    denoise_node.connect("prompt_embeds", prompt_node, "prompt_embeds")
    denoise_node.connect("negative_prompt_embeds", prompt_node, "negative_prompt_embeds")
    denoise_node.connect("guidance_scale", guidance_node, "value")
    denoise_node.connect("use_torch_compile", compile_node, "value")

    for node in (
        model_node,
        prompt_node,
        latents_node,
        scheduler_node,
        steps_node,
        guidance_node,
        compile_node,
        denoise_node,
    ):
        node.device = device
        node.dtype = dtype

    with mm.device_scope(device=device, dtype=dtype):
        model_node()

        mm.get_raw("text_encoder").eval()
        mm.get_raw("transformer").eval()
        mm.get_raw("vae").eval()

        prompt_node()
        latents_node()
        scheduler_node()
        steps_node()
        guidance_node()
        compile_node()
        denoise_node()

    out = denoise_node.values.get("latents")
    assert isinstance(out, torch.Tensor)
    assert out.device.type == "cpu", "DenoiseNode returns CPU latents for downstream nodes"
    assert out.ndim == 4
    return out


@pytest.mark.hf
@pytest.mark.slow
def test_denoise_tiny_random_z_image_cpu_golden_regression():
    """Deterministic CPU regression test (exact match)."""

    if os.environ.get("IARTISANZ_RUN_HF_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_HF_TESTS=1 to enable Hugging Face integration tests")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    out = _run_tiny_random_z_image(device=torch.device("cpu"), dtype=torch.float32)
    assert _slice_bits_i32(out) == CPU_SLICE_BITS_I32


@pytest.mark.hf
@pytest.mark.slow
def test_denoise_tiny_random_z_image_cpu_torch_compile_matches_eager():
    """Compiled vs eager should match on CPU.

    This specifically checks that enabling/disabling torch.compile does not change outputs.
    """

    if os.environ.get("IARTISANZ_RUN_HF_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_HF_TESTS=1 to enable Hugging Face integration tests")

    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile not available")

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)

    device = torch.device("cpu")
    dtype = torch.float32

    out_eager_1 = _run_tiny_random_z_image(device=device, dtype=dtype, use_torch_compile=False, clear_mm=True)
    out_compiled = _run_tiny_random_z_image(device=device, dtype=dtype, use_torch_compile=True, clear_mm=False)
    out_eager_2 = _run_tiny_random_z_image(device=device, dtype=dtype, use_torch_compile=False, clear_mm=False)

    # Eager should be exactly reproducible.
    torch.testing.assert_close(_slice_f32(out_eager_2), _slice_f32(out_eager_1), rtol=0.0, atol=0.0)

    # torch.compile is allowed tiny fp drift depending on backend / compiler decisions.
    atol = float(os.environ.get("IARTISANZ_HF_CPU_COMPILE_ATOL", "1e-5"))
    rtol = float(os.environ.get("IARTISANZ_HF_CPU_COMPILE_RTOL", "1e-4"))
    torch.testing.assert_close(_slice_f32(out_compiled), _slice_f32(out_eager_1), rtol=rtol, atol=atol)


@pytest.mark.hf
@pytest.mark.slow
def test_denoise_tiny_random_z_image_cuda_close_to_cpu():
    """CUDA regression test.

    Assumes the same GPU + software environment; compares against a stored CUDA test slice.
    configure tolerance as needed
    """

    if os.environ.get("IARTISANZ_RUN_HF_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_HF_TESTS=1 to enable Hugging Face integration tests")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Use fp32 on GPU to reduce numeric drift vs CPU.
    device = torch.device("cuda")
    dtype = torch.float32

    atol = float(os.environ.get("IARTISANZ_HF_CUDA_ATOL", "1e-4"))
    rtol = float(os.environ.get("IARTISANZ_HF_CUDA_RTOL", "1e-4"))

    # Disable flash/mem-efficient SDPA backends to improve determinism across GPUs.
    # Prefer math SDP.
    sdpa_ctx = torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)

    expected = _golden_cuda_slice_f32()

    with sdpa_ctx:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = _run_tiny_random_z_image(device=device, dtype=dtype)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0

    got = _slice_f32(out)
    torch.testing.assert_close(got, expected, rtol=rtol, atol=atol)

    print(f"HF tiny model denoise (CUDA): dtype={dtype} time={dt:.3f}s atol={atol} rtol={rtol}")
