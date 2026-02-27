"""Tests for the three manual offload strategies (model_offload, group_offload,
sequential_group_offload) using the tiny-random/z-image HF model.

Each test runs the full graph (model node → prompt encoder → latents → scheduler →
denoise → latents decoder) under a specific strategy and verifies that:

1. The strategy is correctly applied by ``ModelManager``.
2. Components are on the expected devices at the right moments.
3. The pipeline produces a valid output image.
"""

import os
from pathlib import Path

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


REPO_ID = "tiny-random/z-image"
PROMPT = "a small red cube on a white table"
NEG_PROMPT = ""
SEED = 42
SIZE = 32
STEPS = 2
GUIDANCE_SCALE = 4.0


def _module_device(module: torch.nn.Module) -> torch.device | None:
    """Return the device of the first parameter/buffer of a module."""
    try:
        for p in module.parameters(recurse=True):
            return p.device
    except Exception:
        pass
    return None


def _has_group_offload_hooks(module: torch.nn.Module) -> bool:
    """Check whether any submodule has diffusers group_offloading hooks."""
    for submod in module.modules():
        if hasattr(submod, "_diffusers_hook"):
            reg = submod._diffusers_hook
            hooks = getattr(reg, "hooks", None) or getattr(reg, "_hooks", None)
            if hooks and "group_offloading" in hooks:
                return True
    return False


def _download_tiny_model() -> Path:
    huggingface_hub = pytest.importorskip("huggingface_hub")
    model_path = Path(huggingface_hub.snapshot_download(repo_id=REPO_ID))
    for sub in ("tokenizer", "text_encoder", "transformer", "vae"):
        assert (model_path / sub).is_dir(), f"Missing expected folder: {sub}"
    return model_path


def _build_graph(model_path: Path, device: torch.device, dtype: torch.dtype):
    """Build the full generation graph and return all nodes in execution order."""
    from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
    from iartisanz.modules.generation.graph.nodes.zimage_denoise_node import ZImageDenoiseNode
    from iartisanz.modules.generation.graph.nodes.zimage_latents_node import ZImageLatentsNode
    from iartisanz.modules.generation.graph.nodes.number_node import NumberNode
    from iartisanz.modules.generation.graph.nodes.zimage_prompt_encode_node import ZImagePromptEncoderNode
    from iartisanz.modules.generation.graph.nodes.scheduler_node import SchedulerNode
    from iartisanz.modules.generation.graph.nodes.zimage_model_node import ZImageModelNode
    from iartisanz.modules.generation.graph.nodes.zimage_latents_decoder_node import ZImageLatentsDecoderNode

    model_node = ZImageModelNode(
        path=str(model_path),
        model_name=REPO_ID,
        version="hf-test",
        model_type="Z-Image Turbo",
    )

    prompt_node = ZImagePromptEncoderNode()
    prompt_node.positive_prompt = PROMPT
    prompt_node.negative_prompt = NEG_PROMPT

    latents_node = ZImageLatentsNode()
    latents_node.width = SIZE
    latents_node.height = SIZE
    latents_node.seed = SEED

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
    denoise_node = ZImageDenoiseNode()
    decoder_node = ZImageLatentsDecoderNode()

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
    decoder_node.connect("vae", model_node, "vae")
    decoder_node.connect("latents", denoise_node, "latents")

    nodes = [model_node, prompt_node, latents_node, scheduler_node, steps_node, guidance_node, denoise_node, decoder_node]
    for node in nodes:
        node.device = device
        node.dtype = dtype

    return nodes


@pytest.fixture(autouse=True)
def _clean_model_manager():
    """Ensure each test starts and ends with a clean ModelManager."""
    from iartisanz.app.model_manager import get_model_manager

    mm = get_model_manager()
    mm.clear()
    mm.offload_strategy = "auto"
    mm.group_offload_use_stream = False
    mm.group_offload_low_cpu_mem = False
    yield
    mm.clear()
    mm.offload_strategy = "auto"
    mm.group_offload_use_stream = False
    mm.group_offload_low_cpu_mem = False


def _run_pipeline(
    strategy: str,
    device: torch.device,
    dtype: torch.dtype,
):
    """Run the full pipeline under the given offload strategy and return
    (mm, nodes, image) for assertions.
    """
    from iartisanz.app.model_manager import get_model_manager

    model_path = _download_tiny_model()
    mm = get_model_manager()
    mm.clear()
    mm.offload_strategy = strategy
    mm.group_offload_use_stream = False
    mm.group_offload_low_cpu_mem = False

    nodes = _build_graph(model_path, device, dtype)

    with mm.device_scope(device=device, dtype=dtype):
        for node in nodes:
            node()

    image = nodes[-1].values.get("image")
    assert image is not None
    assert isinstance(image, np.ndarray)
    assert image.ndim == 3
    assert image.shape[2] == 3

    return mm, nodes, image


# ---------------------------------------------------------------------------
# model_offload: components should be on CPU between nodes
# ---------------------------------------------------------------------------

@pytest.mark.hf
@pytest.mark.slow
def test_model_offload_strategy():
    """Model offload: bulk CPU↔GPU per node, components on CPU after pipeline."""
    if os.environ.get("IARTISANZ_RUN_HF_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_HF_TESTS=1 to enable Hugging Face integration tests")

    device = torch.device("cpu")
    dtype = torch.float32

    mm, nodes, image = _run_pipeline("model_offload", device, dtype)

    # Strategy should have been resolved and applied
    assert mm.applied_strategy == "model_offload"

    # After the full pipeline completes, model_offload's use_components
    # context managers should have moved components back to CPU.
    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    assert text_encoder is not None
    assert transformer is not None
    assert vae is not None

    te_dev = _module_device(text_encoder)
    tr_dev = _module_device(transformer)
    vae_dev = _module_device(vae)

    assert te_dev is not None and te_dev.type == "cpu", f"text_encoder should be on CPU after pipeline, got {te_dev}"
    assert tr_dev is not None and tr_dev.type == "cpu", f"transformer should be on CPU after pipeline, got {tr_dev}"
    assert vae_dev is not None and vae_dev.type == "cpu", f"vae should be on CPU after pipeline, got {vae_dev}"

    # No diffusers hooks should be installed
    assert not _has_group_offload_hooks(text_encoder), "text_encoder should not have group offload hooks"
    assert not _has_group_offload_hooks(transformer), "transformer should not have group offload hooks"
    assert not _has_group_offload_hooks(vae), "vae should not have group offload hooks"

    # Image should be valid
    assert image.shape[0] == SIZE and image.shape[1] == SIZE


# ---------------------------------------------------------------------------
# group_offload: diffusers hooks applied at load time, persist through pipeline
# ---------------------------------------------------------------------------

@pytest.mark.hf
@pytest.mark.slow
def test_group_offload_strategy():
    """Group offload: hooks applied at load, persist through pipeline."""
    if os.environ.get("IARTISANZ_RUN_HF_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_HF_TESTS=1 to enable Hugging Face integration tests")

    device = torch.device("cpu")
    dtype = torch.float32

    mm, nodes, image = _run_pipeline("group_offload", device, dtype)

    assert mm.applied_strategy == "group_offload"

    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    assert text_encoder is not None
    assert transformer is not None
    assert vae is not None

    # Group offload hooks should be present on submodules (applied at load time,
    # persist through the pipeline).
    assert _has_group_offload_hooks(text_encoder), "text_encoder should have group offload hooks"
    assert _has_group_offload_hooks(transformer), "transformer should have group offload hooks"
    assert _has_group_offload_hooks(vae), "vae should have group offload hooks"

    # Image should be valid
    assert image.shape[0] == SIZE and image.shape[1] == SIZE


# ---------------------------------------------------------------------------
# sequential_group_offload: hooks applied per node, removed after
# ---------------------------------------------------------------------------

@pytest.mark.hf
@pytest.mark.slow
def test_sequential_group_offload_strategy():
    """Sequential group offload: hooks applied/removed per node, CPU between nodes."""
    if os.environ.get("IARTISANZ_RUN_HF_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_HF_TESTS=1 to enable Hugging Face integration tests")

    device = torch.device("cpu")
    dtype = torch.float32

    mm, nodes, image = _run_pipeline("sequential_group_offload", device, dtype)

    assert mm.applied_strategy == "sequential_group_offload"

    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    assert text_encoder is not None
    assert transformer is not None
    assert vae is not None

    # After pipeline completes, sequential hooks should have been removed
    # and components moved back to CPU.
    te_dev = _module_device(text_encoder)
    tr_dev = _module_device(transformer)
    vae_dev = _module_device(vae)

    assert te_dev is not None and te_dev.type == "cpu", f"text_encoder should be on CPU, got {te_dev}"
    assert tr_dev is not None and tr_dev.type == "cpu", f"transformer should be on CPU, got {tr_dev}"
    assert vae_dev is not None and vae_dev.type == "cpu", f"vae should be on CPU, got {vae_dev}"

    # Hooks should be cleaned up after the context managers exit
    assert not _has_group_offload_hooks(text_encoder), "text_encoder hooks should be removed after pipeline"
    assert not _has_group_offload_hooks(transformer), "transformer hooks should be removed after pipeline"
    assert not _has_group_offload_hooks(vae), "vae hooks should be removed after pipeline"

    # Image should be valid
    assert image.shape[0] == SIZE and image.shape[1] == SIZE


# ---------------------------------------------------------------------------
# Strategy transition: switch from one strategy to another mid-session
# ---------------------------------------------------------------------------

@pytest.mark.hf
@pytest.mark.slow
def test_strategy_transition_group_to_model_offload():
    """Switching from group_offload to model_offload should remove hooks."""
    if os.environ.get("IARTISANZ_RUN_HF_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_HF_TESTS=1 to enable Hugging Face integration tests")

    device = torch.device("cpu")
    dtype = torch.float32

    # First run with group_offload
    mm, nodes, image1 = _run_pipeline("group_offload", device, dtype)
    assert mm.applied_strategy == "group_offload"

    transformer = mm.get_managed_component("transformer")
    assert _has_group_offload_hooks(transformer), "Should have hooks after group_offload"

    # Switch to model_offload
    mm.offload_strategy = "model_offload"
    mm.apply_offload_strategy(device)

    assert mm.applied_strategy == "model_offload"

    # Hooks should be cleaned up
    assert not _has_group_offload_hooks(transformer), "Hooks should be removed after switching to model_offload"

    # Components should be on CPU (prepare_strategy_transition moves to CPU)
    tr_dev = _module_device(transformer)
    assert tr_dev is not None and tr_dev.type == "cpu"
