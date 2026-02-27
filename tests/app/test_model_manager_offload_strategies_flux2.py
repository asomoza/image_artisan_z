"""Tests for the three manual offload strategies (model_offload, group_offload,
sequential_group_offload) using the real Flux.2 Klein 4B model loaded from the
database — exercising the full UI/graph workflow including component registry
path resolution.

Each test runs the full Flux2 graph (model node → prompt encoder → latents →
scheduler → denoise → latents decoder) under a specific strategy and verifies
that:

1. The strategy is correctly applied by ``ModelManager``.
2. Components are transferred to GPU when needed during execution.
3. Components are back on CPU after the pipeline completes (for model_offload
   and sequential_group_offload).
4. The pipeline produces a valid output image.

A second set of tests repeats the above with a fake LoRA adapter loaded via
``LoraNode`` to verify that PEFT adapters and offload strategies coexist.
"""

import os
import tempfile

import pytest

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from safetensors.torch import save_file

from iartisanz.app.directories import DirectoriesObject


# ---------------------------------------------------------------------------
# Constants — match the real DB row for FLUX.2-klein-4B (id=10, model_type=5)
# ---------------------------------------------------------------------------

DB_MODEL_ID = 10
MODEL_NAME = "FLUX.2-klein-4B"
MODEL_VERSION = "1.0"
MODEL_TYPE = 5  # Flux.2 Klein 4B distilled
MODEL_PATH = "/home/ozzy/ImageArtisanZ/models/diffusers/FLUX.2-klein-4B"
DATA_PATH = "/home/ozzy/ImageArtisanZ/data"
DB_PATH = os.path.join(DATA_PATH, "app.db")
MODELS_DIFFUSERS_PATH = "/home/ozzy/ImageArtisanZ/models/diffusers"

PROMPT = "a small red cube on a white table"
NEG_PROMPT = ""
SEED = 42
SIZE = 64  # smallest viable for Flux2 (patchify needs 2x2 spatial)
STEPS = 2
GUIDANCE_SCALE = 1.0  # distilled model, no CFG

DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16

# Flux2 Klein 4B single_transformer_blocks.0.attn.to_qkv_mlp_proj dimensions
_LORA_RANK = 4
_LORA_IN = 3072
_LORA_OUT = 27648
_LORA_ADAPTER_NAME = "test_fake_lora"


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


def _check_real_model_available():
    """Skip if the real model, database, or CUDA GPU is not available."""
    if os.environ.get("IARTISANZ_RUN_REAL_MODEL_TESTS") != "1":
        pytest.skip("Set IARTISANZ_RUN_REAL_MODEL_TESTS=1 to enable real model tests")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for offload strategy tests")
    if not os.path.isfile(DB_PATH):
        pytest.skip(f"Database not found at {DB_PATH}")
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"Model directory not found at {MODEL_PATH}")


def _setup_app_singletons():
    """Set up the app singletons to point to the real user data."""
    import iartisanz.app.app as app_mod

    app_mod.set_app_database_path(DB_PATH)
    app_mod.set_app_directories(
        DirectoriesObject(
            data_path=DATA_PATH,
            models_diffusers=MODELS_DIFFUSERS_PATH,
            models_loras="",
            models_controlnets="",
            outputs_images="",
            outputs_source_images="",
            outputs_source_masks="",
            outputs_controlnet_source_images="",
            outputs_conditioning_images="",
            outputs_edit_source_images="",
            outputs_edit_images="",
            temp_path="",
        )
    )


def _create_fake_lora(path: str) -> str:
    """Create a minimal fake LoRA .safetensors targeting one Flux2 Klein 4B layer.

    Uses zero-initialized lora_B so the adapter has no effect on the output,
    but PEFT still injects it and ``set_adapters`` activates it.
    """
    state_dict = {
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_A.weight": torch.randn(
            _LORA_RANK, _LORA_IN, dtype=torch.bfloat16
        ),
        "transformer.single_transformer_blocks.0.attn.to_qkv_mlp_proj.lora_B.weight": torch.zeros(
            _LORA_OUT, _LORA_RANK, dtype=torch.bfloat16
        ),
    }
    save_file(state_dict, path)
    return path


@pytest.fixture()
def fake_lora_path(tmp_path):
    """Create a temporary fake LoRA file and return its path."""
    return _create_fake_lora(str(tmp_path / "fake_lora.safetensors"))


def _build_flux2_graph(device: torch.device, dtype: torch.dtype, lora_path: str | None = None):
    """Build the full Flux2 generation graph and return all nodes in execution order.

    If ``lora_path`` is provided, a ``LoraNode`` is inserted between the model
    node and the denoise node — exactly as the UI does via ``add_lora()``.
    """
    from iartisanz.modules.generation.data_objects.scheduler_data_object import SchedulerDataObject
    from iartisanz.modules.generation.graph.nodes.flux2_denoise_node import Flux2DenoiseNode
    from iartisanz.modules.generation.graph.nodes.flux2_latents_decoder_node import Flux2LatentsDecoderNode
    from iartisanz.modules.generation.graph.nodes.flux2_latents_node import Flux2LatentsNode
    from iartisanz.modules.generation.graph.nodes.flux2_model_node import Flux2ModelNode
    from iartisanz.modules.generation.graph.nodes.flux2_prompt_encode_node import Flux2PromptEncoderNode
    from iartisanz.modules.generation.graph.nodes.number_node import NumberNode
    from iartisanz.modules.generation.graph.nodes.number_range_node import NumberRangeNode
    from iartisanz.modules.generation.graph.nodes.scheduler_node import SchedulerNode
    from iartisanz.modules.generation.graph.nodes.text_node import TextNode

    # Model node — initialized the same way as the real UI workflow
    model_node = Flux2ModelNode(
        path=MODEL_PATH,
        model_name=MODEL_NAME,
        version=MODEL_VERSION,
        model_type=MODEL_TYPE,
        db_model_id=DB_MODEL_ID,
    )

    positive_prompt = TextNode(text=PROMPT)
    negative_prompt = TextNode(text=NEG_PROMPT)

    prompt_node = Flux2PromptEncoderNode()
    prompt_node.connect("tokenizer", model_node, "tokenizer")
    prompt_node.connect("text_encoder", model_node, "text_encoder")
    prompt_node.connect("positive_prompt", positive_prompt, "value")
    prompt_node.connect("negative_prompt", negative_prompt, "value")

    seed_node = NumberNode(number=SEED)
    width_node = NumberNode(number=SIZE)
    height_node = NumberNode(number=SIZE)

    latents_node = Flux2LatentsNode()
    latents_node.connect("seed", seed_node, "value")
    latents_node.connect("num_channels_latents", model_node, "num_channels_latents")
    latents_node.connect("width", width_node, "value")
    latents_node.connect("height", height_node, "value")
    latents_node.connect("vae_scale_factor", model_node, "vae_scale_factor")

    sdo = SchedulerDataObject()  # defaults: Euler, FlowMatchEulerDiscreteScheduler
    scheduler_node = SchedulerNode(scheduler_data_object=sdo)

    steps_node = NumberNode(number=STEPS)
    guidance_node = NumberNode(number=GUIDANCE_SCALE)
    guidance_start_end_node = NumberRangeNode(value=[0.0, 1.0])

    denoise_node = Flux2DenoiseNode()
    denoise_node.connect("transformer", model_node, "transformer")
    denoise_node.connect("num_inference_steps", steps_node, "value")
    denoise_node.connect("latents", latents_node, "latents")
    denoise_node.connect("latent_ids", latents_node, "latent_ids")
    denoise_node.connect("scheduler", scheduler_node, "scheduler")
    denoise_node.connect("prompt_embeds", prompt_node, "prompt_embeds")
    denoise_node.connect("text_ids", prompt_node, "text_ids")
    denoise_node.connect("negative_prompt_embeds", prompt_node, "negative_prompt_embeds")
    denoise_node.connect("negative_text_ids", prompt_node, "negative_text_ids")
    denoise_node.connect("guidance_scale", guidance_node, "value")
    denoise_node.connect("guidance_start_end", guidance_start_end_node, "value")
    denoise_node.connect("positive_prompt_text", positive_prompt, "value")

    # Optionally add a LoRA node (same wiring as generation_thread.add_lora)
    lora_node = None
    if lora_path is not None:
        from iartisanz.modules.generation.graph.nodes.lora_node import LoraNode

        lora_node = LoraNode(
            path=lora_path,
            adapter_name=_LORA_ADAPTER_NAME,
            lora_name="test_fake",
            version="1.0",
            transformer_weight=1.0,
        )
        lora_node.connect("transformer", model_node, "transformer")
        denoise_node.connect("lora", lora_node, "lora")

    decoder_node = Flux2LatentsDecoderNode()
    decoder_node.connect("vae", model_node, "vae")
    decoder_node.connect("latents", denoise_node, "latents")
    decoder_node.connect("latent_ids", denoise_node, "latent_ids")

    # Execution order matches the graph dependencies
    nodes = [
        seed_node,
        width_node,
        height_node,
        positive_prompt,
        negative_prompt,
        model_node,
    ]
    if lora_node is not None:
        nodes.append(lora_node)
    nodes.extend([
        prompt_node,
        latents_node,
        scheduler_node,
        steps_node,
        guidance_node,
        guidance_start_end_node,
        denoise_node,
        decoder_node,
    ])
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


def _run_pipeline(strategy: str, lora_path: str | None = None):
    """Run the full Flux2 pipeline under the given offload strategy and return
    (mm, nodes, image) for assertions.
    """
    from iartisanz.app.model_manager import get_model_manager

    _check_real_model_available()
    _setup_app_singletons()

    mm = get_model_manager()
    mm.clear()
    mm.offload_strategy = strategy
    mm.group_offload_use_stream = False
    mm.group_offload_low_cpu_mem = False

    nodes = _build_flux2_graph(DEVICE, DTYPE, lora_path=lora_path)

    with mm.device_scope(device=DEVICE, dtype=DTYPE):
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

@pytest.mark.real_model
@pytest.mark.slow
def test_flux2_model_offload_strategy():
    """Flux2 Model offload: bulk CPU↔GPU per node, components on CPU after pipeline."""
    mm, nodes, image = _run_pipeline("model_offload")

    assert mm.applied_strategy == "model_offload"

    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    assert text_encoder is not None
    assert transformer is not None
    assert vae is not None

    # After pipeline completes, model_offload should have moved everything back to CPU.
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

@pytest.mark.real_model
@pytest.mark.slow
def test_flux2_group_offload_strategy():
    """Flux2 Group offload: hooks applied at load, persist through pipeline."""
    mm, nodes, image = _run_pipeline("group_offload")

    assert mm.applied_strategy == "group_offload"

    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    assert text_encoder is not None
    assert transformer is not None
    assert vae is not None

    # Group offload hooks should persist after the pipeline.
    assert _has_group_offload_hooks(text_encoder), "text_encoder should have group offload hooks"
    assert _has_group_offload_hooks(transformer), "transformer should have group offload hooks"
    assert _has_group_offload_hooks(vae), "vae should have group offload hooks"

    # Image should be valid
    assert image.shape[0] == SIZE and image.shape[1] == SIZE


# ---------------------------------------------------------------------------
# sequential_group_offload: hooks applied per node, removed after
# ---------------------------------------------------------------------------

@pytest.mark.real_model
@pytest.mark.slow
def test_flux2_sequential_group_offload_strategy():
    """Flux2 Sequential group offload: hooks applied/removed per node, CPU between nodes."""
    mm, nodes, image = _run_pipeline("sequential_group_offload")

    assert mm.applied_strategy == "sequential_group_offload"

    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    assert text_encoder is not None
    assert transformer is not None
    assert vae is not None

    # After pipeline, sequential hooks should have been removed and components on CPU.
    te_dev = _module_device(text_encoder)
    tr_dev = _module_device(transformer)
    vae_dev = _module_device(vae)

    assert te_dev is not None and te_dev.type == "cpu", f"text_encoder should be on CPU, got {te_dev}"
    assert tr_dev is not None and tr_dev.type == "cpu", f"transformer should be on CPU, got {tr_dev}"
    assert vae_dev is not None and vae_dev.type == "cpu", f"vae should be on CPU, got {vae_dev}"

    # Hooks should be cleaned up
    assert not _has_group_offload_hooks(text_encoder), "text_encoder hooks should be removed after pipeline"
    assert not _has_group_offload_hooks(transformer), "transformer hooks should be removed after pipeline"
    assert not _has_group_offload_hooks(vae), "vae hooks should be removed after pipeline"

    # Image should be valid
    assert image.shape[0] == SIZE and image.shape[1] == SIZE


# ---------------------------------------------------------------------------
# Strategy transition: switch from one strategy to another mid-session
# ---------------------------------------------------------------------------

@pytest.mark.real_model
@pytest.mark.slow
def test_flux2_strategy_transition_group_to_model_offload():
    """Switching from group_offload to model_offload should remove hooks."""
    # First run with group_offload
    mm, nodes, image1 = _run_pipeline("group_offload")
    assert mm.applied_strategy == "group_offload"

    transformer = mm.get_managed_component("transformer")
    assert _has_group_offload_hooks(transformer), "Should have hooks after group_offload"

    # Switch to model_offload
    mm.offload_strategy = "model_offload"
    mm.apply_offload_strategy(DEVICE)

    assert mm.applied_strategy == "model_offload"

    # Hooks should be cleaned up
    assert not _has_group_offload_hooks(transformer), "Hooks should be removed after switching to model_offload"

    # Components should be on CPU (prepare_strategy_transition moves to CPU)
    tr_dev = _module_device(transformer)
    assert tr_dev is not None and tr_dev.type == "cpu"


# ===========================================================================
# LoRA variants — same three strategies, with a fake PEFT adapter loaded
# ===========================================================================


def _assert_lora_loaded(mm):
    """Verify the fake PEFT adapter was injected into the transformer."""
    transformer = mm.get_managed_component("transformer")
    assert transformer is not None
    peft_config = getattr(transformer, "peft_config", {})
    assert _LORA_ADAPTER_NAME in peft_config, (
        f"PEFT adapter '{_LORA_ADAPTER_NAME}' not found in transformer.peft_config "
        f"(loaded: {list(peft_config.keys())})"
    )


@pytest.mark.real_model
@pytest.mark.slow
def test_flux2_model_offload_with_lora(fake_lora_path):
    """Model offload + LoRA: PEFT adapter loaded, components on CPU after pipeline."""
    mm, nodes, image = _run_pipeline("model_offload", lora_path=fake_lora_path)

    assert mm.applied_strategy == "model_offload"
    _assert_lora_loaded(mm)

    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    te_dev = _module_device(text_encoder)
    tr_dev = _module_device(transformer)
    vae_dev = _module_device(vae)

    assert te_dev is not None and te_dev.type == "cpu", f"text_encoder should be on CPU, got {te_dev}"
    assert tr_dev is not None and tr_dev.type == "cpu", f"transformer should be on CPU, got {tr_dev}"
    assert vae_dev is not None and vae_dev.type == "cpu", f"vae should be on CPU, got {vae_dev}"

    assert not _has_group_offload_hooks(transformer)
    assert image.shape[0] == SIZE and image.shape[1] == SIZE


@pytest.mark.real_model
@pytest.mark.slow
def test_flux2_group_offload_with_lora(fake_lora_path):
    """Group offload + LoRA: hooks persist, PEFT adapter loaded."""
    mm, nodes, image = _run_pipeline("group_offload", lora_path=fake_lora_path)

    assert mm.applied_strategy == "group_offload"
    _assert_lora_loaded(mm)

    transformer = mm.get_managed_component("transformer")
    assert _has_group_offload_hooks(transformer), "transformer should have group offload hooks"

    assert image.shape[0] == SIZE and image.shape[1] == SIZE


@pytest.mark.real_model
@pytest.mark.slow
def test_flux2_sequential_group_offload_with_lora(fake_lora_path):
    """Sequential group offload + LoRA: hooks removed after, components on CPU."""
    mm, nodes, image = _run_pipeline("sequential_group_offload", lora_path=fake_lora_path)

    assert mm.applied_strategy == "sequential_group_offload"
    _assert_lora_loaded(mm)

    text_encoder = mm.get_managed_component("text_encoder")
    transformer = mm.get_managed_component("transformer")
    vae = mm.get_managed_component("vae")

    te_dev = _module_device(text_encoder)
    tr_dev = _module_device(transformer)
    vae_dev = _module_device(vae)

    assert te_dev is not None and te_dev.type == "cpu", f"text_encoder should be on CPU, got {te_dev}"
    assert tr_dev is not None and tr_dev.type == "cpu", f"transformer should be on CPU, got {tr_dev}"
    assert vae_dev is not None and vae_dev.type == "cpu", f"vae should be on CPU, got {vae_dev}"

    assert not _has_group_offload_hooks(transformer), "hooks should be removed after pipeline"
    assert image.shape[0] == SIZE and image.shape[1] == SIZE
