from diffusers import (
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    FlowMatchEulerDiscreteScheduler,
    FlowMatchHeunDiscreteScheduler,
    FlowMatchLCMScheduler,
    SASolverScheduler,
    UniPCMultistepScheduler,
)


# Factors taken from ComfyUI:
# https://github.com/comfyanonymous/ComfyUI/blob/092ee8a5008c8d558b0a72cc7961a31d9cc5400b/comfy/latent_formats.py#L155
ZIMAGE_LATENT_RGB_FACTORS = [
    [-0.0346, 0.0244, 0.0681],
    [0.0034, 0.0210, 0.0687],
    [0.0275, -0.0668, -0.0433],
    [-0.0174, 0.0160, 0.0617],
    [0.0859, 0.0721, 0.0329],
    [0.0004, 0.0383, 0.0115],
    [0.0405, 0.0861, 0.0915],
    [-0.0236, -0.0185, -0.0259],
    [-0.0245, 0.0250, 0.1180],
    [0.1008, 0.0755, -0.0421],
    [-0.0515, 0.0201, 0.0011],
    [0.0428, -0.0012, -0.0036],
    [0.0817, 0.0765, 0.0749],
    [-0.1264, -0.0522, -0.1103],
    [-0.0280, -0.0881, -0.0499],
    [-0.1262, -0.0982, -0.0778],
]

FLUX2_LATENT_RGB_FACTORS = [
    [0.0058, 0.0113, 0.0073],
    [0.0495, 0.0443, 0.0836],
    [-0.0099, 0.0096, 0.0644],
    [0.2144, 0.3009, 0.3652],
    [0.0166, -0.0039, -0.0054],
    [0.0157, 0.0103, -0.0160],
    [-0.0398, 0.0902, -0.0235],
    [-0.0052, 0.0095, 0.0109],
    [-0.3527, -0.2712, -0.1666],
    [-0.0301, -0.0356, -0.0180],
    [-0.0107, 0.0078, 0.0013],
    [0.0746, 0.0090, -0.0941],
    [0.0156, 0.0169, 0.0070],
    [-0.0034, -0.0040, -0.0114],
    [0.0032, 0.0181, 0.0080],
    [-0.0939, -0.0008, 0.0186],
    [0.0018, 0.0043, 0.0104],
    [0.0284, 0.0056, -0.0127],
    [-0.0024, -0.0022, -0.0030],
    [0.1207, -0.0026, 0.0065],
    [0.0128, 0.0101, 0.0142],
    [0.0137, -0.0072, -0.0007],
    [0.0095, 0.0092, -0.0059],
    [0.0000, -0.0077, -0.0049],
    [-0.0465, -0.0204, -0.0312],
    [0.0095, 0.0012, -0.0066],
    [0.0290, -0.0034, 0.0025],
    [0.0220, 0.0169, -0.0048],
    [-0.0332, -0.0457, -0.0468],
    [-0.0085, 0.0389, 0.0609],
    [-0.0076, 0.0003, -0.0043],
    [-0.0111, -0.0460, -0.0614],
]

MODEL_TYPES = {
    1: "Z-Image Turbo",
    2: "Z-Image",
    3: "Flux.2 Klein 9B",
    5: "Flux.2 Klein 4B",
    7: "Flux.2 Dev",
}

# Model family groupings — used to select the correct graph / pipeline.
ZIMAGE_MODEL_TYPES = {1, 2}
FLUX2_MODEL_TYPES = {3, 5, 7}
FLUX2_KLEIN_MODEL_TYPES = {3, 5}
FLUX2_DEV_MODEL_TYPES = {7}

# (num_double_blocks, num_single_blocks) for Flux2 models
FLUX2_LAYER_COUNTS = {
    3: (8, 24),   # Klein 9B
    5: (5, 20),   # Klein 4B
    7: (8, 48),   # Dev
}


def get_default_granular_weights(model_type: int) -> dict:
    """Return default per-layer granular weights dict for the given model type."""
    if model_type in ZIMAGE_MODEL_TYPES:
        return {f"layers.{i}": 1.0 for i in range(30)}
    if model_type in FLUX2_MODEL_TYPES:
        n_double, n_single = FLUX2_LAYER_COUNTS[model_type]
        weights = {f"transformer_blocks.{i}": 1.0 for i in range(n_double)}
        weights.update({f"single_transformer_blocks.{i}": 1.0 for i in range(n_single)})
        return weights
    return {}


# Default generation parameters per model type.
# Klein types (3, 5) have different defaults for distilled vs base variants.
MODEL_TYPE_DEFAULTS: dict[int, dict[str, int | float]] = {
    1: {"num_inference_steps": 9, "guidance_scale": 1.0},
    2: {"num_inference_steps": 50, "guidance_scale": 5.0},
    3: {"num_inference_steps": 4, "guidance_scale": 1.0},
    5: {"num_inference_steps": 4, "guidance_scale": 1.0},
    7: {"num_inference_steps": 50, "guidance_scale": 4.0},
}

# Base (non-distilled) Klein defaults — used when distilled=False.
_KLEIN_BASE_DEFAULTS: dict[str, int | float] = {"num_inference_steps": 30, "guidance_scale": 4.0}


def get_model_type_defaults(model_type: int, distilled: bool = True) -> dict[str, int | float]:
    """Return default generation parameters for a model type and variant."""
    if model_type in FLUX2_KLEIN_MODEL_TYPES and not distilled:
        return dict(_KLEIN_BASE_DEFAULTS)
    return dict(MODEL_TYPE_DEFAULTS.get(model_type, {}))

SCHEDULER_CLASS_MAPPING = {
    "DEISMultistepScheduler": DEISMultistepScheduler,
    "DPMSolverMultistepScheduler": DPMSolverMultistepScheduler,
    "DPMSolverSinglestepScheduler": DPMSolverSinglestepScheduler,
    "FlowMatchEulerDiscreteScheduler": FlowMatchEulerDiscreteScheduler,
    "FlowMatchHeunDiscreteScheduler": FlowMatchHeunDiscreteScheduler,
    "FlowMatchLCMScheduler": FlowMatchLCMScheduler,
    "SASolverScheduler": SASolverScheduler,
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
}

SCHEDULER_NAME_CLASS_MAPPING = {
    "DEIS": "DEISMultistepScheduler",
    "DPM++ 2M": "DPMSolverMultistepScheduler",
    "DPM++ 2S": "DPMSolverSinglestepScheduler",
    "Euler": "FlowMatchEulerDiscreteScheduler",
    "Heun": "FlowMatchHeunDiscreteScheduler",
    "LCM": "FlowMatchLCMScheduler",
    "SA": "SASolverScheduler",
    "UniPC": "UniPCMultistepScheduler",
}

SCHEDULER_NAMES = list(SCHEDULER_NAME_CLASS_MAPPING.keys())

OFFLOAD_STRATEGIES: dict[str, str] = {
    "auto": "Auto",
    "model_offload": "Model Offload",
    "group_offload": "Group Offload",
    "sequential_group_offload": "Sequential Group Offload",
}
