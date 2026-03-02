"""LoRA format conversion utilities.

Pure functions for converting LoRA state dicts between formats.
No runtime model dependencies — those stay in lora_node.py.
"""

import re

import torch


def _is_diffusers_format(state_dict: dict) -> bool:
    """Check if state dict is already in diffusers PEFT format.

    Diffusers PEFT keys start with ``transformer.`` and use ``lora_A``/``lora_B``
    naming.  Non-diffusers formats use other prefixes (``diffusion_model.``,
    ``lora_unet``), kohya naming (``lora_down``/``lora_up``), or ``.alpha`` keys.
    A file can have a ``transformer.`` prefix but still use kohya lora_down/lora_up
    naming — that is NOT diffusers format and needs conversion.
    """
    return (
        all(k.startswith("transformer.") for k in state_dict)
        and not any("lora_down" in k for k in state_dict)
    )


def _is_lokr_format(state_dict: dict) -> bool:
    """Check if state dict contains LoKr-format weights (lokr_w1/lokr_w2)."""
    return any("lokr_w" in k for k in state_dict)


def _convert_non_diffusers_z_image_lora_to_diffusers(state_dict):
    """Convert non-diffusers Z-Image LoRA state dict to diffusers format.

    Internalized from diffusers with three fixes:
    1. Handle ``lora_unet__`` (double underscore) — strip longer prefix first
    2. Fix suffix extraction — match ``.lora_down.weight`` / ``.lora_up.weight``
       / ``.alpha`` properly instead of ``rsplit(".", 1)``
    3. Add ``("context", "refiner")`` and ``("noise", "refiner")`` to protected
       n-grams

    Handles:
    - ``diffusion_model.`` prefix removal
    - ``lora_unet_`` / ``lora_unet__`` prefix conversion with key mapping
    - ``default.`` prefix removal
    - ``.lora_down.weight``/``.lora_up.weight`` → ``.lora_A.weight``/``.lora_B.weight``
      conversion with alpha scaling
    """
    has_diffusion_model = any(k.startswith("diffusion_model.") for k in state_dict)
    if has_diffusion_model:
        state_dict = {k.removeprefix("diffusion_model."): v for k, v in state_dict.items()}

    # FIX 1: Handle both lora_unet_ and lora_unet__ prefixes
    # (Fun-distilled format uses double underscore)
    has_lora_unet = any(k.startswith("lora_unet_") for k in state_dict)
    if has_lora_unet:
        # Strip double underscore first (longer prefix), then single
        state_dict = {
            k.removeprefix("lora_unet__").removeprefix("lora_unet_"): v
            for k, v in state_dict.items()
        }

        def convert_key(key: str) -> str:
            # ZImage has: layers, noise_refiner, context_refiner blocks
            # Keys may be like: layers_0_attention_to_q.lora_down.weight

            # FIX 2: Properly separate base path from lora suffix
            suffix = ""
            for sfx in (".lora_down.weight", ".lora_up.weight", ".alpha"):
                if key.endswith(sfx):
                    base = key[: -len(sfx)]
                    suffix = sfx
                    break
            else:
                base = key

            # FIX 3: Protected n-grams including context_refiner and noise_refiner
            protected = {
                ("to", "q"),
                ("to", "k"),
                ("to", "v"),
                ("to", "out"),
                ("feed", "forward"),
                ("context", "refiner"),
                ("noise", "refiner"),
            }

            prot_by_len = {}
            for ng in protected:
                prot_by_len.setdefault(len(ng), set()).add(ng)

            parts = base.split("_")
            merged = []
            i = 0
            lengths_desc = sorted(prot_by_len.keys(), reverse=True)

            while i < len(parts):
                matched = False
                for length in lengths_desc:
                    if i + length <= len(parts) and tuple(parts[i : i + length]) in prot_by_len[length]:
                        merged.append("_".join(parts[i : i + length]))
                        i += length
                        matched = True
                        break
                if not matched:
                    merged.append(parts[i])
                    i += 1

            converted_base = ".".join(merged)
            return converted_base + suffix

        state_dict = {convert_key(k): v for k, v in state_dict.items()}

    def normalize_out_key(k: str) -> str:
        if ".to_out" in k:
            return k
        return re.sub(
            r"\.out(?=\.(?:lora_down|lora_up|lora_A|lora_B)\.weight$|\.alpha$)",
            ".to_out.0",
            k,
        )

    state_dict = {normalize_out_key(k): v for k, v in state_dict.items()}

    has_default = any("default." in k for k in state_dict)
    if has_default:
        state_dict = {k.replace("default.", ""): v for k, v in state_dict.items()}

    converted_state_dict = {}
    all_keys = list(state_dict.keys())
    down_key = ".lora_down.weight"
    up_key = ".lora_up.weight"
    a_key = ".lora_A.weight"
    b_key = ".lora_B.weight"

    has_non_diffusers_lora_id = any(down_key in k or up_key in k for k in all_keys)
    has_diffusers_lora_id = any(a_key in k or b_key in k for k in all_keys)

    if has_non_diffusers_lora_id:

        def get_alpha_scales(down_weight, alpha_key):
            rank = down_weight.shape[0]
            alpha = state_dict.pop(alpha_key).item()
            scale = alpha / rank
            scale_down = scale
            scale_up = 1.0
            while scale_down * 2 < scale_up:
                scale_down *= 2
                scale_up /= 2
            return scale_down, scale_up

        for k in all_keys:
            if k.endswith(down_key):
                diffusers_down_key = k.replace(down_key, ".lora_A.weight")
                diffusers_up_key = k.replace(down_key, up_key).replace(up_key, ".lora_B.weight")
                alpha_key = k.replace(down_key, ".alpha")

                down_weight = state_dict.pop(k)
                up_weight = state_dict.pop(k.replace(down_key, up_key))
                scale_down, scale_up = get_alpha_scales(down_weight, alpha_key)
                converted_state_dict[diffusers_down_key] = down_weight * scale_down
                converted_state_dict[diffusers_up_key] = up_weight * scale_up

    elif has_diffusers_lora_id:
        for k in all_keys:
            if a_key in k or b_key in k:
                converted_state_dict[k] = state_dict.pop(k)
            elif ".alpha" in k:
                state_dict.pop(k)

    if len(state_dict) > 0:
        raise ValueError(f"`state_dict` should be empty at this point but has {state_dict.keys()=}")

    # Split fused QKV keys into separate to_q/to_k/to_v.
    # Original Z-Image format uses attention.qkv with shape (3*hidden, rank) for lora_B.
    split_dict = {}
    for k, v in converted_state_dict.items():
        if ".attention.qkv." not in k:
            split_dict[k] = v
            continue
        if k.endswith(b_key):
            # lora_B: (3*hidden, rank) → split dim 0 into Q, K, V
            q, kk, vv = v.chunk(3, dim=0)
            split_dict[k.replace(".attention.qkv.", ".attention.to_q.")] = q
            split_dict[k.replace(".attention.qkv.", ".attention.to_k.")] = kk
            split_dict[k.replace(".attention.qkv.", ".attention.to_v.")] = vv
        elif k.endswith(a_key):
            # lora_A: (rank, in_features) — shared across Q, K, V
            split_dict[k.replace(".attention.qkv.", ".attention.to_q.")] = v.clone()
            split_dict[k.replace(".attention.qkv.", ".attention.to_k.")] = v.clone()
            split_dict[k.replace(".attention.qkv.", ".attention.to_v.")] = v.clone()

    converted_state_dict = {f"transformer.{k}": v for k, v in split_dict.items()}
    return converted_state_dict


def _convert_zimage_lora(state_dict: dict) -> dict:
    """Convert a non-diffusers Z-Image LoRA to diffusers PEFT format.

    Handles kohya format (lora_unet__ prefixed keys with lora_down/lora_up)
    and other non-diffusers formats (alpha keys, diffusion_model. prefix, etc.).
    Already-diffusers-format LoRAs are returned as-is.
    """
    if _is_diffusers_format(state_dict):
        return state_dict
    return _convert_non_diffusers_z_image_lora_to_diffusers(state_dict)


def _normalize_flux2_lora_keys(state_dict: dict) -> dict:
    """Normalize kohya-format lora_down/lora_up keys to lora_A/lora_B.

    Handles alpha scaling: when .alpha keys are present, bakes scale into weights
    (scale = alpha / rank). When absent, default alpha = rank so scale = 1.0.

    Mixed-format files (some layers already using lora_A/lora_B, others using
    lora_down/lora_up) are handled by preserving existing lora_A/lora_B keys.
    """
    if not any("lora_down" in k for k in state_dict):
        return state_dict

    normalized = {}

    # Preserve keys that already use lora_A/lora_B naming (mixed-format files)
    for k, v in state_dict.items():
        if k.endswith(".lora_A.weight") or k.endswith(".lora_B.weight"):
            normalized[k] = v

    # Group keys by layer prefix (everything before .lora_down/.lora_up/.alpha)
    layer_prefixes = set()
    for k in state_dict:
        for suffix in (".lora_down.weight", ".lora_up.weight", ".alpha"):
            if k.endswith(suffix):
                layer_prefixes.add(k[: -len(suffix)])
                break

    for lp in layer_prefixes:
        down_key = f"{lp}.lora_down.weight"
        up_key = f"{lp}.lora_up.weight"
        alpha_key = f"{lp}.alpha"

        if down_key not in state_dict or up_key not in state_dict:
            continue

        down = state_dict[down_key]
        up = state_dict[up_key]
        rank = down.shape[0]

        alpha = state_dict.get(alpha_key)
        if alpha is not None:
            scale = alpha.item() / rank
        else:
            scale = 1.0

        # Bake alpha scaling into lora_A (matches diffusers convention)
        normalized[f"{lp}.lora_A.weight"] = down * scale if scale != 1.0 else down
        normalized[f"{lp}.lora_B.weight"] = up

    return normalized


def _strip_lora_unet_prefix(sd: dict) -> dict:
    """Convert kohya lora_unet_ prefixed keys to raw Flux2 block format.

    Examples:
        lora_unet_single_blocks_8_linear1.lora_A.weight
            → single_blocks.8.linear1.lora_A.weight
        lora_unet_double_blocks_5_img_attn_qkv.lora_A.weight
            → double_blocks.5.img_attn.qkv.lora_A.weight
    """
    lora_unet = "lora_unet_"
    result = {}
    for k, v in sd.items():
        if not k.startswith(lora_unet):
            result[k] = v
            continue
        remainder = k[len(lora_unet):]
        dot_pos = remainder.find(".")
        layer_path = remainder[:dot_pos] if dot_pos >= 0 else remainder
        suffix = remainder[dot_pos:] if dot_pos >= 0 else ""
        m = re.match(r"^(single_blocks|double_blocks)_(\d+)_(.+)$", layer_path)
        if m:
            block_type, idx, rest = m.groups()
            converted_rest = re.sub(r"^(img|txt)_(attn|mlp)_", r"\1_\2.", rest)
            result[f"{block_type}.{idx}.{converted_rest}{suffix}"] = v
        else:
            result[k] = v
    return result


def _convert_flux2_lora_to_diffusers(state_dict: dict) -> dict:
    """Convert non-diffusers Flux2 LoRA to diffusers PEFT format.

    Unlike diffusers' built-in converter which hardcodes 48 single blocks (full Flux2),
    this dynamically detects block counts from the keys — works for Klein 9B (24 single),
    Klein 4B, or any other variant. Also handles kohya-format lora_down/lora_up naming
    and mixed-format files.
    """
    converted = {}

    # Strip known prefixes (diffusion_model. from ComfyUI, base_model.model. from PEFT)
    strip_prefixes = ("base_model.model.", "diffusion_model.")
    sd = {}
    for k, v in state_dict.items():
        for pfx in strip_prefixes:
            if k.startswith(pfx):
                k = k[len(pfx):]
                break
        sd[k] = v
    sd = _strip_lora_unet_prefix(sd)

    single_indices = set()
    double_indices = set()
    for k in sd:
        if k.startswith("single_blocks."):
            single_indices.add(int(k.split(".")[1]))
        elif k.startswith("double_blocks."):
            double_indices.add(int(k.split(".")[1]))

    lora_keys = ("lora_A", "lora_B")

    for sl in sorted(single_indices):
        src = f"single_blocks.{sl}"
        dst = f"single_transformer_blocks.{sl}.attn"
        single_mappings = [
            ("linear1", "to_qkv_mlp_proj"),
            ("linear2", "to_out"),
        ]
        for org, diff in single_mappings:
            for lk in lora_keys:
                src_key = f"{src}.{org}.{lk}.weight"
                if src_key in sd:
                    converted[f"{dst}.{diff}.{lk}.weight"] = sd.pop(src_key)

    for dl in sorted(double_indices):
        tb = f"transformer_blocks.{dl}"
        for lk in lora_keys:
            for attn_type in ("img_attn", "txt_attn"):
                src_key = f"double_blocks.{dl}.{attn_type}.qkv.{lk}.weight"
                if src_key not in sd:
                    continue
                fused = sd.pop(src_key)
                if lk == "lora_A":
                    proj_keys = (
                        ["to_q", "to_k", "to_v"]
                        if attn_type == "img_attn"
                        else ["add_q_proj", "add_k_proj", "add_v_proj"]
                    )
                    for pk in proj_keys:
                        converted[f"{tb}.attn.{pk}.{lk}.weight"] = torch.cat([fused])
                else:
                    q, k_val, v = torch.chunk(fused, 3, dim=0)
                    if attn_type == "img_attn":
                        converted[f"{tb}.attn.to_q.{lk}.weight"] = q
                        converted[f"{tb}.attn.to_k.{lk}.weight"] = k_val
                        converted[f"{tb}.attn.to_v.{lk}.weight"] = v
                    else:
                        converted[f"{tb}.attn.add_q_proj.{lk}.weight"] = q
                        converted[f"{tb}.attn.add_k_proj.{lk}.weight"] = k_val
                        converted[f"{tb}.attn.add_v_proj.{lk}.weight"] = v

        proj_mappings = [
            ("img_attn.proj", "attn.to_out.0"),
            ("txt_attn.proj", "attn.to_add_out"),
        ]
        mlp_mappings = [
            ("img_mlp.0", "ff.linear_in"),
            ("img_mlp.2", "ff.linear_out"),
            ("txt_mlp.0", "ff_context.linear_in"),
            ("txt_mlp.2", "ff_context.linear_out"),
        ]
        for org, diff in proj_mappings + mlp_mappings:
            for lk in lora_keys:
                src_key = f"double_blocks.{dl}.{org}.{lk}.weight"
                if src_key in sd:
                    converted[f"{tb}.{diff}.{lk}.weight"] = sd.pop(src_key)

    root_mappings = {
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "txt_in": "context_embedder",
        "img_in": "x_embedder",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "final_layer.linear": "proj_out",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    }
    for org, diff in root_mappings.items():
        for lk in lora_keys:
            src_key = f"{org}.{lk}.weight"
            if src_key in sd:
                converted[f"{diff}.{lk}.weight"] = sd.pop(src_key)

    if sd:
        raise ValueError(f"Unexpected keys remaining after Flux2 LoRA conversion: {list(sd.keys())}")

    return {f"transformer.{k}": v for k, v in converted.items()}


def _convert_flux2_lora(state_dict: dict) -> dict:
    """Convert a non-diffusers Flux2 LoRA to diffusers PEFT format.

    Handles kohya format (lora_down/lora_up naming), original format
    (double_blocks/single_blocks), and already-diffusers-format LoRAs.
    """
    if _is_diffusers_format(state_dict):
        return state_dict

    state_dict = _normalize_flux2_lora_keys(state_dict)

    is_original_format = any(
        "double_blocks." in k or ("single_blocks." in k and "single_transformer_blocks." not in k)
        for k in state_dict
    )
    if is_original_format:
        state_dict = _convert_flux2_lora_to_diffusers(state_dict)

    return state_dict


def _map_zimage_lokr_layer_to_targets(layer_prefix: str) -> list[tuple[str, int | None]]:
    """Map a Z-Image LoKr layer prefix to diffusers model parameter targets.

    Z-Image LoKr keys (after stripping diffusion_model. prefix) already match
    the diffusers parameter names, so this is a 1:1 identity mapping.
    """
    if re.match(r"layers\.\d+\.", layer_prefix):
        return [(layer_prefix, None)]
    if re.match(r"(context_refiner|noise_refiner)\.\d+\.", layer_prefix):
        return [(layer_prefix, None)]
    return []


def _map_flux2_lokr_layer_to_targets(layer_prefix: str) -> list[tuple[str, int | None]]:
    """Map a Flux2 LoKr layer prefix to diffusers model parameter targets.

    Returns list of (module_path, split_idx) tuples.
    split_idx is None for 1:1 mappings, or 0/1/2 for QKV chunk index.
    """
    m = re.match(r"single_blocks\.(\d+)\.linear1$", layer_prefix)
    if m:
        return [(f"single_transformer_blocks.{m.group(1)}.attn.to_qkv_mlp_proj", None)]

    m = re.match(r"single_blocks\.(\d+)\.linear2$", layer_prefix)
    if m:
        return [(f"single_transformer_blocks.{m.group(1)}.attn.to_out", None)]

    m = re.match(r"double_blocks\.(\d+)\.img_attn\.qkv$", layer_prefix)
    if m:
        tb = f"transformer_blocks.{m.group(1)}"
        return [(f"{tb}.attn.to_q", 0), (f"{tb}.attn.to_k", 1), (f"{tb}.attn.to_v", 2)]

    m = re.match(r"double_blocks\.(\d+)\.txt_attn\.qkv$", layer_prefix)
    if m:
        tb = f"transformer_blocks.{m.group(1)}"
        return [(f"{tb}.attn.add_q_proj", 0), (f"{tb}.attn.add_k_proj", 1), (f"{tb}.attn.add_v_proj", 2)]

    simple_patterns = [
        (r"double_blocks\.(\d+)\.img_attn\.proj$", "transformer_blocks.{}.attn.to_out.0"),
        (r"double_blocks\.(\d+)\.txt_attn\.proj$", "transformer_blocks.{}.attn.to_add_out"),
        (r"double_blocks\.(\d+)\.img_mlp\.0$", "transformer_blocks.{}.ff.linear_in"),
        (r"double_blocks\.(\d+)\.img_mlp\.2$", "transformer_blocks.{}.ff.linear_out"),
        (r"double_blocks\.(\d+)\.txt_mlp\.0$", "transformer_blocks.{}.ff_context.linear_in"),
        (r"double_blocks\.(\d+)\.txt_mlp\.2$", "transformer_blocks.{}.ff_context.linear_out"),
    ]
    for pattern, template in simple_patterns:
        m = re.match(pattern, layer_prefix)
        if m:
            return [(template.format(m.group(1)), None)]

    root_map = {
        "double_stream_modulation_txt.lin": "double_stream_modulation_txt.linear",
        "double_stream_modulation_img.lin": "double_stream_modulation_img.linear",
        "single_stream_modulation.lin": "single_stream_modulation.linear",
        "txt_in": "context_embedder",
        "img_in": "x_embedder",
        "final_layer.adaLN_modulation.1": "norm_out.linear",
        "final_layer.linear": "proj_out",
        "time_in.in_layer": "time_guidance_embed.timestep_embedder.linear_1",
        "time_in.out_layer": "time_guidance_embed.timestep_embedder.linear_2",
    }
    if layer_prefix in root_map:
        return [(root_map[layer_prefix], None)]

    return []
