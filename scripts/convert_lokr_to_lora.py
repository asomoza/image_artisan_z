#!/usr/bin/env python3
"""Convert a LoKr LoRA safetensors file to standard PEFT (lora_A/lora_B) format.

LoKr LoRAs use Kronecker-product weight merging, which requires saving full
original weights to CPU RAM (~12 GB) and is slower to load/unload than PEFT
adapters. This script decomposes LoKr weights via SVD into standard lora_A/lora_B
tensors that can be loaded as normal LoRAs.

Usage:
    python scripts/convert_lokr_to_lora.py input.safetensors
    python scripts/convert_lokr_to_lora.py input.safetensors -o output.safetensors --rank 128
    python scripts/convert_lokr_to_lora.py input.safetensors --energy 0.999
    python scripts/convert_lokr_to_lora.py input.safetensors --model-type flux2
"""

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

# Add project src to path so we can import conversion utilities
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from iartisanz.modules.generation.graph.nodes.lora_conversion import (
    _is_lokr_format,
    _map_flux2_lokr_layer_to_targets,
    _map_zimage_lokr_layer_to_targets,
)


def _detect_model_type(state_dict: dict) -> str:
    """Auto-detect model type from LoKr key patterns."""
    for key in state_dict:
        if "single_blocks" in key or "double_blocks" in key:
            return "flux2"
    return "zimage"


def _parse_lokr_layers(state_dict: dict, map_fn):
    """Parse LoKr state dict into entries with w1/w2 and target mappings.

    Adapted from lora_node._parse_lokr_entries but without needing a
    transformer model object.
    """
    dm_prefix = "diffusion_model."
    sd = {k[len(dm_prefix):] if k.startswith(dm_prefix) else k: v for k, v in state_dict.items()}

    lokr_w_suffixes = (
        ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w1_b",
        ".lokr_w2_a", ".lokr_w2_b", ".lokr_t2",
    )
    lokr_prefixes = set()
    for k in sd:
        for sfx in lokr_w_suffixes:
            if k.endswith(sfx):
                lokr_prefixes.add(k[: -len(sfx)])
                break

    entries = []
    for lp in sorted(lokr_prefixes):
        w1 = sd.get(f"{lp}.lokr_w1")
        w2 = sd.get(f"{lp}.lokr_w2")
        w1_a = sd.get(f"{lp}.lokr_w1_a")
        w1_b = sd.get(f"{lp}.lokr_w1_b")
        w2_a = sd.get(f"{lp}.lokr_w2_a")
        w2_b = sd.get(f"{lp}.lokr_w2_b")
        t2 = sd.get(f"{lp}.lokr_t2")
        alpha = sd.get(f"{lp}.alpha")

        if w1 is None and w1_a is not None and w1_b is not None:
            w1 = w1_a @ w1_b
        if w2 is None and w2_a is not None and w2_b is not None:
            w2 = w2_a @ w2_b

        if w1 is None or w2 is None:
            print(f"  WARNING: Incomplete layer {lp}, skipping")
            continue

        if t2 is not None:
            w2 = t2 @ w2

        w1_scaled = w1.float()
        if alpha is not None:
            lokr_dim = min(w1.shape)
            scale = alpha.float().item() / lokr_dim
            if 1e-3 <= abs(scale) <= 1e3:
                w1_scaled = w1_scaled * scale

        targets = map_fn(lp)
        if not targets:
            print(f"  WARNING: No target mapping for layer {lp}, skipping")
            continue

        entries.append({"w1": w1_scaled, "w2": w2, "targets": targets})

    return entries


def _determine_rank(S: torch.Tensor, fixed_rank: int | None, energy_threshold: float | None) -> int:
    """Choose SVD truncation rank."""
    max_rank = len(S)

    if energy_threshold is not None:
        cumulative = torch.cumsum(S ** 2, dim=0) / (S ** 2).sum()
        rank = int((cumulative < energy_threshold).sum().item()) + 1
        return min(rank, max_rank)

    return min(fixed_rank, max_rank)


def convert(input_path: Path, output_path: Path, fixed_rank: int, energy_threshold: float | None, model_type: str | None):
    print(f"Loading {input_path}")
    state_dict = load_file(str(input_path))

    if not _is_lokr_format(state_dict):
        print("ERROR: Input file is not in LoKr format (no lokr_w keys found)")
        sys.exit(1)

    if model_type is None:
        model_type = _detect_model_type(state_dict)
    print(f"Model type: {model_type}")

    map_fn = _map_flux2_lokr_layer_to_targets if model_type == "flux2" else _map_zimage_lokr_layer_to_targets

    entries = _parse_lokr_layers(state_dict, map_fn)
    if not entries:
        print("ERROR: No valid LoKr entries found")
        sys.exit(1)

    print(f"Found {len(entries)} LoKr layers\n")

    HIGH_RANK_THRESHOLD = 256
    output_sd = {}
    total_targets = 0
    high_rank_count = 0

    for entry in entries:
        delta = torch.kron(entry["w1"], entry["w2"].float())

        for module_path, split_idx in entry["targets"]:
            if split_idx is not None:
                chunk = delta.chunk(3, dim=0)[split_idx]
            else:
                chunk = delta

            U, S, Vt = torch.linalg.svd(chunk, full_matrices=False)
            rank = _determine_rank(S, fixed_rank, energy_threshold)

            sqrt_S = S[:rank].sqrt()
            lora_B = U[:, :rank] * sqrt_S.unsqueeze(0)  # (out_features, r)
            lora_A = sqrt_S.unsqueeze(1) * Vt[:rank, :]  # (r, in_features)

            # Reconstruction error
            reconstructed = lora_B @ lora_A
            error = (chunk - reconstructed).norm() / chunk.norm()

            # Store in bfloat16 (contiguous for safetensors)
            output_sd[f"transformer.{module_path}.lora_A.weight"] = lora_A.bfloat16().contiguous()
            output_sd[f"transformer.{module_path}.lora_B.weight"] = lora_B.bfloat16().contiguous()
            total_targets += 1
            if rank > HIGH_RANK_THRESHOLD:
                high_rank_count += 1

            print(f"  {module_path}: {list(chunk.shape)} -> rank {rank}, error {error:.6f}")

    if high_rank_count > 0:
        print(f"\n  WARNING: {high_rank_count}/{total_targets} layers needed rank > {HIGH_RANK_THRESHOLD}.")
        print("  This LoKr LoRA has high-rank weight deltas that don't compress well")
        print("  into standard lora_A/lora_B format. The converted file will be much")
        print("  larger than the original LoKr and may not be worth converting.")
        print("  Consider keeping the original LoKr format instead.")

    print(f"\nConverted {total_targets} targets")
    print(f"Saving to {output_path}")
    save_file(output_sd, str(output_path))

    input_size = input_path.stat().st_size / (1024 * 1024)
    output_size = output_path.stat().st_size / (1024 * 1024)
    print(f"Input size:  {input_size:.1f} MB")
    print(f"Output size: {output_size:.1f} MB")
    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Convert LoKr LoRA to standard PEFT format via SVD decomposition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", type=Path, help="Path to LoKr safetensors file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output path (default: {input_stem}_peft.safetensors)")
    parser.add_argument("--rank", type=int, default=64, help="Fixed SVD rank (default: 64)")
    parser.add_argument("--energy", type=float, default=None, help="Adaptive rank via energy threshold (e.g. 0.999). Overrides --rank.")
    parser.add_argument("--model-type", choices=["zimage", "flux2"], default=None, help="Model type (auto-detected if omitted)")

    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: File not found: {args.input}")
        sys.exit(1)

    if args.output is None:
        args.output = args.input.with_stem(f"{args.input.stem}_peft")

    if args.energy is not None and not (0 < args.energy <= 1):
        print("ERROR: --energy must be between 0 and 1 (e.g. 0.999)")
        sys.exit(1)

    convert(
        input_path=args.input,
        output_path=args.output,
        fixed_rank=args.rank,
        energy_threshold=args.energy,
        model_type=args.model_type,
    )


if __name__ == "__main__":
    main()
