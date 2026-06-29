#!/usr/bin/env python3
"""Auto-detecting Stream1 weight exporter wrapper."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

import torch

try:
    from tools.export_stream1_mlp import (
        export_batchnorm_folded,
        export_resmlp_layernorm,
        strip_orig_mod,
        unwrap_state_dict,
    )
    from tools.export_stream1_transformer import export_piece_transformer, strip_state_prefixes
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from export_stream1_mlp import export_batchnorm_folded, export_resmlp_layernorm, strip_orig_mod, unwrap_state_dict
    from export_stream1_transformer import export_piece_transformer, strip_state_prefixes


PIECE_TRANSFORMER_KEYS = {
    "local_value_embedding.weight",
    "piece_projection.weight",
    "blocks.0.attn.in_proj_weight",
    "output_layer.weight",
}


def detect_format_from_state_dict(sd: Mapping[str, object]) -> str:
    keys = set(strip_state_prefixes(sd).keys())
    if PIECE_TRANSFORMER_KEYS.issubset(keys):
        return "piece-transformer"
    if {"embedding.weight", "input_stack.0.weight", "head.weight"}.issubset(keys):
        return "resmlp-layernorm"
    if {"input_layer.weight", "bn1.running_mean", "output_layer.weight"}.issubset(keys):
        return "batchnorm-folded"
    raise ValueError("could not auto-detect Stream1 weight format; pass --format explicitly")


def load_state_for_detection(weights_path: Path) -> Mapping[str, object]:
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    return strip_orig_mod(unwrap_state_dict(checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument(
        "--format",
        choices=["auto", "piece-transformer", "resmlp-layernorm", "batchnorm-folded"],
        default="auto",
    )
    parser.add_argument("--num-classes", type=int, default=120)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--generators", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--reference-out", type=Path)
    parser.add_argument("--reference-count", type=int, default=0)
    parser.add_argument("--reference-seed", type=int, default=0)
    args = parser.parse_args()

    resolved_format = args.format
    if resolved_format == "auto":
        resolved_format = detect_format_from_state_dict(load_state_for_detection(args.weights))

    if resolved_format == "piece-transformer":
        export_piece_transformer(
            weights_path=args.weights,
            out_dir=args.out,
            dtype=args.dtype,
            num_classes=args.num_classes,
            metadata_path=args.metadata,
            generator_path=args.generators,
            source_root=args.source_root,
            reference_out=args.reference_out,
            reference_count=args.reference_count,
            reference_seed=args.reference_seed,
        )
    elif resolved_format == "resmlp-layernorm":
        if args.reference_out is not None:
            raise ValueError("--reference-out is only supported for --format piece-transformer")
        export_resmlp_layernorm(args.weights, args.out, args.dtype)
    elif resolved_format == "batchnorm-folded":
        if args.reference_out is not None:
            raise ValueError("--reference-out is only supported for --format piece-transformer")
        export_batchnorm_folded(args.weights, args.out, args.dtype, args.num_classes)
    else:
        raise ValueError(f"unsupported Stream1 format: {resolved_format}")


if __name__ == "__main__":
    main()
