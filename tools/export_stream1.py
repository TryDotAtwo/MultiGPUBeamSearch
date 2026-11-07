#!/usr/bin/env python3
"""Auto-detect and export Stream1 model weights."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from export_stream1_mlp import (
    export_batchnorm_folded,
    export_resmlp_layernorm,
    strip_orig_mod,
    unwrap_state_dict,
)
from export_stream1_transformer import (
    export_piece_transformer,
    find_generator,
    find_metadata,
    load_json,
    strip_prefixes,
)


def detect_format(weights: Path) -> str:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    sd = strip_prefixes(strip_orig_mod(unwrap_state_dict(checkpoint)))
    keys = set(sd)
    if {
        "local_value_embedding.weight",
        "piece_projection.weight",
        "blocks.0.attn.in_proj_weight",
        "output_layer.weight",
    }.issubset(keys):
        return "piece-transformer"
    if {"embedding.weight", "input_stack.0.weight", "head.weight"}.issubset(keys):
        return "resmlp-layernorm"
    if {"input_layer.weight", "hidden_layer.weight", "output_layer.weight", "bn1.weight", "bn2.weight"}.issubset(keys):
        return "batchnorm-folded"
    sample = ", ".join(sorted(keys)[:20])
    raise ValueError(f"cannot auto-detect Stream1 checkpoint format; first_keys={sample}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--format", choices=["auto", "piece-transformer", "resmlp-layernorm", "batchnorm-folded"], default="auto")
    parser.add_argument("--num-classes", type=int, default=120)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--generators", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    args = parser.parse_args()

    fmt = detect_format(args.weights) if args.format == "auto" else args.format
    if fmt == "piece-transformer":
        metadata = args.metadata or find_metadata(args.weights)
        metadata_json = load_json(metadata)
        group_id = int(metadata_json["group_id"]) if "group_id" in metadata_json else None
        generators = args.generators or find_generator(args.weights, group_id)
        export_piece_transformer(args.weights, args.out, args.dtype, metadata, generators, args.source_root)
    elif fmt == "resmlp-layernorm":
        export_resmlp_layernorm(args.weights, args.out, args.dtype)
    elif fmt == "batchnorm-folded":
        export_batchnorm_folded(args.weights, args.out, args.dtype, args.num_classes)
    else:
        raise AssertionError(fmt)


if __name__ == "__main__":
    main()
