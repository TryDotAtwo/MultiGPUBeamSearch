#!/usr/bin/env python3
"""Fail-closed check of the Cube4 model, generator ordering and puzzle row."""

import argparse
import csv
import json
from pathlib import Path


def required_tensor_names() -> list[str]:
    names = ["cls_token.fp16", "fast_piece_static.fp16", "fast_slot_projected.fp16",
             "input_ln_beta.fp16", "input_ln_gamma.fp16", "output_bias.fp16",
             "output_ln_beta.fp16", "output_ln_gamma.fp16", "output_weight_hxk.fp16",
             "piece_mask.u8", "piece_positions.u16", "piece_types.u8"]
    for block in range(4):
        names.extend(f"block{block}_{suffix}.fp16" for suffix in (
            "attn_out_bias", "attn_out_weight_hxk", "attn_qkv_bias", "attn_qkv_weight_hxk",
            "ff1_bias", "ff1_weight_hxk", "ff2_bias", "ff2_weight_hxk",
            "ln1_beta", "ln1_gamma", "ln2_beta", "ln2_gamma"))
    return names


def verify(data_dir: Path, weight_dir: Path, puzzle_id: str) -> None:
    puzzle = json.loads((data_dir / "puzzle_info.json").read_text(encoding="utf-8"))
    model = json.loads((weight_dir / "manifest.json").read_text(encoding="utf-8"))
    if len(puzzle["central_state"]) != 96:
        raise ValueError("Cube4 puzzle central_state must have 96 entries")
    generators = puzzle["generators"]
    moves = list(generators)
    if len(moves) != 24 or any(len(generators[name]) != 96 for name in moves):
        raise ValueError("Cube4 requires 24 permutations of 96 positions")
    expected = {"backend": "piece_transformer", "state_len": 96, "move_count": 24,
                "output_dim": 24, "seq_len": 57, "dtype": "fp16", "activation": "relu", "pooling": "cls"}
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(f"model {key} must be {value!r}, got {model.get(key)!r}")
    if model.get("move_names") != moves:
        raise ValueError("model output order differs from puzzle generators")
    missing = [name for name in required_tensor_names()
               if not (weight_dir / name).is_file() or (weight_dir / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"missing or empty model tensors: {', '.join(missing[:5])}")
    with (data_dir / "test.csv").open(newline="", encoding="utf-8") as stream:
        rows = csv.DictReader(stream)
        if not {"initial_state_id", "initial_state"}.issubset(rows.fieldnames or []):
            raise ValueError("test.csv is missing required columns")
        matches = [row for row in rows if row["initial_state_id"] == puzzle_id]
    if len(matches) != 1:
        raise ValueError(f"puzzle {puzzle_id} must occur exactly once in test.csv")
    if len(matches[0]["initial_state"].split(",")) != 96:
        raise ValueError(f"puzzle {puzzle_id} initial_state must have 96 entries")
    print(f"cube4_bundle_ok puzzle={puzzle_id} state=96 moves=24 seq=57 dtype=fp16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("weight_dir", type=Path)
    parser.add_argument("puzzle_id")
    args = parser.parse_args()
    verify(args.data_dir, args.weight_dir, args.puzzle_id)


if __name__ == "__main__":
    main()
