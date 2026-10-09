#!/usr/bin/env python3
"""Fail-closed check of the Cube4 model, generator ordering and puzzle row."""

import argparse
from pathlib import Path
try:
    from tools.cube4_bundle_contract import cube4_tensor_sizes, validate_cube4_bundle
except ModuleNotFoundError:
    from cube4_bundle_contract import cube4_tensor_sizes, validate_cube4_bundle


def required_tensor_names() -> list[str]:
    return list(cube4_tensor_sizes())


def verify(data_dir: Path, weight_dir: Path, puzzle_id: str) -> None:
    validate_cube4_bundle(data_dir, weight_dir, puzzle_id)
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
