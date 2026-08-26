#!/usr/bin/env python3
"""Build deterministic Cube4 frontier corpora and SM120 block-FP8 statistics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch


HISTORY_ENTRY = struct.Struct("<QII")
FP8_MAX = 448.0
INT8_MAX = 127.0
BLOCK = 128
NVFP4_MAX = 6.0
NVFP4_BLOCK = 16
NVFP4_SCALE_MIN = 2.0 ** -9
_NVFP4_POSITIVE_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _read_history(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) % HISTORY_ENTRY.size:
        raise ValueError(f"{path}: history byte count is not divisible by 16")
    records = np.frombuffer(data, dtype=np.dtype([("parent", "<u8"), ("route", "<u4"), ("reserved", "<u4")]))
    if np.any(records["reserved"] != 0):
        raise ValueError(f"{path}: reserved history field must be zero")
    return records["parent"].astype(np.int64), (records["route"] & 0xFF).astype(np.int64)


def reconstruct_frontiers(
    history_dir: Path,
    initial_state: np.ndarray,
    generators: np.ndarray,
) -> dict[int, np.ndarray]:
    history_dir = Path(history_dir)
    initial_state = np.asarray(initial_state, dtype=np.uint8)
    generators = np.asarray(generators, dtype=np.int64)
    if initial_state.ndim != 1 or generators.ndim != 2 or generators.shape[1] != initial_state.size:
        raise ValueError("initial state and generator shapes are incompatible")
    if np.any(generators < 0) or np.any(generators >= initial_state.size):
        raise ValueError("generator index is outside the logical state")
    current = initial_state.reshape(1, -1)
    result: dict[int, np.ndarray] = {}
    paths = sorted(
        history_dir.glob("depth_*.candidate_meta.bin"),
        key=lambda path: int(path.name.split("_")[1].split(".")[0]),
    )
    if not paths:
        raise ValueError(f"no candidate history files found under {history_dir}")
    for expected_depth, path in enumerate(paths):
        depth = int(path.name.split("_")[1].split(".")[0])
        if depth != expected_depth:
            raise ValueError(f"history depths must be contiguous from zero, got {depth}")
        parents, moves = _read_history(path)
        if np.any(parents < 0) or np.any(parents >= current.shape[0]):
            raise ValueError(f"{path}: parent index exceeds prior frontier")
        if np.any(moves < 0) or np.any(moves >= generators.shape[0]):
            raise ValueError(f"{path}: move index exceeds generator table")
        selected = current[parents]
        next_frontier = np.take_along_axis(selected, generators[moves], axis=1).astype(np.uint8, copy=False)
        result[depth] = np.ascontiguousarray(next_frontier)
        current = next_frontier
    return result


def stratified_reservoir(
    frontiers: Mapping[int, np.ndarray],
    *,
    depths: Sequence[int],
    per_depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if per_depth <= 0:
        raise ValueError("per_depth must be positive")
    states: list[np.ndarray] = []
    depth_labels: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    for depth in depths:
        if depth not in frontiers:
            raise ValueError(f"requested depth {depth} is missing")
        frontier = np.asarray(frontiers[depth], dtype=np.uint8)
        count = min(per_depth, frontier.shape[0])
        indices = np.floor(np.arange(count, dtype=np.float64) * frontier.shape[0] / count).astype(np.int64)
        states.append(frontier[indices])
        depth_labels.append(np.full(count, depth, dtype=np.int16))
        source_indices.append(indices)
    return (
        np.ascontiguousarray(np.concatenate(states)),
        np.concatenate(depth_labels),
        np.concatenate(source_indices),
    )


def _fp8_qdq(grouped: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    scaled = torch.clamp(grouped / scale, -FP8_MAX, FP8_MAX)
    return scaled.to(torch.float8_e4m3fn).to(grouped.dtype) * scale


def fake_sm120_activation_quant(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if tensor.shape[-1] % BLOCK:
        raise ValueError("SM120 activation K dimension must be divisible by 128")
    shape = tensor.shape
    flat = tensor.reshape(-1, shape[-1])
    grouped = flat.reshape(flat.shape[0], flat.shape[1] // BLOCK, BLOCK)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    quantized = _fp8_qdq(grouped, scale).reshape(shape)
    return quantized, scale.squeeze(-1)


def fake_sm120_weight_quant(weight_hxk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if weight_hxk.ndim != 2 or weight_hxk.shape[0] % BLOCK or weight_hxk.shape[1] % BLOCK:
        raise ValueError("SM120 weight H/O dimensions must be divisible by 128")
    h_blocks = weight_hxk.shape[0] // BLOCK
    o_blocks = weight_hxk.shape[1] // BLOCK
    grouped = weight_hxk.reshape(h_blocks, BLOCK, o_blocks, BLOCK)
    amax = grouped.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    quantized = _fp8_qdq(grouped, scale).reshape_as(weight_hxk)
    return quantized, scale[:, 0, :, 0]


def _ue4m3_scale_qdq(raw_scale: torch.Tensor) -> torch.Tensor:
    """Round positive NVFP4 scales to the hardware UE4M3 value set.

    Positive E4M3FN and UE4M3 have the same finite encoding.  The explicit
    lower clamp prevents a non-zero block from acquiring a zero scale after
    conversion.  Zero blocks are represented with the neutral scale one.
    """
    finite = torch.nan_to_num(
        raw_scale.float(), nan=1.0, posinf=448.0, neginf=NVFP4_SCALE_MIN
    )
    finite = finite.clamp(NVFP4_SCALE_MIN, 448.0)
    return finite.to(torch.float8_e4m3fn).float()


def _e2m1_qdq(values: torch.Tensor) -> torch.Tensor:
    """Deterministic nearest-value emulation of CUTLASS float_e2m1_t."""
    levels = values.new_tensor(_NVFP4_POSITIVE_LEVELS)
    magnitude = values.abs().unsqueeze(-1)
    indices = torch.argmin((magnitude - levels).abs(), dim=-1)
    return torch.copysign(levels[indices], values)


def fake_sm120_nvfp4_activation_quant(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logical NVFP4 QDQ: per-row/K16 E2M1 values with UE4M3 scales."""
    if tensor.shape[-1] % NVFP4_BLOCK:
        raise ValueError("SM120 NVFP4 activation K dimension must be divisible by 16")
    shape = tensor.shape
    flat = tensor.float().reshape(-1, shape[-1])
    grouped = flat.reshape(flat.shape[0], flat.shape[1] // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    raw_scale = torch.where(amax > 0, amax / NVFP4_MAX, torch.ones_like(amax))
    scale = _ue4m3_scale_qdq(raw_scale)
    quantized = _e2m1_qdq((grouped / scale).clamp(-NVFP4_MAX, NVFP4_MAX)) * scale
    return quantized.reshape(shape).to(tensor.dtype), scale.squeeze(-1)


def fake_sm120_nvfp4_weight_quant(
    weight_hxk: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Logical offline NVFP4 QDQ: per-output/K16 E2M1 + UE4M3 scale."""
    if weight_hxk.ndim != 2 or weight_hxk.shape[0] % NVFP4_BLOCK:
        raise ValueError("SM120 NVFP4 weight K dimension must be divisible by 16")
    weight = weight_hxk.float()
    # Runtime HxO weights are GEMM B[K,N].  Every output column owns one
    # UE4M3 scale for each contiguous K16 vector.
    grouped = weight.reshape(
        weight.shape[0] // NVFP4_BLOCK, NVFP4_BLOCK, weight.shape[1]
    ).permute(0, 2, 1)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    raw_scale = torch.where(amax > 0, amax / NVFP4_MAX, torch.ones_like(amax))
    scale = _ue4m3_scale_qdq(raw_scale)
    quantized = _e2m1_qdq((grouped / scale).clamp(-NVFP4_MAX, NVFP4_MAX)) * scale
    restored = quantized.permute(0, 2, 1).reshape_as(weight_hxk)
    return restored.to(weight_hxk.dtype), scale.squeeze(-1)


def fake_sm120_int8_activation_quant(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-row/per-K128 INT8 QDQ with deterministic ties-to-even."""
    if tensor.shape[-1] % BLOCK:
        raise ValueError("SM120 INT8 activation K dimension must be divisible by 128")
    shape = tensor.shape
    flat = tensor.float().reshape(-1, shape[-1])
    grouped = flat.reshape(flat.shape[0], flat.shape[1] // BLOCK, BLOCK)
    amax = grouped.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax > 0, amax / INT8_MAX, torch.ones_like(amax))
    quantized = torch.round(grouped / scale).clamp(-INT8_MAX, INT8_MAX) * scale
    return quantized.reshape(shape).to(tensor.dtype), scale.squeeze(-1)


def fake_sm120_int8_weight_quant(weight_hxk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-K128/N128 offline INT8 QDQ."""
    if weight_hxk.ndim != 2 or weight_hxk.shape[0] % BLOCK or weight_hxk.shape[1] % BLOCK:
        raise ValueError("SM120 INT8 weight H/O dimensions must be divisible by 128")
    grouped = weight_hxk.float().reshape(
        weight_hxk.shape[0] // BLOCK, BLOCK,
        weight_hxk.shape[1] // BLOCK, BLOCK,
    )
    amax = grouped.abs().amax(dim=(1, 3), keepdim=True)
    scale = torch.where(amax > 0, amax / INT8_MAX, torch.ones_like(amax))
    quantized = torch.round(grouped / scale).clamp(-INT8_MAX, INT8_MAX) * scale
    return quantized.reshape_as(weight_hxk).to(weight_hxk.dtype), scale[:, 0, :, 0]


def activation_block_statistics(tensor: torch.Tensor) -> dict[str, float | int]:
    if tensor.shape[-1] % BLOCK:
        raise ValueError("activation statistics require K divisible by 128")
    flat = tensor.detach().float().reshape(-1, tensor.shape[-1])
    grouped = flat.reshape(flat.shape[0], flat.shape[1] // BLOCK, BLOCK)
    amax = grouped.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    quantized, _ = fake_sm120_activation_quant(flat)
    q_grouped = quantized.reshape_as(grouped)
    exponent = torch.floor(torch.log2(scale))
    original_nonzero = grouped != 0
    quantized_zero = (q_grouped == 0) & original_nonzero
    q_values = torch.clamp(grouped / scale.unsqueeze(-1), -FP8_MAX, FP8_MAX)
    return {
        "rows": int(flat.shape[0]),
        "features": int(flat.shape[1]),
        "blocks_per_row": int(grouped.shape[1]),
        "abs_max": float(grouped.abs().max().item()),
        "rms": float(grouped.square().mean().sqrt().item()),
        "zero_block_fraction": float((amax == 0).float().mean().item()),
        "scale_exponent_min": float(exponent.min().item()),
        "scale_exponent_max": float(exponent.max().item()),
        "quantized_zero_fraction": float(quantized_zero.float().mean().item()),
        "saturation_fraction": float((q_values.abs() >= FP8_MAX).float().mean().item()),
        "qdq_nmse": float(((quantized.float() - flat).square().sum() / flat.square().sum().clamp_min(1e-30)).item()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_initial_state(test_csv: Path) -> np.ndarray:
    with test_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1:
        raise ValueError("calibration test.csv must contain exactly one puzzle")
    text = rows[0]["initial_state"]
    separator = ";" if ";" in text else ","
    return np.asarray([int(value) for value in text.split(separator)], dtype=np.uint8)


def generator_actions(payload: Mapping[str, object]) -> np.ndarray:
    """Accept both public generator schemas without silently guessing."""
    has_actions = "actions" in payload
    has_moves = "moves" in payload
    if has_actions == has_moves:
        raise ValueError("generator JSON must contain exactly one of actions or moves")
    actions = np.asarray(payload["actions" if has_actions else "moves"], dtype=np.int64)
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] == 0:
        raise ValueError("generator action table must be a non-empty matrix")
    return actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument("--generator-json", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--depths", default="4,5,6,7,8")
    parser.add_argument("--per-depth", type=int, default=4096)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generator_payload = json.loads(args.generator_json.read_text(encoding="utf-8"))
    generators = generator_actions(generator_payload)
    initial = _load_initial_state(args.test_csv)
    frontiers = reconstruct_frontiers(args.history_dir, initial, generators)
    depths = tuple(int(value) for value in args.depths.split(",") if value)
    states, labels, indices = stratified_reservoir(frontiers, depths=depths, per_depth=args.per_depth)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, states=states, depths=labels, source_indices=indices)
    manifest = {
        "schema_version": 1,
        "states": int(states.shape[0]),
        "state_len": int(states.shape[1]),
        "depths": list(depths),
        "per_depth": args.per_depth,
        "corpus_sha256": _sha256(args.output),
        "generator_sha256": _sha256(args.generator_json),
        "test_csv_sha256": _sha256(args.test_csv),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
