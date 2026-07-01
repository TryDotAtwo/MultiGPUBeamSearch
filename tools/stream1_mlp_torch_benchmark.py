#!/usr/bin/env python3
"""Benchmark exported Stream1 MLP weights with PyTorch.

This is a diagnostic benchmark for the exported runtime weight directory. It
uses the same folded position-class input table as the CUDA Stream1 MLP path and
reports candidate throughput as rows * MOVE_COUNT / ms.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import time
from typing import Dict, Iterable, List

import numpy as np
import torch
import torch.nn.functional as F

MOVE_COUNT = 24
SCORE_MAX_Q = 300.0
SCORE_SCALE = 1024.0
SCORE_MAX_KEY = int(SCORE_MAX_Q * SCORE_SCALE)


def load_manifest(weight_dir: Path) -> Dict[str, object]:
    with (weight_dir / "manifest.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def suffix_for(manifest: Dict[str, object]) -> str:
    dtype = str(manifest.get("dtype", "fp16"))
    if dtype != "fp16":
        raise ValueError(f"torch MLP benchmark currently expects fp16 weights, got dtype={dtype}")
    return ".fp16"


def load_half(path: Path, shape: Iterable[int], device: torch.device) -> torch.Tensor:
    shape_tuple = tuple(int(x) for x in shape)
    arr = np.fromfile(path, dtype=np.float16)
    expected = int(np.prod(shape_tuple))
    if arr.size != expected:
        raise ValueError(f"{path} size mismatch: got {arr.size}, expected {expected} for shape={shape_tuple}")
    return torch.from_numpy(arr.reshape(shape_tuple)).to(device=device, dtype=torch.float16)


def parse_initial_state(test_csv: Path, puzzle_id: int, state_len: int) -> List[int]:
    with test_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_id = row.get("id") or row.get("puzzle_id") or row.get("ID")
            if raw_id is None or int(raw_id) != puzzle_id:
                continue
            raw_state = row.get("initial_state") or row.get("state")
            if raw_state is None:
                values = list(row.values())
                raw_state = values[1] if len(values) > 1 else ""
            nums = [int(x) for x in re.findall(r"\d+", raw_state)]
            if len(nums) < state_len:
                raise ValueError(f"puzzle_id={puzzle_id} has only {len(nums)} state values, expected {state_len}")
            return nums[:state_len]
    raise ValueError(f"puzzle_id={puzzle_id} not found in {test_csv}")


class ExportedStream1Mlp:
    def __init__(self, weight_dir: Path, device: torch.device) -> None:
        self.weight_dir = weight_dir
        self.device = device
        self.manifest = load_manifest(weight_dir)
        if str(self.manifest.get("backend", "mlp")) not in ("mlp", ""):
            raise ValueError(f"expected MLP backend manifest, got {self.manifest.get('backend')}")
        normalization = str(self.manifest.get("normalization", "batchnorm_folded"))
        if normalization not in ("batchnorm_folded", "", "none"):
            raise ValueError(f"this benchmark is for folded/no-normalization MLP weights, got normalization={normalization}")
        suffix = suffix_for(self.manifest)
        self.state_len = int(self.manifest["state_len"])
        self.num_classes = int(self.manifest["num_classes"])
        self.hidden1 = int(self.manifest.get("hd1", self.manifest.get("hidden1")))
        self.hidden2 = int(self.manifest.get("hd2", self.manifest.get("hidden2")))
        self.residual_count = int(self.manifest.get("nrd", self.manifest.get("residual_count")))
        self.output_dim = int(self.manifest.get("output_dim", MOVE_COUNT))
        if self.output_dim != MOVE_COUNT:
            raise ValueError(f"this MLP comparison expects output_dim={MOVE_COUNT}, got {self.output_dim}")
        self.input_weight = load_half(weight_dir / f"input_weight_hxk{suffix}", (self.state_len * self.num_classes, self.hidden1), device)
        self.input_bias = load_half(weight_dir / f"input_bias{suffix}", (self.hidden1,), device)
        self.hidden_weight = load_half(weight_dir / f"hidden_weight_hxk{suffix}", (self.hidden1, self.hidden2), device)
        self.hidden_bias = load_half(weight_dir / f"hidden_bias{suffix}", (self.hidden2,), device)
        self.residual_fc1_weight = []
        self.residual_fc1_bias = []
        self.residual_fc2_weight = []
        self.residual_fc2_bias = []
        for block in range(self.residual_count):
            prefix = f"residual{block}"
            self.residual_fc1_weight.append(load_half(weight_dir / f"{prefix}_fc1_weight_hxk{suffix}", (self.hidden2, self.hidden2), device))
            self.residual_fc1_bias.append(load_half(weight_dir / f"{prefix}_fc1_bias{suffix}", (self.hidden2,), device))
            self.residual_fc2_weight.append(load_half(weight_dir / f"{prefix}_fc2_weight_hxk{suffix}", (self.hidden2, self.hidden2), device))
            self.residual_fc2_bias.append(load_half(weight_dir / f"{prefix}_fc2_bias{suffix}", (self.hidden2,), device))
        self.output_weight = load_half(weight_dir / f"output_weight_hxk{suffix}", (self.hidden2, self.output_dim), device)
        self.output_bias = load_half(weight_dir / f"output_bias{suffix}", (self.output_dim,), device)
        self.position_offsets = (torch.arange(self.state_len, device=device, dtype=torch.long) * self.num_classes)

    def make_indices(self, seed_state: torch.Tensor, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        states = seed_state.repeat(rows, 1)
        states[:, 0] = (seed_state[0] + torch.arange(rows, device=self.device, dtype=torch.long)) % self.num_classes
        indices = (states + self.position_offsets).reshape(-1).contiguous()
        offsets = torch.arange(0, rows * self.state_len, self.state_len, device=self.device, dtype=torch.long)
        return indices, offsets

    def forward_preindexed(self, indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        x = F.embedding_bag(indices, self.input_weight, offsets=offsets, mode="sum", include_last_offset=False)
        x = torch.relu(x + self.input_bias)
        x = torch.relu(x.matmul(self.hidden_weight) + self.hidden_bias)
        for block in range(self.residual_count):
            residual = x
            y = torch.relu(x.matmul(self.residual_fc1_weight[block]) + self.residual_fc1_bias[block])
            y = y.matmul(self.residual_fc2_weight[block])
            x = torch.relu(y + residual + self.residual_fc2_bias[block])
        logits = x.matmul(self.output_weight) + self.output_bias
        return torch.clamp(torch.round(torch.clamp(logits, 0.0, SCORE_MAX_Q) * SCORE_SCALE), 0, SCORE_MAX_KEY).to(torch.int32)


def iterations_for_rows(rows: int) -> int:
    if rows >= 131072:
        return 4
    if rows >= 65536:
        return 6
    if rows >= 32768:
        return 8
    return 12


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, default=Path("data/test.csv"))
    parser.add_argument("--puzzle-id", type=int, default=0)
    parser.add_argument("--gpu-label", type=int, default=0)
    parser.add_argument("--rows", type=str, default="2048,4096,8192,16384,32768,65536,131072")
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = ExportedStream1Mlp(args.weight_dir, device)
    seed_values = parse_initial_state(args.test_csv, args.puzzle_id, model.state_len)
    seed_state = torch.tensor(seed_values, device=device, dtype=torch.long)
    rows_values = [int(x) for x in args.rows.split(",") if x.strip()]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows_out = []
    print(
        "torch_mlp_benchmark_start=1"
        f" gpu={args.gpu_label}"
        f" state_len={model.state_len}"
        f" num_classes={model.num_classes}"
        f" hidden1={model.hidden1} hidden2={model.hidden2} nrd={model.residual_count}",
        flush=True,
    )
    with torch.inference_mode():
        for rows in rows_values:
            status = "pass"
            try:
                indices, offsets = model.make_indices(seed_state, rows)
                for _ in range(args.warmup):
                    out = model.forward_preindexed(indices, offsets)
                torch.cuda.synchronize()
                iterations = iterations_for_rows(rows)
                start = torch.cuda.Event(enable_timing=True)
                stop = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    out = model.forward_preindexed(indices, offsets)
                stop.record()
                torch.cuda.synchronize()
                ms = float(start.elapsed_time(stop)) / float(iterations)
                del out, indices, offsets
                parents_per_sec = float(rows) * 1000.0 / ms
                candidates_per_sec = parents_per_sec * float(MOVE_COUNT)
                torch.cuda.empty_cache()
                row = {
                    "gpu": args.gpu_label,
                    "implementation": "torch_mlp_preindexed",
                    "rows_per_launch_group": rows,
                    "ms_per_launch_group": f"{ms:.4f}",
                    "parents_per_sec": f"{parents_per_sec:.1f}",
                    "candidates_per_sec": f"{candidates_per_sec:.1f}",
                    "status": status,
                }
                rows_out.append(row)
                print(
                    "torch_mlp_micro"
                    f" gpu={args.gpu_label}"
                    f" rows_per_launch_group={rows}"
                    f" ms_per_launch_group={ms:.4f}"
                    f" parents_per_sec={parents_per_sec:.1f}"
                    f" candidates_per_sec={candidates_per_sec:.1f}"
                    f" status={status}",
                    flush=True,
                )
            except RuntimeError as exc:
                message = str(exc).replace("\n", " ")[:240]
                if "out of memory" not in message.lower():
                    raise
                torch.cuda.empty_cache()
                row = {
                    "gpu": args.gpu_label,
                    "implementation": "torch_mlp_preindexed",
                    "rows_per_launch_group": rows,
                    "ms_per_launch_group": "",
                    "parents_per_sec": "",
                    "candidates_per_sec": "",
                    "status": "oom",
                }
                rows_out.append(row)
                print(
                    "torch_mlp_micro_skip"
                    f" gpu={args.gpu_label} rows_per_launch_group={rows} reason=oom message={message}",
                    flush=True,
                )
    with args.out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "gpu",
                "implementation",
                "rows_per_launch_group",
                "ms_per_launch_group",
                "parents_per_sec",
                "candidates_per_sec",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"torch_mlp_benchmark_csv={args.out_csv}", flush=True)


if __name__ == "__main__":
    main()