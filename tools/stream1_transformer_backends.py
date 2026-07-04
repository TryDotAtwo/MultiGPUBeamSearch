#!/usr/bin/env python3
"""Explicit Stream1 piece-transformer backend launcher.

This is a thin registry for the three backend families we keep developing:

- pytorch: Python Torch over exported Stream1 weights.
- libtorch: explicit C++ LibTorch tool, built only with BEAM_ENABLE_LIBTORCH_STREAM1=ON.
- native_cutlass: native CUDA/CUTLASS Stream1 transformer path via stream_benchmark.

The registry is intentionally not a fallback chain. A caller selects exactly one
backend and mode; missing tools or dependencies should fail at that backend.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class BackendSpec:
    name: str
    owner: str
    modes: tuple[str, ...]
    default_mode: str
    description: str
    build_hint: str


@dataclass(frozen=True)
class BackendInvocation:
    backend: str
    mode: str
    command: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "mode": self.mode,
            "command": list(self.command),
            "env": dict(self.env),
        }


BACKENDS: Dict[str, BackendSpec] = {
    "pytorch": BackendSpec(
        name="pytorch",
        owner="Python/Torch",
        modes=("eager",),
        default_mode="eager",
        description="Torch implementation over exported piece_transformer weights.",
        build_hint="Requires Python torch and exported weights; no CMake build target.",
    ),
    "libtorch": BackendSpec(
        name="libtorch",
        owner="C++/LibTorch",
        modes=("eager", "cuda_graph"),
        default_mode="eager",
        description="C++ LibTorch implementation using at::linear and SDPA.",
        build_hint="Build stream1_transformer_libtorch_benchmark with BEAM_ENABLE_LIBTORCH_STREAM1=ON.",
    ),
    "native_cutlass": BackendSpec(
        name="native_cutlass",
        owner="C++/CUDA/CUTLASS",
        modes=("eager", "graph"),
        default_mode="graph",
        description="Native CUDA/CUTLASS Stream1 transformer path via stream_benchmark.",
        build_hint="Build stream_benchmark with CUTLASS_DIR set.",
    ),
}


def parse_csv_values(text: str, name: str) -> List[str]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    for value in values:
        int(value)
    return values


def executable_path(build_dir: Path, name: str) -> Path:
    path = build_dir / name
    if os.name == "nt" and not path.exists():
        exe_path = path.with_suffix(".exe")
        if exe_path.exists() or not path.exists():
            return exe_path
    return path


def default_report_path(backend: str, mode: str) -> Path:
    return REPO_ROOT / "test_results" / f"stream1_transformer_{backend}_{mode}.md"


def build_invocation(args: argparse.Namespace) -> BackendInvocation:
    spec = BACKENDS[args.backend]
    mode = args.mode or spec.default_mode
    if mode not in spec.modes:
        raise ValueError(f"backend {args.backend} supports modes {','.join(spec.modes)}, got {mode}")

    weight_dir = Path(args.weight_dir) if args.weight_dir else None
    if weight_dir is None:
        raise ValueError("--weight-dir is required for explicit backend runs")

    if args.backend == "pytorch":
        parse_csv_values(args.batches, "--batches")
        report = Path(args.report) if args.report else default_report_path("pytorch", mode)
        command_parts = [
            sys.executable,
            str(REPO_ROOT / "tools" / "stream1_transformer_torch_benchmark.py"),
            "--weight-dir",
            str(weight_dir),
            "--batch-sizes",
            args.batches,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--report",
            str(report),
        ]
        if args.reference_json:
            command_parts.extend(["--reference-json", args.reference_json])
        if not args.require_reference:
            command_parts.append("--skip-reference")
        return BackendInvocation(args.backend, mode, tuple(command_parts))

    if args.backend == "libtorch":
        parse_csv_values(args.batches, "--batches")
        binary = executable_path(Path(args.build_dir), "stream1_transformer_libtorch_benchmark")
        csv_path = Path(args.csv) if args.csv else default_report_path("libtorch", mode).with_suffix(".csv")
        command_parts: List[str] = [
            str(binary),
            "--weight-dir",
            str(weight_dir),
            "--batches",
            args.batches,
            "--warmup",
            str(args.warmup),
            "--iters",
            str(args.iters),
            "--device",
            args.device,
            "--csv",
            str(csv_path),
        ]
        if mode == "cuda_graph":
            command_parts.append("--cuda-graph")
        return BackendInvocation(args.backend, mode, tuple(command_parts))

    if args.backend == "native_cutlass":
        binary = executable_path(Path(args.build_dir), "stream_benchmark")
        report = Path(args.report) if args.report else default_report_path("native_cutlass", mode)
        env: Dict[str, str] = {
            "BEAM_WEIGHT_DIR": str(weight_dir),
            "BEAM_STREAM_MICRO_ONLY": "1",
            "BEAM_STREAM1_TRANSFORMER_BLOCK51": "1",
            "BEAM_STREAM_BENCH_REPORT": str(report),
        }
        if mode == "graph":
            env["BEAM_STREAM1_TRANSFORMER_GRAPH_BENCH"] = "1"
        if args.synthetic_states:
            env["BEAM_STREAM1_SYNTHETIC_STATES"] = "1"
        if args.b_micro:
            int(args.b_micro)
            env["BEAM_STREAM1_TRANSFORMER_B_MICRO"] = str(args.b_micro)
        if args.concurrency:
            int(args.concurrency)
            env["BEAM_STREAM1_TRANSFORMER_CONCURRENCY"] = str(args.concurrency)
        command = (str(binary), str(args.puzzle_id))
        return BackendInvocation(args.backend, mode, command, tuple(sorted(env.items())))

    raise ValueError(f"unknown backend {args.backend}")


def list_backends() -> List[Dict[str, object]]:
    return [
        {
            "backend": spec.name,
            "owner": spec.owner,
            "modes": list(spec.modes),
            "default_mode": spec.default_mode,
            "description": spec.description,
            "build_hint": spec.build_hint,
        }
        for spec in BACKENDS.values()
    ]


def print_backend_table(rows: Iterable[Mapping[str, object]]) -> None:
    print("backend\tdefault_mode\tmodes\towner\tdescription")
    for row in rows:
        print(
            f"{row['backend']}\t{row['default_mode']}\t{','.join(row['modes'])}\t"
            f"{row['owner']}\t{row['description']}"
        )


def run_invocation(invocation: BackendInvocation) -> int:
    env = os.environ.copy()
    env.update(dict(invocation.env))
    completed = subprocess.run(invocation.command, cwd=REPO_ROOT, env=env, check=False)
    return int(completed.returncode)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list-backends", action="store_true", help="Print the explicit backend registry and exit.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON for list/dry-run output.")
    parser.add_argument("--backend", choices=tuple(BACKENDS), help="Backend to run or describe.")
    parser.add_argument("--mode", help="Backend mode. Defaults to the selected backend default mode.")
    parser.add_argument("--weight-dir", help="Exported Stream1 piece_transformer weight directory.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory for C++ backends.")
    parser.add_argument("--device", default="cuda:0", help="Torch/LibTorch device string.")
    parser.add_argument("--batches", default="384,512,768,1024", help="CSV batch sizes for PyTorch/LibTorch.")
    parser.add_argument("--reference-json", help="Optional PyTorch reference JSON path.")
    parser.add_argument("--require-reference", action="store_true", help="Require PyTorch reference validation instead of adding --skip-reference.")
    parser.add_argument("--synthetic-states", action="store_true", help="Use synthetic arange-pattern states for native CUTLASS parity runs.")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--puzzle-id", default="991", help="Puzzle id passed to native stream_benchmark.")
    parser.add_argument("--b-micro", help="Optional native CUTLASS B_MICRO filter.")
    parser.add_argument("--concurrency", help="Optional native CUTLASS concurrency filter.")
    parser.add_argument("--report", help="Report path for PyTorch/native CUTLASS backends.")
    parser.add_argument("--csv", help="CSV path for LibTorch backend.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected command/env without executing.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.list_backends:
        rows = list_backends()
        if args.json:
            print(json.dumps(rows, indent=2, sort_keys=True))
        else:
            print_backend_table(rows)
        return 0
    if not args.backend:
        parser.error("--backend is required unless --list-backends is used")
    invocation = build_invocation(args)
    if args.dry_run or args.json:
        payload = invocation.as_dict()
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print("backend=" + invocation.backend)
            print("mode=" + invocation.mode)
            if invocation.env:
                print("env=" + json.dumps(dict(invocation.env), sort_keys=True))
            print("command=" + json.dumps(list(invocation.command)))
        return 0
    return run_invocation(invocation)


if __name__ == "__main__":
    raise SystemExit(main())
