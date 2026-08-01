"""Safe login-node CLI for one-puzzle SLURM submission."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import argparse
import os
import re
import shlex
import subprocess
import sys
import uuid
from typing import Callable, Sequence


@dataclass(frozen=True)
class SubmitConfig:
    gpu_ids: tuple[int, ...]
    beam: int
    puzzle: int
    reflect: str
    original_solution: Path | None
    depth: int | None
    dry_run: bool


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True, allow_abbrev=False)
    parser.add_argument("--gpus")
    parser.add_argument("--beam")
    parser.add_argument("--puzzle")
    parser.add_argument("--reflect", choices=("off", "after", "only"), default="off")
    parser.add_argument("--original-solution", type=Path)
    parser.add_argument("--depth")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _positive(value: str | None, option: str, *, allow_zero: bool = False) -> int:
    if value is None:
        raise ValueError(f"missing required {option}")
    if not re.fullmatch(r"[0-9]+", value):
        raise ValueError(f"{option} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    parsed = int(value)
    if parsed < 0 or (parsed == 0 and not allow_zero):
        raise ValueError(f"{option} must be a {'nonnegative' if allow_zero else 'positive'} integer")
    return parsed


def _gpu_ids(value: str | None) -> tuple[int, ...]:
    if value is None:
        raise ValueError("missing required --gpus")
    if not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", value):
        raise ValueError("--gpus must be unique comma-separated nonnegative ids")
    values = tuple(int(part) for part in value.split(","))
    if len(set(values)) != len(values):
        raise ValueError("--gpus must contain unique ids")
    return values


def parse_args(argv: Sequence[str]) -> SubmitConfig:
    args = _parser().parse_args(list(argv))
    gpus = _gpu_ids(args.gpus)
    beam = _positive(args.beam, "--beam")
    if args.puzzle is None:
        raise ValueError("missing required --puzzle; specify one Megaminx puzzle id")
    puzzle = _positive(args.puzzle, "--puzzle", allow_zero=True)
    depth = None if args.depth is None else _positive(args.depth, "--depth")
    if args.reflect == "only" and args.original_solution is None:
        raise ValueError("--reflect only requires --original-solution")
    if args.reflect != "only" and args.original_solution is not None:
        raise ValueError("--original-solution is valid only with --reflect only")
    return SubmitConfig(gpus, beam, puzzle, args.reflect, args.original_solution, depth, args.dry_run)


CLUSTER_KEYS = frozenset({
    "SLURM_PARTITION", "SLURM_ACCOUNT", "SLURM_QOS", "SLURM_TIME",
    "SLURM_CPUS_PER_TASK", "SLURM_MEM", "MEGAMINX_SCRATCH_ROOT",
})


def load_cluster_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid cluster.env line {line_number}")
        key, value = line.split("=", 1)
        if key not in CLUSTER_KEYS:
            raise ValueError(f"unknown cluster.env key: {key}")
        if not value or "," in value or any(ord(char) < 32 for char in value):
            raise ValueError(f"invalid cluster.env value for {key}")
        values[key] = value
    return values

def _safe_export_value(value: object) -> str:
    text = str(value)
    if not text or "," in text or any(ord(char) < 32 for char in text):
        raise ValueError(f"unsafe value for SLURM export: {text!r}")
    return text


def build_sbatch_command(config: SubmitConfig, archive_root: Path, run_dir: Path, cluster: dict[str, str] | None = None) -> list[str]:
    cluster = {} if cluster is None else dict(cluster)
    exports = {
        "ALL": "",
        "MEGAMINX_ARCHIVE_ROOT": archive_root.resolve(),
        "MEGAMINX_RUN_DIR": run_dir.resolve(),
        "MEGAMINX_GPU_IDS": ":".join(str(value) for value in config.gpu_ids),
        "MEGAMINX_BEAM": config.beam,
        "MEGAMINX_PUZZLE": config.puzzle,
        "MEGAMINX_REFLECT": config.reflect,
        **({"MEGAMINX_SCRATCH_ROOT": cluster["MEGAMINX_SCRATCH_ROOT"]} if "MEGAMINX_SCRATCH_ROOT" in cluster else {}),
    }
    if config.depth is not None:
        exports["MEGAMINX_DEPTH"] = config.depth
    if config.original_solution is not None:
        exports["MEGAMINX_ORIGINAL_SOLUTION"] = config.original_solution.resolve()
    export_text = ",".join(
        "ALL" if key == "ALL" else f"{key}={_safe_export_value(value)}"
        for key, value in exports.items()
    )
    command = [
        "sbatch",
        "--parsable",
        "--nodes=1",
        "--ntasks=1",
        f"--gres=gpu:{len(config.gpu_ids)}",
        f"--job-name=megaminx-p{config.puzzle}",
        f"--output={run_dir.resolve() / 'slurm-%j.out'}",
        f"--export={export_text}",
    ]
    option_map = {
        "SLURM_PARTITION": "partition", "SLURM_ACCOUNT": "account",
        "SLURM_QOS": "qos", "SLURM_TIME": "time",
        "SLURM_CPUS_PER_TASK": "cpus-per-task", "SLURM_MEM": "mem",
    }
    command.extend(f"--{option_map[key]}={cluster[key]}" for key in option_map if key in cluster)
    command.append(str(archive_root.resolve() / "scripts" / "job.sh"))
    return command


def _new_run_dir(root: Path, puzzle: int) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "runs" / f"p{puzzle}-{timestamp}-{uuid.uuid4().hex[:8]}"


def main(
    argv: Sequence[str] | None = None,
    *,
    run_sbatch: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    archive_root: Path | None = None,
) -> int:
    try:
        config = parse_args(sys.argv[1:] if argv is None else argv)
        root = (archive_root or Path(__file__).resolve().parent).resolve()
        run_dir = _new_run_dir(root, config.puzzle)
        cluster = load_cluster_env(root / "cluster.env")
        command = build_sbatch_command(config, root, run_dir, cluster)
        if config.dry_run:
            print(shlex.join(command))
            return 0
        job_script = root / "scripts" / "job.sh"
        if not job_script.is_file():
            raise ValueError(f"missing job payload: {job_script}")
        run_dir.mkdir(parents=True, exist_ok=False)
        result = run_sbatch(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {(result.stderr or '').strip()}")
        token = (result.stdout or "").strip().splitlines()[-1].split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", token):
            raise RuntimeError(f"cannot parse sbatch job id from: {result.stdout!r}")
        print(f"submitted_job_id={token} run_dir={run_dir}")
        return 0
    except (ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
