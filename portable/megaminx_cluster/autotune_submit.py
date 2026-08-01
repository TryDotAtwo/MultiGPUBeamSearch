"""Safe login-node CLI for adaptive cluster profile tuning."""
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

from portable.megaminx_cluster.submit import default_archive_root, load_cluster_env


@dataclass(frozen=True)
class AutotuneSubmitConfig:
    gpu_ids: tuple[int, ...]
    puzzle_ids: tuple[int, int, int]
    min_beam: int
    time_budget_seconds: int
    bfs_hash_budget_mib: int
    dry_run: bool


def _positive(text: str, option: str, *, allow_zero: bool = False) -> int:
    if not re.fullmatch(r"[0-9]+", text or ""):
        raise ValueError(f"{option} must be a positive integer")
    value = int(text)
    if value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{option} must be a positive integer")
    return value


def _ids(text: str | None, option: str, count: int | None = None) -> tuple[int, ...]:
    if text is None or not re.fullmatch(r"[0-9]+(?:,[0-9]+)*", text):
        raise ValueError(f"missing or invalid {option}")
    values = tuple(int(item) for item in text.split(","))
    if len(values) != len(set(values)) or (count is not None and len(values) != count):
        raise ValueError(f"{option} requires {count or 'unique'} unique ids")
    return values


def _duration_seconds(text: str) -> int:
    match = re.fullmatch(r"([1-9][0-9]*)([hms])", text)
    if not match:
        raise ValueError("--time-budget must use positive h, m, or s syntax")
    scale = {"h": 3600, "m": 60, "s": 1}[match.group(2)]
    return int(match.group(1)) * scale


def parse_args(argv: Sequence[str]) -> AutotuneSubmitConfig:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--gpus")
    parser.add_argument("--puzzles", default="900,950,1000")
    parser.add_argument("--min-beam", default="30000000")
    parser.add_argument("--time-budget", default="6h")
    parser.add_argument("--bfs-hash-budget-mib", default="256")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(list(argv))
    gpus = _ids(args.gpus, "--gpus")
    puzzles = _ids(args.puzzles, "--puzzles", count=3)
    return AutotuneSubmitConfig(
        gpus,
        (puzzles[0], puzzles[1], puzzles[2]),
        _positive(args.min_beam, "--min-beam"),
        _duration_seconds(args.time_budget),
        _positive(args.bfs_hash_budget_mib, "--bfs-hash-budget-mib"),
        args.dry_run,
    )


def _safe(value: object) -> str:
    text = str(value)
    if not text or "," in text or any(ord(char) < 32 for char in text):
        raise ValueError(f"unsafe SLURM export value: {text!r}")
    return text


def build_autotune_sbatch_command(
    config: AutotuneSubmitConfig,
    archive_root: Path,
    run_dir: Path,
    cluster: dict[str, str] | None = None,
) -> list[str]:
    cluster = {} if cluster is None else dict(cluster)
    exports = {
        "ALL": "",
        "MEGAMINX_ARCHIVE_ROOT": archive_root.resolve(),
        "MEGAMINX_AUTOTUNE_RUN_DIR": run_dir.resolve(),
        "MEGAMINX_AUTOTUNE_GPU_IDS": ":".join(map(str, config.gpu_ids)),
        "MEGAMINX_AUTOTUNE_PUZZLES": ":".join(map(str, config.puzzle_ids)),
        "MEGAMINX_AUTOTUNE_MIN_BEAM": config.min_beam,
        "MEGAMINX_AUTOTUNE_TIME_BUDGET_SECONDS": config.time_budget_seconds,
        "MEGAMINX_AUTOTUNE_BFS_HASH_BUDGET_MIB": config.bfs_hash_budget_mib,
    }
    if "MEGAMINX_SCRATCH_ROOT" in cluster:
        exports["MEGAMINX_SCRATCH_ROOT"] = cluster["MEGAMINX_SCRATCH_ROOT"]
    export_text = ",".join(
        "ALL" if key == "ALL" else f"{key}={_safe(value)}" for key, value in exports.items()
    )
    command = [
        "sbatch", "--parsable", "--nodes=1", "--ntasks=1",
        f"--gres=gpu:{len(config.gpu_ids)}", "--job-name=megaminx-autotune",
        f"--output={run_dir.resolve() / 'slurm-%j.out'}", f"--export={export_text}",
    ]
    option_map = {
        "SLURM_PARTITION": "partition", "SLURM_ACCOUNT": "account",
        "SLURM_QOS": "qos", "SLURM_TIME": "time",
        "SLURM_CPUS_PER_TASK": "cpus-per-task", "SLURM_MEM": "mem",
    }
    command.extend(f"--{option_map[key]}={cluster[key]}" for key in option_map if key in cluster)
    command.append(str(archive_root.resolve() / "scripts" / "autotune_job.sh"))
    return command


def _new_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return root / "autotune-runs" / f"tune-{stamp}-{uuid.uuid4().hex[:8]}"


def main(
    argv: Sequence[str] | None = None,
    *,
    run_sbatch: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    archive_root: Path | None = None,
) -> int:
    try:
        config = parse_args(sys.argv[1:] if argv is None else argv)
        root = (archive_root or default_archive_root(Path(__file__))).resolve()
        run_dir = _new_run_dir(root)
        command = build_autotune_sbatch_command(config, root, run_dir, load_cluster_env(root / "cluster.env"))
        if config.dry_run:
            print(shlex.join(command))
            return 0
        if not (root / "scripts" / "autotune_job.sh").is_file():
            raise ValueError("missing autotune job payload")
        run_dir.mkdir(parents=True, exist_ok=False)
        result = run_sbatch(command, text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {(result.stderr or '').strip()}")
        token = (result.stdout or "").strip().splitlines()[-1].split(";", 1)[0]
        if not re.fullmatch(r"[0-9]+", token):
            raise RuntimeError(f"cannot parse sbatch job id from: {result.stdout!r}")
        print(f"submitted_job_id={token} run_dir={run_dir}")
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
