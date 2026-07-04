#!/usr/bin/env python3
"""Compare explicit Stream1 piece-transformer backends on one synthetic batch.

This tool is a correctness gate, not a fallback path. It runs selected backends
through ``tools.stream1_transformer_backends`` and compares the first row of
quantized score keys. Full tensor dumps can be added later; the first-row gate is
small enough to keep in normal benchmark logs and catches backend contract drift.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.stream1_transformer_backends import BACKENDS, BackendInvocation, build_invocation


PAIR_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>\S+)")
ROW_PREFIXES = {
    "pytorch": "torch_stream1_transformer",
    "libtorch": "stream1_transformer_libtorch_micro",
    "native_cutlass": "stream1_transformer_micro",
}


@dataclass
class BackendRun:
    backend: str
    mode: str
    command: List[str]
    env: Dict[str, str]
    return_code: int
    checksum: int | None
    score_key_digest: int | None
    first_score_keys: List[int]
    log_path: str
    status: str


def parse_csv_ints(text: str) -> List[int]:
    if not text:
        return []
    return [int(part) for part in text.split(",") if part]


def parse_pairs(line: str) -> Dict[str, str]:
    return {match.group("key"): match.group("value") for match in PAIR_RE.finditer(line)}


def make_args(
    backend: str,
    mode: str,
    weight_dir: Path,
    build_dir: Path,
    device: str,
    batch: int,
    warmup: int,
    iters: int,
    out_dir: Path,
) -> argparse.Namespace:
    report = out_dir / f"{backend}_{mode}.md"
    csv = out_dir / f"{backend}_{mode}.csv"
    return argparse.Namespace(
        backend=backend,
        mode=mode,
        weight_dir=str(weight_dir),
        build_dir=str(build_dir),
        device=device,
        batches=str(batch),
        reference_json=None,
        require_reference=False,
        synthetic_states=True,
        warmup=warmup,
        iters=iters,
        puzzle_id="991",
        b_micro=str(batch) if backend == "native_cutlass" else None,
        concurrency="1" if backend == "native_cutlass" else None,
        report=str(report),
        csv=str(csv),
    )


def run_backend(invocation: BackendInvocation, out_dir: Path) -> BackendRun:
    env = os.environ.copy()
    env.update(dict(invocation.env))
    log_path = out_dir / f"{invocation.backend}_{invocation.mode}.log"
    completed = subprocess.run(
        invocation.command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    log_path.write_text(completed.stdout, encoding="utf-8")

    parsed: Dict[str, str] = {}
    prefix = ROW_PREFIXES[invocation.backend]
    for line in completed.stdout.splitlines():
        if line.startswith(prefix):
            parsed = parse_pairs(line)

    checksum = int(parsed["checksum"]) if "checksum" in parsed else None
    score_digest = int(parsed["score_key_digest"]) if "score_key_digest" in parsed else None
    first_score_keys = parse_csv_ints(parsed.get("first_score_keys", ""))
    status = "ok" if completed.returncode == 0 and checksum is not None and score_digest is not None and first_score_keys else "failed"
    return BackendRun(
        backend=invocation.backend,
        mode=invocation.mode,
        command=list(invocation.command),
        env=dict(invocation.env),
        return_code=completed.returncode,
        checksum=checksum,
        score_key_digest=score_digest,
        first_score_keys=first_score_keys,
        log_path=str(log_path),
        status=status,
    )


def compare_runs(runs: List[BackendRun], tolerance: int) -> Dict[str, object]:
    ok_runs = [run for run in runs if run.status == "ok"]
    if not ok_runs:
        return {"status": "failed", "reason": "no backend produced score keys", "comparisons": []}
    baseline = ok_runs[0]
    if baseline.score_key_digest is None:
        return {"status": "failed", "reason": "baseline missing score key digest", "comparisons": []}
    comparisons = []
    overall = "pass"
    for run in runs:
        if run.status != "ok":
            overall = "failed"
            comparisons.append({"backend": run.backend, "mode": run.mode, "status": run.status})
            continue
        if run.score_key_digest is None:
            overall = "failed"
            comparisons.append({
                "backend": run.backend,
                "mode": run.mode,
                "status": "failed",
                "reason": "missing score key digest",
            })
            continue
        if run.score_key_digest != baseline.score_key_digest:
            overall = "failed"
            comparisons.append({
                "backend": run.backend,
                "mode": run.mode,
                "status": "failed",
                "reason": "score key digest mismatch",
                "score_key_digest": run.score_key_digest,
                "baseline_score_key_digest": baseline.score_key_digest,
                "checksum_delta": None if run.checksum is None or baseline.checksum is None else run.checksum - baseline.checksum,
            })
            continue
        if len(run.first_score_keys) != len(baseline.first_score_keys):
            overall = "failed"
            comparisons.append({
                "backend": run.backend,
                "mode": run.mode,
                "status": "failed",
                "reason": "score key length mismatch",
            })
            continue
        diffs = [abs(a - b) for a, b in zip(run.first_score_keys, baseline.first_score_keys)]
        max_abs_diff = max(diffs, default=0)
        status = "pass" if max_abs_diff <= tolerance else "failed"
        if status != "pass":
            overall = "failed"
        comparisons.append({
            "backend": run.backend,
            "mode": run.mode,
            "status": status,
            "max_abs_first_row_diff": max_abs_diff,
            "checksum_delta": None if run.checksum is None or baseline.checksum is None else run.checksum - baseline.checksum,
            "score_key_digest": run.score_key_digest,
        })
    return {
        "status": overall,
        "baseline": {"backend": baseline.backend, "mode": baseline.mode},
        "tolerance": tolerance,
        "comparisons": comparisons,
    }


def write_reports(out_dir: Path, runs: List[BackendRun], comparison: Dict[str, object]) -> None:
    rows = [
        {
            "backend": run.backend,
            "mode": run.mode,
            "status": run.status,
            "return_code": run.return_code,
            "checksum": run.checksum,
            "score_key_digest": run.score_key_digest,
            "first_score_keys": run.first_score_keys,
            "log_path": run.log_path,
            "command": run.command,
            "env": run.env,
        }
        for run in runs
    ]
    payload = {"runs": rows, "comparison": comparison}
    (out_dir / "stream1_transformer_parity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    lines = [
        "# Stream1 Transformer Backend Parity",
        "",
        f"- status={comparison['status']}",
        f"- tolerance={comparison.get('tolerance')}",
        "",
        "| backend | mode | status | return_code | checksum | score_key_digest | first_score_keys |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for run in runs:
        first = ",".join(str(value) for value in run.first_score_keys)
        lines.append(
            f"| {run.backend} | {run.mode} | {run.status} | {run.return_code} | {run.checksum} | {run.score_key_digest} | `{first}` |"
        )
    lines.extend(["", "## Comparisons", "", "```json", json.dumps(comparison, indent=2, sort_keys=True), "```", ""])
    (out_dir / "stream1_transformer_parity.md").write_text("\n".join(lines), encoding="utf-8")


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weight-dir", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backends", default="pytorch,libtorch,native_cutlass")
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--tolerance", type=int, default=3072)
    parser.add_argument("--out-dir", type=Path, default=Path("test_results/stream1_transformer_parity"))
    parser.add_argument("--dry-run", action="store_true", help="Write planned backend invocations without executing them.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    backend_names = [part.strip() for part in args.backends.split(",") if part.strip()]
    unknown = [name for name in backend_names if name not in BACKENDS]
    if unknown:
        raise SystemExit(f"unknown backend(s): {','.join(unknown)}")

    runs: List[BackendRun] = []
    for backend in backend_names:
        mode = "eager"
        invocation = build_invocation(
            make_args(backend, mode, args.weight_dir, args.build_dir, args.device, args.batch, args.warmup, args.iters, args.out_dir)
        )
        if args.dry_run:
            print(f"stream1_transformer_parity_dry_run backend={backend} mode={mode}", flush=True)
            runs.append(BackendRun(
                backend=invocation.backend,
                mode=invocation.mode,
                command=list(invocation.command),
                env=dict(invocation.env),
                return_code=0,
                checksum=None,
                score_key_digest=None,
                first_score_keys=[],
                log_path="",
                status="dry_run",
            ))
        else:
            print(f"stream1_transformer_parity_run backend={backend} mode={mode}", flush=True)
            runs.append(run_backend(invocation, args.out_dir))

    comparison = (
        {
            "status": "dry_run",
            "reason": "backend invocations generated without execution",
            "tolerance": args.tolerance,
            "backends": [{"backend": run.backend, "mode": run.mode} for run in runs],
        }
        if args.dry_run
        else compare_runs(runs, args.tolerance)
    )
    write_reports(args.out_dir, runs, comparison)
    print(f"stream1_transformer_parity_status={comparison['status']}")
    print(f"stream1_transformer_parity_report={args.out_dir / 'stream1_transformer_parity.md'}")
    return 0 if comparison["status"] in {"pass", "dry_run"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
