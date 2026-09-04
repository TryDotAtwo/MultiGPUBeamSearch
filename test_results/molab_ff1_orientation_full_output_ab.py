"""Strict foreground A/B gate for the corrected SM120 FF1 orientations.

Run only inside a visible durable Molab notebook cell after both binaries were
built for sm_120a.  This script deliberately does not detach work: each process
finishes before the next one starts, keeping marimo's execution slot alive.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone


LINE = re.compile(
    r"orientation=(?P<orientation>\S+).*?"
    r"us=(?P<us>[0-9.]+).*?"
    r"useful_tflops=(?P<tflops>[0-9.]+).*?"
    r"d_hash=(?P<d_hash>[0-9]+).*?"
    r"logical_sfd_hash=(?P<sfd_hash>[0-9]+)"
)


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile95(values: list[float]) -> float:
    # Nearest-rank is deterministic and conservative for the seven-run gate.
    ordered = sorted(values)
    return ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]


def main() -> int:
    if len(sys.argv) not in (1, 3):
        raise SystemExit("usage: molab_ff1_orientation_full_output_ab.py [ROWS_BIN TRANSPOSED_BIN]")
    binaries = {
        "rows": pathlib.Path(sys.argv[1] if len(sys.argv) == 3 else
                             "/tmp/stream1_transformer_sm120_nvfp4_cutlass_ff1_no_store_rows"),
        "transposed": pathlib.Path(sys.argv[2] if len(sys.argv) == 3 else
                                   "/tmp/stream1_transformer_sm120_nvfp4_cutlass_ff1_no_store_transposed"),
    }
    for path in binaries.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    report: dict[str, object] = {
        "schema": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "complete E2M1 FF1 payload plus logical SFD; Molab SM120 only",
        "shape": {"m": 51072, "n": 1024, "k": 256},
        "iterations_per_process": 100,
        "alternating_pairs": 7,
        "binary_sha256": {name: sha256(path) for name, path in binaries.items()},
        "runs": [],
    }
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
         "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    )
    report["gpu"] = gpu.stdout.strip()
    runs: list[dict[str, object]] = []
    by_variant: dict[str, list[float]] = {name: [] for name in binaries}
    hashes: dict[str, set[tuple[str, str]]] = {name: set() for name in binaries}

    for pair in range(7):
        order = list(binaries) if pair % 2 == 0 else list(reversed(binaries))
        for name in order:
            completed = subprocess.run(
                [str(binaries[name]), "51072", "100"],
                check=False, capture_output=True, text=True, timeout=120,
            )
            match = LINE.search(completed.stdout)
            item: dict[str, object] = {
                "pair": pair,
                "variant": name,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
            runs.append(item)
            if completed.returncode or match is None:
                report["runs"] = runs
                raise RuntimeError(f"{name} failed or emitted an unparsable result: {item}")
            latency = float(match.group("us"))
            by_variant[name].append(latency)
            hashes[name].add((match.group("d_hash"), match.group("sfd_hash")))

    if any(len(items) != 1 for items in hashes.values()):
        raise RuntimeError(f"nondeterministic output hashes: {hashes}")
    if next(iter(hashes["rows"])) != next(iter(hashes["transposed"])):
        raise RuntimeError(f"orientation output mismatch: {hashes}")

    summary = {
        name: {
            "median_us": statistics.median(values),
            "p95_us": percentile95(values),
            "samples_us": values,
        }
        for name, values in by_variant.items()
    }
    report["runs"] = runs
    report["hashes"] = {name: list(next(iter(items))) for name, items in hashes.items()}
    report["summary"] = summary
    report["verdict"] = "pass"
    output = pathlib.Path("/tmp/molab_ff1_orientation_full_output_ab.json")
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
