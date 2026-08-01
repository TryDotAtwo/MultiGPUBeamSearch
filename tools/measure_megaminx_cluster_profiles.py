"""Convert repeated exact profile sweeps into deterministic winner records."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping, Sequence


def select_winners(candidates: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    accepted: list[dict[str, object]] = []
    for source in candidates:
        item = dict(source)
        timings = item.get("timings_ms")
        if item.get("status") not in {"measured", "bounded_from_measured"}:
            continue
        if item.get("correctness_digest") != item.get("expected_correctness_digest"):
            continue
        if isinstance(timings, (str, bytes)) or not isinstance(timings, Sequence) or len(timings) < 3:
            continue
        try:
            samples = [float(value) for value in timings]
        except (TypeError, ValueError):
            continue
        if any(not math.isfinite(value) or value <= 0 for value in samples):
            continue
        power = item.get("beam_power")
        if isinstance(power, bool) or not isinstance(power, int) or power <= 0:
            continue
        item["median_ms"] = median(samples)
        accepted.append(item)
    winners: dict[tuple[str, int, int, int, str, str, int], dict[str, object]] = {}
    for item in accepted:
        hardware = item.get("hardware")
        if not isinstance(hardware, Mapping):
            continue
        try:
            key = (str(hardware["gpu_family"]), int(hardware["vram_mib"]), int(hardware["sm"]), int(hardware["world_size"]), str(item["backend"]), str(item["model_class"]), int(item["beam_power"]))
        except (KeyError, TypeError, ValueError):
            continue
        rank = (float(item["median_ms"]), str(item.get("evidence_id", "")), json.dumps(item.get("runtime"), sort_keys=True))
        prior = winners.get(key)
        if prior is None or rank < (float(prior["median_ms"]), str(prior.get("evidence_id", "")), json.dumps(prior.get("runtime"), sort_keys=True)):
            winners[key] = item
    return [winners[key] for key in sorted(winners)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    candidates = [json.loads(line) for line in args.input.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    args.output.write_text(json.dumps({"schema_version": 1, "winners": select_winners(candidates)}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
