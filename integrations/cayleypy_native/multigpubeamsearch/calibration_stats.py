"""Statistical selection for inference-first beam calibration.

Measurements must use identical parent counts, graph, models and rank cohort.
Memory admission remains owned by the native planner. This selector cannot
change capacity margins, requested beam, algorithm, or communicator membership.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics
from typing import Sequence


@dataclass(frozen=True)
class Measurement:
    profile: str
    parents: int
    seconds_by_rank: tuple[float, ...]
    correctness_passed: bool
    memory_admitted: bool
    throttled: bool = False

    @property
    def seconds_per_parent(self) -> float:
        if type(self.parents) is not int or self.parents <= 0:
            raise ValueError("parent count must be a positive integer")
        if not self.seconds_by_rank or any(
            not math.isfinite(x) or x <= 0 for x in self.seconds_by_rank
        ):
            raise ValueError("all rank timings must be finite and positive")
        # Global parents / slowest rank wall time, never sum per-rank rates.
        return max(self.seconds_by_rank) / self.parents


@dataclass(frozen=True)
class Estimate:
    profile: str
    median: float
    low: float
    high: float
    relative_mad: float
    repeats: int


def estimate(profile: str, samples: Sequence[Measurement], *, min_repeats=5,
             bootstrap_repeats=2000, seed=0, max_relative_mad=0.10) -> Estimate:
    if min_repeats < 3 or bootstrap_repeats < 100:
        raise ValueError("insufficient statistical repeats")
    selected = [x for x in samples if x.profile == profile]
    if len(selected) < min_repeats:
        raise ValueError("insufficient measured repeats")
    if any(not x.correctness_passed or not x.memory_admitted or x.throttled for x in selected):
        raise ValueError("reject incorrect, unadmitted or throttled profile")
    if len({(x.parents, len(x.seconds_by_rank)) for x in selected}) != 1:
        raise ValueError("mixed workload or rank cohort")
    values = [x.seconds_per_parent for x in selected]
    median = statistics.median(values)
    mad = statistics.median(abs(x - median) for x in values) / median
    if mad > max_relative_mad:
        raise ValueError("unstable timing samples")
    rng = random.Random(seed)
    boot = sorted(statistics.median(rng.choices(values, k=len(values)))
                  for _ in range(bootstrap_repeats))
    return Estimate(profile, median, boot[int(0.025 * len(boot))],
                    boot[min(len(boot)-1, int(0.975 * len(boot)))], mad, len(values))


def select(samples: Sequence[Measurement], *, baseline: str,
           min_improvement=0.02, **estimate_options) -> tuple[Estimate, dict[str, str]]:
    """Keep baseline unless an admitted candidate has a clear stable gain.

    Caller interleaves randomized measurement rounds after warmup; no warmup
    records belong in samples. A failing baseline aborts calibration.
    """
    if not 0 <= min_improvement < 1:
        raise ValueError("invalid improvement threshold")
    cohorts = {(x.parents, len(x.seconds_by_rank)) for x in samples}
    if len(cohorts) != 1:
        raise ValueError("all candidates must share workload and rank cohort")
    chosen = estimate(baseline, samples, **estimate_options)
    rejected = {}
    valid = []
    for profile in sorted({x.profile for x in samples} - {baseline}):
        try:
            valid.append(estimate(profile, samples, **estimate_options))
        except ValueError as error:
            rejected[profile] = str(error)
    for candidate in sorted(valid, key=lambda x: x.median):
        if candidate.high < chosen.low * (1-min_improvement):
            chosen = candidate
            break
        rejected[candidate.profile] = "no statistically separated material gain"
    return chosen, rejected


def pipeline_overhead(inference: Estimate, pipeline: Estimate) -> dict[str, float]:
    """Both estimates must represent the same full-frontier parent workload."""
    return {"inference_seconds_per_parent": inference.median,
            "pipeline_seconds_per_parent": pipeline.median,
            "relative_slowdown": pipeline.median / inference.median - 1,
            "overhead_seconds_per_parent": pipeline.median - inference.median}
