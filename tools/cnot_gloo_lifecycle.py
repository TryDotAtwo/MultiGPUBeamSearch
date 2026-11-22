"""Validate run-local receipts for the installed CNOT Gloo lifecycle test."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence


EXPECTED_CASES = {
    "engine_mismatch": ("engine_consensus", "rejected"),
    "engine_mismatch_retry": ("healthy_retry", "passed"),
    "scorer_failure_rank_0": ("scorer_failure", "coordinated"),
    "scorer_failure_rank_0_retry": ("healthy_retry", "passed"),
    "scorer_failure_rank_1": ("scorer_failure", "coordinated"),
    "scorer_failure_rank_1_retry": ("healthy_retry", "passed"),
}
EXPECTED_ENGINE = {0: "auto", 1: "python"}
EXPECTED_REPORTS = {
    "engine_mismatch_retry": "healthy-after-engine-mismatch-0.json",
    "scorer_failure_rank_0_retry": "healthy-after-0-0.json",
    "scorer_failure_rank_1_retry": "healthy-after-1-0.json",
}


def validate_lifecycle(
    events_by_rank: Mapping[int, Sequence[Mapping[str, object]]],
    rank0_reports: Mapping[str, Mapping[str, object]],
) -> None:
    if set(events_by_rank) != {0, 1}:
        raise ValueError("lifecycle receipts must contain exactly ranks 0 and 1")
    if set(rank0_reports) != set(EXPECTED_REPORTS):
        raise ValueError("rank-zero replay reports are missing or unexpected")

    indexed: dict[int, dict[str, Mapping[str, object]]] = {}
    for rank in (0, 1):
        records = events_by_rank[rank]
        by_case: dict[str, Mapping[str, object]] = {}
        for event in records:
            if not isinstance(event, Mapping):
                raise ValueError(f"rank {rank} has a non-object lifecycle receipt")
            if event.get("rank") != rank:
                raise ValueError(f"lifecycle receipt rank mismatch for rank {rank}")
            case_id = event.get("case_id")
            if not isinstance(case_id, str) or case_id not in EXPECTED_CASES:
                raise ValueError(f"rank {rank} has an unknown lifecycle case")
            if case_id in by_case:
                raise ValueError(f"rank {rank} has duplicate lifecycle case {case_id}")
            by_case[case_id] = event
        if set(by_case) != set(EXPECTED_CASES):
            missing = sorted(set(EXPECTED_CASES) - set(by_case))
            raise ValueError(f"rank {rank} is missing lifecycle phases: {missing}")
        indexed[rank] = by_case

    for case_id, (phase, common_outcome) in EXPECTED_CASES.items():
        for rank in (0, 1):
            event = indexed[rank][case_id]
            if event.get("phase") != phase:
                raise ValueError(f"lifecycle phase mismatch for {case_id}, rank {rank}")
            if event.get("common_outcome") != common_outcome:
                raise ValueError(f"lifecycle outcome mismatch for {case_id}, rank {rank}")

        if case_id == "engine_mismatch":
            for rank in (0, 1):
                event = indexed[rank][case_id]
                if event.get("requested_engine") != EXPECTED_ENGINE[rank]:
                    raise ValueError(f"rank {rank} recorded the wrong local engine input")
                if event.get("error_type") != "ValueError" or event.get("expected_error") is not True:
                    raise ValueError(f"rank {rank} did not record the expected engine consensus error")
        elif case_id.startswith("scorer_failure_rank_") and not case_id.endswith("_retry"):
            failed_rank = int(case_id.rsplit("_", 1)[1])
            for rank in (0, 1):
                event = indexed[rank][case_id]
                if event.get("failed_rank") != failed_rank:
                    raise ValueError(f"failed rank mismatch for {case_id}, rank {rank}")
                if event.get("error_type") != "RuntimeError":
                    raise ValueError(f"rank {rank} recorded an unexpected scorer error type")
                if event.get("expected_phase") is not True or event.get("expected_rank") is not True:
                    raise ValueError(f"rank {rank} did not observe the shared scorer failure phase")
        elif case_id.endswith("_retry") and case_id.startswith("scorer_failure_rank_"):
            failed_rank = int(case_id.removeprefix("scorer_failure_rank_").removesuffix("_retry"))
            for rank in (0, 1):
                if indexed[rank][case_id].get("failed_rank") != failed_rank:
                    raise ValueError(f"retry failed-rank mismatch for {case_id}, rank {rank}")

    for case_id, report in rank0_reports.items():
        if report.get("status") != "solved":
            raise ValueError(f"rank-zero retry report is not solved: {case_id}")
        if report.get("bitwise_replay") is not True:
            raise ValueError(f"rank-zero retry report lacks bitwise replay: {case_id}")


def validate_directory(root: Path) -> None:
    events_by_rank: dict[int, list[dict[str, object]]] = {}
    for rank in (0, 1):
        path = root / f"verify-cnot-gloo-diagnostic-rank-{rank}.json"
        events_by_rank[rank] = json.loads(path.read_text(encoding="utf-8"))
    rank0_reports = {
        case_id: json.loads((root / filename).read_text(encoding="utf-8"))
        for case_id, filename in EXPECTED_REPORTS.items()
    }
    validate_lifecycle(events_by_rank, rank0_reports)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    validate_directory(args.run_dir)
    print("VERIFY_CNOT_GLOO_RANK_FAILURES_AND_REUSE_PASS")


if __name__ == "__main__":
    main()
