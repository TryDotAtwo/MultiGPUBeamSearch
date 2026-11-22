#!/usr/bin/env python3
"""Resolve the private producer receipt before building or invoking its scorer."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any


SUMMARY_SCHEMA = "cnot103_mgbs_handoff_preflight_v1"
_VALID_STATUSES = {"solved", "depth_limit", "resource_limit", "exhausted"}


def _load_resolver(source_root: Path):
    path = source_root / "integrations" / "cnot_scorer" / "handoff.py"
    spec = importlib.util.spec_from_file_location("mgbs_cnot_handoff_preflight", path)
    if spec is None or spec.loader is None:
        raise ValueError("handoff resolver is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_summary(resolved: Any, run_root: Path) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    comparison = resolved.locked_test_comparison
    candidate = comparison.get("candidate_metrics") if isinstance(comparison, dict) else None
    baseline = comparison.get("baseline_metrics") if isinstance(comparison, dict) else None
    candidate_cases = candidate.get("cases") if isinstance(candidate, dict) else None
    if not isinstance(candidate_cases, dict):
        raise ValueError("locked-test candidate cases are missing")
    if set(candidate_cases) != set(resolved.panel_ids):
        raise ValueError("locked-test candidate cases differ from the ordered panel")

    roots = {row["case_id"]: row for row in resolved.panel_roots}
    if set(roots) != set(resolved.panel_ids):
        raise ValueError("sealed panel roots differ from the ordered panel")
    cases = []
    root_pins = []
    panel_roots = []
    for index, case_id in zip(resolved.panel_indices, resolved.panel_ids):
        root_row = roots[case_id]
        candidate_row = candidate_cases[case_id]
        lanes = root_row.get("start_packed_rows")
        if not isinstance(lanes, list):
            raise ValueError("sealed panel root has no packed matrix")
        status = candidate_row.get("status")
        cost = candidate_row.get("physical_cost")
        if status not in _VALID_STATUSES:
            raise ValueError("locked-test candidate status is invalid")
        if status == "solved":
            if type(cost) is not int or cost < 0:
                raise ValueError("locked-test solved candidate cost is invalid")
        elif cost is not None:
            raise ValueError("locked-test unsolved candidate unexpectedly has a cost")
        if (
            root_row.get("panel_index") != index
            or root_row.get("dimension") != candidate_row.get("dimension")
            or root_row.get("alphabet_id") != candidate_row.get("alphabet_id")
            or candidate_row.get("panel_index") != index
        ):
            raise ValueError("candidate row does not match the sealed root identity")
        row_digest = _json_digest(lanes)
        cases.append({
            "panel_index": index,
            "case_id": case_id,
            "dimension": root_row["dimension"],
            "alphabet_id": root_row["alphabet_id"],
            "status": status,
            "physical_cost": cost,
            "start_packed_rows_sha256": row_digest,
        })
        root_pins.append({
            "panel_index": index,
            "case_id": case_id,
            "dimension": root_row["dimension"],
            "alphabet_id": root_row["alphabet_id"],
            "orientation": root_row["orientation"],
            "start_packed_rows_sha256": row_digest,
        })
        panel_roots.append({
            "panel_index": index,
            "case_id": case_id,
            "dimension": root_row["dimension"],
            "alphabet_id": root_row["alphabet_id"],
            "orientation": root_row["orientation"],
            "start_packed_rows": list(lanes),
        })

    if not isinstance(baseline, dict):
        raise ValueError("locked-test baseline metrics are missing")
    solved_cases = [row for row in cases if row["status"] == "solved"]
    candidate_solved_count = candidate.get("solved_count")
    candidate_total_cost = candidate.get("total_physical_cost")
    if (
        type(candidate_solved_count) is not int
        or candidate_solved_count != len(solved_cases)
        or type(candidate_total_cost) is not int
        or candidate_total_cost != sum(row["physical_cost"] for row in solved_cases)
    ):
        raise ValueError("locked-test candidate aggregate differs from its ordered cases")
    search_relative = resolved.search_manifest.resolve(strict=True).relative_to(root).as_posix()
    return {
        "schema": SUMMARY_SCHEMA,
        "status": "passed",
        "producer_source_revision": resolved.producer_source_revision,
        "producer_emitter_sha256": resolved.producer_emitter_sha256,
        "accepted_updates": resolved.accepted_updates,
        "checkpoint_sha256": resolved.checkpoint_sha256,
        "dataset_sha256": resolved.dataset_sha256,
        "training_manifest_sha256": resolved.training_manifest_sha256,
        "training_result_sha256": resolved.training_result_sha256,
        "search_manifest_sha256": resolved.search_manifest_sha256,
        "baseline_search_manifest_sha256": resolved.baseline_search_manifest_sha256,
        "search_manifest_relative_path": search_relative,
        "locked_test_status": resolved.locked_test_status,
        "locked_test_comparison_sha256": resolved.locked_test_comparison_sha256,
        "panel": "test",
        "panel_indices": list(resolved.panel_indices),
        "panel_ids": list(resolved.panel_ids),
        "panel_digest": resolved.panel_digest,
        "search": dict(resolved.search),
        "expansion_upper_bound": resolved.expansion_upper_bound,
        "orientation": resolved.orientation,
        "panel_roots": panel_roots,
        "artifact_path_policy": resolved.artifact_path_policy,
        "baseline_solved_count": baseline.get("solved_count"),
        "baseline_total_physical_cost": baseline.get("total_physical_cost"),
        "candidate_solved_count": candidate.get("solved_count"),
        "candidate_total_physical_cost": candidate.get("total_physical_cost"),
        "root_pins": root_pins,
        "candidate_cases": cases,
    }


def resolve_handoff(run_root: Path, source_root: Path) -> dict[str, Any]:
    root = Path(run_root).resolve(strict=True)
    resolver = _load_resolver(Path(source_root).resolve(strict=True))
    resolved = resolver.resolve_panel_v2_handoff(
        root, "reports/learning-curve.private.json"
    )
    return build_summary(resolved, root)


def write_summary(summary: dict[str, Any], output: Path) -> None:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        summary = resolve_handoff(args.run_root, args.source_root)
        write_summary(summary, args.output)
    except Exception:
        print("CNOT handoff preflight rejected", file=sys.stderr)
        return 2
    print("CNOT_HANDOFF_PREFLIGHT=passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
