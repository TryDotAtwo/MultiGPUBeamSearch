"""Small immutable runtime snapshot for a fully resolved CNOT panel handoff."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


PREFLIGHT_SCHEMA = "cnot103_mgbs_handoff_preflight_v1"
_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_ORIENTATION = "row_target_xor_row_control"


class PanelSnapshotError(ValueError):
    """The resolved panel snapshot does not match its immutable input pin."""


@dataclass(frozen=True)
class PanelV2Snapshot:
    search_manifest: Path
    search_manifest_sha256: str
    baseline_search_manifest_sha256: str
    training_manifest_sha256: str
    training_result_sha256: str
    producer_source_revision: str
    producer_emitter_sha256: str
    accepted_updates: int
    checkpoint_sha256: str
    dataset_sha256: str
    panel_indices: tuple[int, ...]
    panel_ids: tuple[str, ...]
    panel_digest: str
    search: dict[str, Any]
    expansion_upper_bound: int
    locked_test_status: str
    locked_test_comparison_sha256: str
    panel_roots: tuple[dict[str, Any], ...]
    orientation: str


def _json_digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha(value: object, pattern: re.Pattern[str], name: str) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise PanelSnapshotError(f"{name} is not a valid pinned digest")
    return value


def load_panel_v2_snapshot(
    path: Path,
    run_root: Path,
    expected_snapshot_sha256: str,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_search_manifest_sha256: str | None = None,
) -> PanelV2Snapshot:
    _sha(expected_snapshot_sha256, _SHA256, "snapshot SHA-256")
    try:
        raw = Path(path).read_bytes()
    except OSError as error:
        raise PanelSnapshotError("resolved panel snapshot cannot be read") from error
    if hashlib.sha256(raw).hexdigest() != expected_snapshot_sha256:
        raise PanelSnapshotError("resolved panel snapshot changed after preflight")
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise PanelSnapshotError("resolved panel snapshot JSON is invalid") from error
    if not isinstance(payload, dict) or payload.get("schema") != PREFLIGHT_SCHEMA or payload.get("status") != "passed":
        raise PanelSnapshotError("resolved panel snapshot schema or status is invalid")

    for key in (
        "producer_emitter_sha256",
        "checkpoint_sha256",
        "dataset_sha256",
        "training_manifest_sha256",
        "training_result_sha256",
        "search_manifest_sha256",
        "baseline_search_manifest_sha256",
        "locked_test_comparison_sha256",
        "panel_digest",
    ):
        _sha(payload.get(key), _SHA256, key)
    source_revision = _sha(payload.get("producer_source_revision"), _SHA1, "producer source revision")
    if expected_checkpoint_sha256 is not None and payload["checkpoint_sha256"] != expected_checkpoint_sha256:
        raise PanelSnapshotError("checkpoint pin differs from the resolved panel snapshot")
    if expected_search_manifest_sha256 is not None and payload["search_manifest_sha256"] != expected_search_manifest_sha256:
        raise PanelSnapshotError("search manifest pin differs from the resolved panel snapshot")

    root = Path(run_root).resolve(strict=True)
    relative_value = payload.get("search_manifest_relative_path")
    if not isinstance(relative_value, str) or not relative_value:
        raise PanelSnapshotError("search manifest path is missing")
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or ".." in relative.parts or "\\" in relative_value or ":" in relative_value:
        raise PanelSnapshotError("search manifest path is not safely run-relative")
    try:
        search_manifest = root.joinpath(*relative.parts).resolve(strict=True)
        search_manifest.relative_to(root)
    except (OSError, ValueError) as error:
        raise PanelSnapshotError("search manifest escaped or is absent from the private run root") from error

    if payload.get("panel") != "test" or payload.get("artifact_path_policy") != "same_run_root_mount":
        raise PanelSnapshotError("resolved panel selection or artifact policy is invalid")
    search = payload.get("search")
    if (
        not isinstance(search, dict)
        or type(search.get("global_width")) is not int or search["global_width"] < 1
        or type(search.get("shortlist")) is not int or search["shortlist"] < 1
        or type(search.get("max_depth")) is not int or search["max_depth"] < 1
        or type(search.get("max_expansions")) is not int or search["max_expansions"] < 1
    ):
        raise PanelSnapshotError("resolved search configuration is invalid")
    expansion_upper_bound = payload.get("expansion_upper_bound")
    if type(expansion_upper_bound) is not int or expansion_upper_bound != search["max_expansions"]:
        raise PanelSnapshotError("resolved search expansion bound differs from the search configuration")

    indices = payload.get("panel_indices")
    ids = payload.get("panel_ids")
    roots = payload.get("panel_roots")
    root_pins = payload.get("root_pins")
    if (
        not isinstance(indices, list) or not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or len(indices) != len(set(indices))
        or not isinstance(ids, list) or len(ids) != len(indices)
        or any(not isinstance(case_id, str) or not case_id.strip() for case_id in ids)
        or len(ids) != len(set(ids))
        or not isinstance(roots, list) or len(roots) != len(indices)
        or not isinstance(root_pins, list) or len(root_pins) != len(indices)
    ):
        raise PanelSnapshotError("resolved panel identities or roots are invalid")
    if _json_digest({"panel": "test", "indices": indices, "ids": ids}) != payload["panel_digest"]:
        raise PanelSnapshotError("panel digest differs from the resolved panel identities")

    checked_roots = []
    for index, case_id, root_row, pin in zip(indices, ids, roots, root_pins):
        if not isinstance(root_row, dict) or not isinstance(pin, dict):
            raise PanelSnapshotError("resolved panel root row is invalid")
        dimension = root_row.get("dimension")
        alphabet_id = root_row.get("alphabet_id")
        orientation = root_row.get("orientation")
        lanes = root_row.get("start_packed_rows")
        if (
            root_row.get("panel_index") != index
            or root_row.get("case_id") != case_id
            or type(dimension) is not int or dimension < 2 or dimension > 256
            or alphabet_id != f"all_to_all_unit_cnot:{dimension}"
            or orientation != _ORIENTATION
            or not isinstance(lanes, list) or len(lanes) != dimension
            or any(type(lane) is not int for lane in lanes)
            or pin.get("panel_index") != index
            or pin.get("case_id") != case_id
            or pin.get("dimension") != dimension
            or pin.get("alphabet_id") != alphabet_id
            or pin.get("orientation") != orientation
            or pin.get("start_packed_rows_sha256") != _json_digest(lanes)
        ):
            raise PanelSnapshotError("resolved packed root differs from its checked panel pin")
        checked_roots.append(dict(root_row))

    if not isinstance(payload.get("locked_test_status"), str) or not payload["locked_test_status"]:
        raise PanelSnapshotError("locked-test result identity is missing")
    accepted_updates = payload.get("accepted_updates")
    if type(accepted_updates) is not int or accepted_updates < 1:
        raise PanelSnapshotError("accepted-update count is invalid")

    return PanelV2Snapshot(
        search_manifest=search_manifest,
        search_manifest_sha256=payload["search_manifest_sha256"],
        baseline_search_manifest_sha256=payload["baseline_search_manifest_sha256"],
        training_manifest_sha256=payload["training_manifest_sha256"],
        training_result_sha256=payload["training_result_sha256"],
        producer_source_revision=source_revision,
        producer_emitter_sha256=payload["producer_emitter_sha256"],
        accepted_updates=accepted_updates,
        checkpoint_sha256=payload["checkpoint_sha256"],
        dataset_sha256=payload["dataset_sha256"],
        panel_indices=tuple(indices),
        panel_ids=tuple(ids),
        panel_digest=payload["panel_digest"],
        search=dict(search),
        expansion_upper_bound=expansion_upper_bound,
        locked_test_status=payload["locked_test_status"],
        locked_test_comparison_sha256=payload["locked_test_comparison_sha256"],
        panel_roots=tuple(checked_roots),
        orientation=_ORIENTATION,
    )
