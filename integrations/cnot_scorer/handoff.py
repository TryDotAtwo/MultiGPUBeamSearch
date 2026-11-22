"""Resolve and verify a private CNOT panel-v2 handoff before scorer loading.

The handoff is a producer receipt, not a replacement for the Rust loader. The
Rust scorer still validates the frozen dataset, feature policy, checkpoint and
model before returning a usable scorer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any


_SHA1 = re.compile(r"[0-9a-f]{40}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_HANDOFF_SCHEMA = "cnot103_multigpu_beam_handoff_v2"
_RECEIPT_SCHEMA = "cnot103_learning_curve_campaign_v1"
_SEARCH_SCHEMA = "cnot103_panel_search_v2"
_RESULT_SCHEMA = "cnot103_training_result_v1"
_MODEL = {
    "structural_blocks": 8,
    "parameters": 136_421_807,
    "feature_schema": "runtime_packet_layout_v25",
    "input_slots": 102,
}
_SEARCH_CONFIG = {
    "global_width": 36,
    "shortlist": 4,
    "max_depth": 8,
    "max_expansions": 201,
}
_LOCKED_STATUSES = {"length_gain", "coverage_gain", "combined_gain", "tie", "no_practical_gain"}


class HandoffError(ValueError):
    """The private handoff or one of its pinned artifacts is inconsistent."""


@dataclass(frozen=True)
class ResolvedPanelV2Handoff:
    search_manifest: Path
    training_manifest: Path
    training_result: Path
    search_manifest_sha256: str
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
    locked_test_status: str
    dataset_path: Path
    panel_roots: tuple[dict[str, Any], ...]
    orientation: str
    artifact_path_policy: str = "same_run_root_mount"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inside_root(path: Path, root: Path, name: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError):
        raise HandoffError(f"{name} is missing or outside the private run root") from None
    if not resolved.is_file():
        raise HandoffError(f"{name} is not a regular file")
    return resolved


def _json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        raise HandoffError(f"{name} JSON is unreadable") from None
    if not isinstance(value, dict):
        raise HandoffError(f"{name} JSON is malformed")
    return value


def _artifact(root: Path, reference: object, name: str) -> tuple[Path, str]:
    if not isinstance(reference, dict):
        raise HandoffError(f"{name} reference is malformed")
    relative = reference.get("path")
    expected_sha = reference.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or "\x00" in relative
        or not isinstance(expected_sha, str)
        or _SHA256.fullmatch(expected_sha) is None
    ):
        raise HandoffError(f"{name} reference is malformed")
    posix = PurePosixPath(relative)
    if (
        posix.is_absolute()
        or posix.as_posix() != relative
        or not posix.parts
        or any(part in ("", ".", "..") for part in posix.parts)
        or ":" in posix.parts[0]
    ):
        raise HandoffError(f"{name} path is not a safe run-relative path")
    path = _inside_root(root.joinpath(*posix.parts), root, name)
    try:
        actual_sha = _sha256(path.read_bytes())
    except OSError:
        raise HandoffError(f"{name} cannot be read") from None
    if actual_sha != expected_sha:
        raise HandoffError(f"{name} SHA-256 differs from the handoff")
    return path, actual_sha


def _manifest_pin(root: Path, manifest_path: Path, value: object, expected_path: Path,
                  expected_sha: str, name: str) -> None:
    if not isinstance(value, dict):
        raise HandoffError(f"{name} search-manifest pin is malformed")
    pinned_path = value.get("path")
    pinned_sha = value.get("sha256")
    if not isinstance(pinned_path, str) or not pinned_path:
        raise HandoffError(f"{name} search-manifest path is missing")
    if not isinstance(pinned_sha, str) or _SHA256.fullmatch(pinned_sha) is None:
        raise HandoffError(f"{name} search-manifest SHA-256 is malformed")
    candidate = Path(pinned_path)
    if not candidate.is_absolute():
        candidate = manifest_path.parent / candidate
    resolved = _inside_root(candidate, root, name)
    if resolved != expected_path or pinned_sha != expected_sha:
        raise HandoffError(f"{name} search-manifest pin differs from the handoff")


def _panel_digest(panel: str, indices: list[int], ids: list[str]) -> str:
    canonical = json.dumps(
        {"panel": panel, "indices": indices, "ids": ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256(canonical)


def _comparison_digest(comparison: dict[str, Any]) -> str:
    try:
        canonical = json.dumps(
            comparison,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise HandoffError("locked-test comparison is not canonical JSON") from None
    return _sha256(canonical)


_ORIENTATION = "row_target_xor_row_control"


def validate_search_contract(search: object) -> int:
    """Validate the pinned v13 B/K/D/E preset and its frontier expansion bound."""
    if not isinstance(search, dict) or search != _SEARCH_CONFIG:
        raise HandoffError("search configuration differs from the supported CNOT preset")
    if any(type(search.get(name)) is not int for name in _SEARCH_CONFIG):
        raise HandoffError("search configuration fields must be exact integers")
    width = search["global_width"]
    shortlist = search["shortlist"]
    depth = search["max_depth"]
    capacities = [1]
    for _ in range(1, depth):
        capacities.append(min(width, shortlist * capacities[-1]))
    upper_bound = sum(capacities)
    if upper_bound != search["max_expansions"]:
        raise HandoffError("max_expansions differs from the v13 whole-frontier upper bound")
    return upper_bound


def _sealed_panel_roots(
    root: Path,
    training_manifest_path: Path,
    expected_dataset_sha: str,
    panel: str,
    indices: list[int],
    ids: list[str],
    expected_digest: str,
) -> tuple[Path, tuple[dict[str, Any], ...]]:
    training_manifest = _json_object(training_manifest_path, "training manifest")
    if training_manifest.get("schema") != "cnot103_training_run_v1":
        raise HandoffError("training manifest schema is unsupported")
    dataset_pin = training_manifest.get("dataset")
    if not isinstance(dataset_pin, dict):
        raise HandoffError("training manifest lacks a sealed dataset pin")
    path_value = dataset_pin.get("path")
    sha_value = dataset_pin.get("sha256")
    if (
        not isinstance(path_value, str) or not path_value
        or not isinstance(sha_value, str) or _SHA256.fullmatch(sha_value) is None
        or sha_value != expected_dataset_sha
    ):
        raise HandoffError("training manifest dataset identity differs from the handoff")
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = training_manifest_path.parent / candidate
    dataset_path = _inside_root(candidate, root, "sealed dataset")
    try:
        raw = dataset_path.read_bytes()
    except OSError:
        raise HandoffError("sealed dataset cannot be read") from None
    if _sha256(raw) != expected_dataset_sha:
        raise HandoffError("sealed dataset SHA-256 differs from the training manifest")
    try:
        dataset = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise HandoffError("sealed dataset JSON is unreadable") from None
    panels = dataset.get("panels") if isinstance(dataset, dict) else None
    records = panels.get(panel) if isinstance(panels, dict) else None
    if not isinstance(records, list):
        raise HandoffError("sealed dataset panel is missing")
    roots: list[dict[str, Any]] = []
    for position, (index, case_id) in enumerate(zip(indices, ids)):
        if index >= len(records):
            raise HandoffError("panel index is outside the sealed dataset")
        record = records[index]
        state = record.get("state") if isinstance(record, dict) else None
        if not isinstance(record, dict) or record.get("lineage_id") != case_id or not isinstance(state, dict):
            raise HandoffError("panel ID-to-index mapping differs from the sealed dataset")
        dimension = state.get("n")
        alphabet_id = state.get("alphabet_id")
        packed_rows = state.get("packed_rows")
        if (
            type(dimension) is not int or dimension < 2 or dimension > 256
            or alphabet_id != f"all_to_all_unit_cnot:{dimension}"
            or not isinstance(packed_rows, list) or len(packed_rows) != dimension
        ):
            raise HandoffError("sealed panel root identity or CNOT alphabet is invalid")
        orientation = record.get("orientation", state.get("orientation"))
        if orientation is not None and orientation != _ORIENTATION:
            raise HandoffError("sealed panel root orientation differs from the CNOT contract")
        words_per_row = (dimension + 63) // 64
        lanes: list[int] = []
        for row in packed_rows:
            if type(row) is int:
                value = row
            elif isinstance(row, str) and row.isdecimal():
                value = int(row)
            else:
                raise HandoffError("sealed panel packed row is malformed")
            if value < 0 or value.bit_length() > dimension:
                raise HandoffError("sealed panel packed row exceeds its declared width")
            lanes.extend((value >> (64 * word)) & 0xFFFFFFFFFFFFFFFF for word in range(words_per_row))
        roots.append({
            "panel_index": index,
            "case_id": case_id,
            "dimension": dimension,
            "alphabet_id": alphabet_id,
            "orientation": _ORIENTATION,
            "start_packed_rows": lanes,
        })
    if _panel_digest(panel, indices, ids) != expected_digest:
        raise HandoffError("sealed panel digest differs from its ordered case identities")
    return dataset_path, tuple(roots)


def _validate_locked_comparison(
    comparison: object,
    *,
    status: object,
    panel_indices: list[int],
    panel_ids: list[str],
    panel_digest: str,
    dataset_sha: str,
    source_revision: str,
    search_sha: str,
    training_manifest_sha: str,
    training_result_sha: str,
    checkpoint_sha: str,
) -> dict[str, Any]:
    if not isinstance(comparison, dict):
        raise HandoffError("locked-test comparison is malformed")
    if comparison.get("status") != status or comparison.get("panel") != "test":
        raise HandoffError("locked-test comparison status or split differs")
    if comparison.get("baseline_head") != "hamming" or comparison.get("candidate_head") != "qhat":
        raise HandoffError("locked-test comparison scorer differs from the accepted protocol")
    if comparison.get("case_count") != len(panel_indices):
        raise HandoffError("locked-test comparison case count differs from the frozen panel")
    for name in ("baseline_metrics", "candidate_metrics"):
        metrics = comparison.get(name)
        if (
            not isinstance(metrics, dict)
            or metrics.get("panel_indices") != panel_indices
            or metrics.get("panel_ids") != panel_ids
            or metrics.get("panel_digest") != panel_digest
            or metrics.get("case_count") != len(panel_indices)
        ):
            raise HandoffError(f"locked-test {name} panel pins differ from the frozen cohort")
        pins = metrics.get("receipt_pins")
        if not isinstance(pins, dict):
            raise HandoffError(f"locked-test {name} lacks receipt pins")
        if pins.get("dataset_sha256") != dataset_sha:
            raise HandoffError(f"locked-test {name} dataset pin differs")
        if name == "baseline_metrics":
            if (
                pins.get("source_revision") != source_revision
                or pins.get("runtime_contract") != "native_device_search_v13"
                or pins.get("training_manifest_sha256") is not None
                or pins.get("training_result_sha256") is not None
                or pins.get("checkpoint_sha256") is not None
                or not isinstance(pins.get("search_manifest_sha256"), str)
                or _SHA256.fullmatch(pins["search_manifest_sha256"]) is None
            ):
                raise HandoffError("locked-test Hamming baseline pins differ from the producer run")
        elif (
            pins.get("search_manifest_sha256") != search_sha
            or pins.get("training_manifest_sha256") != training_manifest_sha
            or pins.get("training_result_sha256") != training_result_sha
            or pins.get("checkpoint_sha256") != checkpoint_sha
        ):
            raise HandoffError("locked-test candidate pins differ from the selected checkpoint")
    return comparison


def resolve_panel_v2_handoff(
    run_root: str | Path,
    learning_curve_receipt: str | Path,
) -> ResolvedPanelV2Handoff:
    """Verify the producer's private learning-curve receipt and return its test manifest.

    Absolute paths embedded in the already-hashed search manifest are accepted only
    when they resolve inside this same private run-root mount. Relocating that run
    without rewriting the hashed manifest is rejected. The resolved ordered roots
    are loaded from the pinned sealed dataset before a scorer may be constructed.
    """
    try:
        root = Path(run_root).resolve(strict=True)
    except (OSError, TypeError):
        raise HandoffError("private run root is unavailable") from None
    if not root.is_dir():
        raise HandoffError("private run root is not a directory")
    receipt_candidate = Path(learning_curve_receipt)
    if not receipt_candidate.is_absolute():
        receipt_candidate = root / receipt_candidate
    receipt_path = _inside_root(receipt_candidate, root, "learning-curve receipt")
    receipt = _json_object(receipt_path, "learning-curve receipt")
    if receipt.get("schema") != _RECEIPT_SCHEMA:
        raise HandoffError("learning-curve receipt schema is unsupported")

    handoff = receipt.get("multi_gpu_beam_handoff")
    locked_test = receipt.get("locked_test")
    if not isinstance(handoff, dict) or not isinstance(locked_test, dict):
        raise HandoffError("learning-curve receipt lacks a locked-test handoff")
    if locked_test.get("consumer_handoff") != handoff:
        raise HandoffError("top-level and locked-test handoffs differ")
    if locked_test.get("status") != "evaluated":
        raise HandoffError("locked-test evaluation was not completed")

    source_revision = handoff.get("producer_source_revision")
    producer_emitter_sha256 = handoff.get("producer_emitter_sha256")
    accepted_updates = handoff.get("accepted_updates")
    checkpoint_sha = handoff.get("checkpoint_sha256")
    dataset_sha = handoff.get("dataset_sha256")
    search = handoff.get("search")
    model = handoff.get("model")
    if receipt.get("source_revision") != source_revision:
        raise HandoffError("learning-curve receipt source revision differs from the handoff")
    if receipt.get("producer_emitter_sha256") != producer_emitter_sha256:
        raise HandoffError("learning-curve receipt producer emitter pin differs from the handoff")
    if (
        handoff.get("schema") != _HANDOFF_SCHEMA
        or handoff.get("consumer") != "MultiGPUBeamSearch"
        or not isinstance(source_revision, str)
        or _SHA1.fullmatch(source_revision) is None
        or not isinstance(producer_emitter_sha256, str)
        or _SHA256.fullmatch(producer_emitter_sha256) is None
        or type(accepted_updates) is not int
        or accepted_updates < 1
        or not isinstance(checkpoint_sha, str)
        or _SHA256.fullmatch(checkpoint_sha) is None
        or not isinstance(dataset_sha, str)
        or _SHA256.fullmatch(dataset_sha) is None
        or model != _MODEL
    ):
        raise HandoffError("handoff source, model or training identity is invalid")

    train_manifest_path, train_manifest_sha = _artifact(
        root, handoff.get("training_manifest"), "training manifest"
    )
    train_result_path, train_result_sha = _artifact(
        root, handoff.get("training_result"), "training result"
    )
    search_path, search_sha = _artifact(
        root, handoff.get("search_manifest"), "panel search manifest"
    )

    if (
        receipt.get("dataset_sha256") != dataset_sha
        or receipt.get("search") != search
    ):
        raise HandoffError("top-level receipt dataset or search identity differs from the handoff")
    selected = receipt.get("selected")
    if not isinstance(selected, dict) or selected.get("accepted_updates") != accepted_updates:
        raise HandoffError("selected accepted-update identity differs from the handoff")
    if selected.get("checkpoint_sha256") != checkpoint_sha:
        raise HandoffError("selected checkpoint differs from the handoff")

    result = _json_object(train_result_path, "training result")
    progress = result.get("end_progress")
    checkpoint = result.get("checkpoint")
    if (
        result.get("schema") != _RESULT_SCHEMA
        or not isinstance(progress, dict)
        or type(progress.get("accepted")) is not int
        or progress.get("accepted") != accepted_updates
        or not isinstance(checkpoint, dict)
        or checkpoint.get("sha256") != checkpoint_sha
        or result.get("model_parameters") != _MODEL["parameters"]
        or result.get("feature_model_inputs") != _MODEL["input_slots"]
        or result.get("dataset_sha256") != dataset_sha
    ):
        raise HandoffError("training result differs from the selected handoff")

    search_manifest = _json_object(search_path, "panel search manifest")
    panel = handoff.get("panel")
    indices = handoff.get("panel_indices")
    ids = handoff.get("panel_ids")
    panel_digest = handoff.get("panel_digest")
    search = handoff.get("search")
    if (
        search_manifest.get("schema") != _SEARCH_SCHEMA
        or panel != "test"
        or not isinstance(indices, list)
        or not indices
        or any(type(index) is not int or index < 0 for index in indices)
        or len(indices) != len(set(indices))
        or not isinstance(ids, list)
        or len(ids) != len(indices)
        or any(not isinstance(case_id, str) or not case_id.strip() for case_id in ids)
        or len(ids) != len(set(ids))
        or not isinstance(panel_digest, str)
        or panel_digest != _panel_digest(panel, indices, ids)
        or not isinstance(search, dict)
        or handoff.get("head") != "qhat"
        or search_manifest.get("panel") != panel
        or search_manifest.get("panel_indices") != indices
        or search_manifest.get("panel_ids") != ids
        or search_manifest.get("panel_digest") != panel_digest
        or search_manifest.get("search") != search
        or search_manifest.get("head") != "qhat"
    ):
        raise HandoffError("panel, search configuration or scorer differs from the handoff")

    validate_search_contract(search)
    _manifest_pin(
        root,
        search_path,
        search_manifest.get("training_manifest"),
        train_manifest_path,
        train_manifest_sha,
        "training manifest",
    )
    _manifest_pin(
        root,
        search_path,
        search_manifest.get("training_result"),
        train_result_path,
        train_result_sha,
        "training result",
    )

    if receipt.get("locked_test_panel_digest") != panel_digest:
        raise HandoffError("top-level locked-test panel digest differs from the handoff")
    dataset_path, panel_roots = _sealed_panel_roots(
        root,
        train_manifest_path,
        dataset_sha,
        panel,
        indices,
        ids,
        panel_digest,
    )
    comparison = locked_test.get("comparison")
    status = handoff.get("locked_test_status")
    comparison_sha = handoff.get("locked_test_comparison_sha256")
    if (
        status not in _LOCKED_STATUSES
        or not isinstance(comparison_sha, str)
        or _SHA256.fullmatch(comparison_sha) is None
        or _comparison_digest(comparison) != comparison_sha
        or receipt.get("locked_test_comparison") != comparison
        or type(locked_test.get("accepted_updates")) is not int
        or locked_test["accepted_updates"] != accepted_updates
        or locked_test.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise HandoffError("locked-test comparison is not bound to the selected checkpoint")
    _validate_locked_comparison(
        comparison,
        status=status,
        panel_indices=indices,
        panel_ids=ids,
        panel_digest=panel_digest,
        dataset_sha=dataset_sha,
        source_revision=source_revision,
        search_sha=search_sha,
        training_manifest_sha=train_manifest_sha,
        training_result_sha=train_result_sha,
        checkpoint_sha=checkpoint_sha,
    )

    return ResolvedPanelV2Handoff(
        search_manifest=search_path,
        training_manifest=train_manifest_path,
        training_result=train_result_path,
        search_manifest_sha256=search_sha,
        training_manifest_sha256=train_manifest_sha,
        training_result_sha256=train_result_sha,
        producer_source_revision=source_revision,
        producer_emitter_sha256=producer_emitter_sha256,
        accepted_updates=accepted_updates,
        checkpoint_sha256=checkpoint_sha,
        dataset_sha256=dataset_sha,
        panel_indices=tuple(indices),
        panel_ids=tuple(ids),
        panel_digest=panel_digest,
        search=dict(search),
        locked_test_status=status,
        dataset_path=dataset_path,
        panel_roots=panel_roots,
        orientation=_ORIENTATION,
    )
