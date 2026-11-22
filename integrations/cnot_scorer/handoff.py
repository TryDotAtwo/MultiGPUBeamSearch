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
_HANDOFF_SCHEMA = "cnot103_multigpu_beam_handoff_v1"
_RECEIPT_SCHEMA = "cnot103_learning_curve_campaign_v1"
_SEARCH_SCHEMA = "cnot103_panel_search_v2"
_RESULT_SCHEMA = "cnot103_training_result_v1"
_MODEL = {
    "structural_blocks": 8,
    "parameters": 136_421_807,
    "feature_schema": "runtime_packet_layout_v25",
    "input_slots": 102,
}
_LOCKED_STATUSES = {"length_gain", "coverage_gain", "tie", "no_practical_gain"}


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
    accepted_updates: int
    checkpoint_sha256: str
    dataset_sha256: str
    panel_indices: tuple[int, ...]
    panel_ids: tuple[str, ...]
    panel_digest: str
    search: dict[str, Any]
    locked_test_status: str


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


def resolve_panel_v2_handoff(
    run_root: str | Path,
    learning_curve_receipt: str | Path,
) -> ResolvedPanelV2Handoff:
    """Verify the producer's private learning-curve receipt and return its test manifest.

    The returned manifest is the input to the existing TQPScorer. Scorer creation
    still runs the pinned Rust loader, which checks the full dataset and checkpoint.
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
    accepted_updates = handoff.get("accepted_updates")
    checkpoint_sha = handoff.get("checkpoint_sha256")
    dataset_sha = handoff.get("dataset_sha256")
    model = handoff.get("model")
    if (
        handoff.get("schema") != _HANDOFF_SCHEMA
        or handoff.get("consumer") != "MultiGPUBeamSearch"
        or not isinstance(source_revision, str)
        or _SHA1.fullmatch(source_revision) is None
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

    comparison = locked_test.get("comparison")
    status = handoff.get("locked_test_status")
    comparison_sha = handoff.get("locked_test_comparison_sha256")
    if (
        not isinstance(comparison, dict)
        or status not in _LOCKED_STATUSES
        or comparison.get("status") != status
        or not isinstance(comparison_sha, str)
        or _SHA256.fullmatch(comparison_sha) is None
        or _comparison_digest(comparison) != comparison_sha
        or type(locked_test.get("accepted_updates")) is not int
        or locked_test["accepted_updates"] != accepted_updates
        or locked_test.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise HandoffError("locked-test comparison is not bound to the selected checkpoint")

    return ResolvedPanelV2Handoff(
        search_manifest=search_path,
        training_manifest=train_manifest_path,
        training_result=train_result_path,
        search_manifest_sha256=search_sha,
        training_manifest_sha256=train_manifest_sha,
        training_result_sha256=train_result_sha,
        producer_source_revision=source_revision,
        accepted_updates=accepted_updates,
        checkpoint_sha256=checkpoint_sha,
        dataset_sha256=dataset_sha,
        panel_indices=tuple(indices),
        panel_ids=tuple(ids),
        panel_digest=panel_digest,
        search=dict(search),
        locked_test_status=status,
    )
