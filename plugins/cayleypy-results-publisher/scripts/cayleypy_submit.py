#!/usr/bin/env python3
"""Safe, dependency-free CayleyPy public result publisher."""

from __future__ import annotations

import copy
import csv
import gzip
import hashlib
import json
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

OFFICIAL_ENDPOINT_BASE: Final[str] = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev"
MAX_INPUT_BYTES: Final[int] = 128 * 1024 * 1024
MAX_ENVELOPES: Final[int] = 100_000
MAX_COMPRESSED_BYTES: Final[int] = 32 * 1024 * 1024
MAX_RAW_BYTES: Final[int] = 64 * 1024 * 1024
SOLUTION_COLUMNS: Final[tuple[str, ...]] = (
    "puzzle_id", "solution", "final_orientation", "search_mode",
    "collection_index", "collection_status", "solved_depth", "touch_depth",
    "reflected_source_solution", "searched_solution",
)


class ClientError(ValueError):
    def __init__(self, code: str, detail: str = "") -> None:
        self.code = code
        super().__init__(code if not detail else f"{code}: {detail}")


@dataclass(frozen=True)
class PublisherConfig:
    schema_version: int
    common: dict[str, Any]
    puzzle_contexts: dict[str, dict[str, Any]]
    solution_defaults: dict[str, str]


@dataclass(frozen=True)
class ArchivePart:
    index: int
    count: int
    version: int
    raw: bytes
    compressed: bytes


@dataclass(frozen=True)
class PreflightReport:
    schema_version: int
    envelope_count: int
    raw_bytes: int
    part_count: int
    endpoint_path: str


def _no_duplicate_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ClientError("INPUT_DUPLICATE_KEY", key)
        result[key] = value
    return result


def _parse_json(text: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_object)
    except ClientError:
        raise
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ClientError("INPUT_INVALID_JSON", str(exc)) from None


def canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ClientError("INPUT_NOT_CANONICAL", str(exc)) from None


def gzip_bytes(payload: bytes) -> bytes:
    return gzip.compress(payload, compresslevel=9, mtime=0)


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _uuid7() -> str:
    millis = int(time.time() * 1000)
    data = bytearray(millis.to_bytes(6, "big") + secrets.token_bytes(10))
    data[6] = (data[6] & 0x0F) | 0x70
    data[8] = (data[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(data)))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _require_version(value: Any) -> int:
    if type(value) is not int or value not in (1, 2):
        raise ClientError("INPUT_SCHEMA_VERSION")
    return value


def load_config(path: Path) -> PublisherConfig:
    value = _parse_json(_read_text(path))
    if not isinstance(value, dict):
        raise ClientError("CONFIG_OBJECT_REQUIRED")
    version = _require_version(value.get("schema_version"))
    common = value.get("common")
    contexts = value.get("puzzle_contexts")
    defaults = value.get("solution_defaults", {})
    if not isinstance(common, dict):
        raise ClientError("CONFIG_FIELD_MISSING", "common")
    if not isinstance(contexts, dict) or not contexts:
        raise ClientError("CONFIG_FIELD_MISSING", "puzzle_contexts")
    if not isinstance(defaults, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in defaults.items()):
        raise ClientError("CONFIG_FIELD_INVALID", "solution_defaults")
    for puzzle_id, context in contexts.items():
        if not isinstance(puzzle_id, str) or not isinstance(context, dict):
            raise ClientError("CONFIG_FIELD_INVALID", "puzzle_contexts")
        for field in ("puzzle_type", "initial_state", "central_state", "generators"):
            if field not in context:
                raise ClientError("CONFIG_FIELD_MISSING", f"puzzle_contexts.{puzzle_id}.{field}")
    return PublisherConfig(version, copy.deepcopy(common), copy.deepcopy(contexts), dict(defaults))


def _read_text(path: Path) -> str:
    try:
        size = path.stat().st_size
        if size == 0:
            raise ClientError("INPUT_EMPTY")
        if size > MAX_INPUT_BYTES:
            raise ClientError("INPUT_TOO_LARGE")
        return path.read_text(encoding="utf-8-sig")
    except ClientError:
        raise
    except (OSError, UnicodeError) as exc:
        raise ClientError("INPUT_READ_FAILED", str(exc)) from None


def _move_path(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [token for token in stripped.split(".") if token]


def _replay(initial: list[int], path: list[str], generators: dict[str, list[int]]) -> list[int]:
    state = list(initial)
    for move in path:
        permutation = generators.get(move)
        if permutation is None:
            raise ClientError("INPUT_UNKNOWN_MOVE", move)
        if len(permutation) != len(state):
            raise ClientError("CONFIG_GENERATOR_LENGTH", move)
        try:
            state = [state[source] for source in permutation]
        except (IndexError, TypeError) as exc:
            raise ClientError("CONFIG_GENERATOR_INVALID", move) from exc
    return state


def _integer(row: dict[str, str], name: str, *, optional: bool = False) -> int | None:
    raw = row.get(name, "").strip()
    if optional and raw == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ClientError("INPUT_FIELD_INVALID", name) from None
    if value < 0:
        raise ClientError("INPUT_FIELD_INVALID", name)
    return value


def _semantic_idempotency(envelope: dict[str, Any]) -> str:
    semantic = {k: v for k, v in envelope.items() if k not in {"client_submission_id", "run_id", "idempotency_key", "submitted_at"}}
    return hashlib.sha256(canonical_bytes(semantic)).hexdigest()


def expand_solution_row(config: PublisherConfig, row: dict[str, str]) -> dict[str, Any]:
    merged = dict(config.solution_defaults)
    merged.update({key: value for key, value in row.items() if value is not None})
    puzzle_id_value = _integer(merged, "puzzle_id")
    assert puzzle_id_value is not None
    context = config.puzzle_contexts.get(str(puzzle_id_value))
    if context is None:
        raise ClientError("CONFIG_PUZZLE_MISSING", str(puzzle_id_value))
    initial = context.get("initial_state")
    central = context.get("central_state")
    generators = context.get("generators")
    if not isinstance(initial, list) or not isinstance(central, list) or not isinstance(generators, dict):
        raise ClientError("CONFIG_PUZZLE_INVALID", str(puzzle_id_value))
    path = _move_path(merged.get("solution", ""))
    reached = _replay(initial, path, generators)
    envelope = copy.deepcopy(config.common)
    envelope.update({
        "schema_version": config.schema_version,
        "client_submission_id": _uuid7(),
        "submitted_at": _utc_now(),
        "puzzle_id": puzzle_id_value,
        "puzzle_type": context["puzzle_type"],
        "proof": {
            "initial_state": initial,
            "central_state": central,
            "generators": generators,
            "initial_state_sha256": _sha(initial),
            "central_state_sha256": _sha(central),
            "generators_sha256": _sha(generators),
            "reached_state_sha256": _sha(reached),
        },
        "orientation": {
            "final_orientation": merged.get("final_orientation", "original"),
            "search_mode": merged.get("search_mode", "off"),
        },
        "solution": {
            "path": path,
            "length": len(path),
            "solved_depth": _integer(merged, "solved_depth") or 0,
            "validation": "valid",
            "collection_status": merged.get("collection_status", "first_solution"),
        },
    })
    if not isinstance(envelope.get("run_id"), str):
        raise ClientError("CONFIG_FIELD_MISSING", "common.run_id")
    collection_index = _integer(merged, "collection_index", optional=True)
    touch_depth = _integer(merged, "touch_depth", optional=True)
    if collection_index is not None:
        envelope["solution"]["collection_index"] = collection_index
    if touch_depth is not None:
        envelope["solution"]["touch_depth"] = touch_depth
    reflected_source = _move_path(merged.get("reflected_source_solution", ""))
    searched = _move_path(merged.get("searched_solution", ""))
    if reflected_source:
        envelope["orientation"]["reflected_source_path"] = reflected_source
        envelope["orientation"]["reflected_source_sha256"] = _sha(reflected_source)
    if searched:
        envelope["orientation"]["searched_path"] = searched
    envelope["idempotency_key"] = ""
    envelope["idempotency_key"] = _semantic_idempotency(envelope)
    return envelope


def _normalize_json(value: Any) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise ClientError("INPUT_OBJECT_REQUIRED")
    if "results" in value:
        version = _require_version(value.get("schema_version"))
        results = value.get("results")
        if not isinstance(results, list) or not results:
            raise ClientError("INPUT_EMPTY")
    else:
        version = _require_version(value.get("schema_version"))
        results = [value]
    if len(results) > MAX_ENVELOPES:
        raise ClientError("INPUT_TOO_MANY_RESULTS")
    output: list[dict[str, Any]] = []
    for envelope in results:
        if not isinstance(envelope, dict):
            raise ClientError("INPUT_ENVELOPE_OBJECT")
        if _require_version(envelope.get("schema_version")) != version:
            raise ClientError("INPUT_VERSION_MISMATCH")
        output.append(envelope)
    return version, output


def _table_rows(path: Path, config: PublisherConfig) -> tuple[int, list[dict[str, Any]]]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    text = _read_text(path)
    reader = csv.DictReader(text.splitlines(), delimiter=delimiter)
    if tuple(reader.fieldnames or ()) != SOLUTION_COLUMNS:
        raise ClientError("INPUT_TABLE_HEADER")
    rows = [expand_solution_row(config, dict(row)) for row in reader]
    if not rows:
        raise ClientError("INPUT_EMPTY")
    if len(rows) > MAX_ENVELOPES:
        raise ClientError("INPUT_TOO_MANY_RESULTS")
    return config.schema_version, rows


def load_envelopes(path: Path, config: PublisherConfig | None) -> tuple[int, list[dict[str, Any]]]:
    suffix = path.suffix.lower()
    if suffix in (".csv", ".tsv"):
        if config is None:
            raise ClientError("CONFIG_REQUIRED")
        return _table_rows(path, config)
    text = _read_text(path)
    if suffix in (".txt", ".moves"):
        if config is None:
            raise ClientError("CONFIG_REQUIRED")
        if len(config.puzzle_contexts) != 1:
            raise ClientError("CONFIG_PUZZLE_AMBIGUOUS")
        puzzle_id = next(iter(config.puzzle_contexts))
        row = dict(config.solution_defaults)
        row.update({"puzzle_id": puzzle_id, "solution": text.strip()})
        return config.schema_version, [expand_solution_row(config, row)]
    if suffix == ".jsonl":
        values = [_parse_json(line) for line in text.splitlines() if line.strip()]
        if not values:
            raise ClientError("INPUT_EMPTY")
        versions = {_require_version(value.get("schema_version")) for value in values if isinstance(value, dict)}
        if len(versions) != 1:
            raise ClientError("INPUT_MIXED_SCHEMA")
        version = next(iter(versions))
        rows: list[dict[str, Any]] = []
        for value in values:
            inner_version, inner_rows = _normalize_json(value)
            if inner_version != version:
                raise ClientError("INPUT_MIXED_SCHEMA")
            rows.extend(inner_rows)
        return version, rows
    return _normalize_json(_parse_json(text))


def partition_batches(
    version: int,
    envelopes: list[dict[str, Any]],
    max_compressed: int = MAX_COMPRESSED_BYTES,
    max_raw: int = MAX_RAW_BYTES,
) -> list[ArchivePart]:
    _require_version(version)
    if not envelopes:
        raise ClientError("INPUT_EMPTY")
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for envelope in envelopes:
        candidate = current + [envelope]
        raw = canonical_bytes({"schema_version": version, "results": candidate})
        compressed = gzip_bytes(raw)
        if len(raw) <= max_raw and len(compressed) <= max_compressed:
            current = candidate
            continue
        if not current:
            raise ClientError("ENVELOPE_TOO_LARGE")
        groups.append(current)
        current = [envelope]
        raw = canonical_bytes({"schema_version": version, "results": current})
        if len(raw) > max_raw or len(gzip_bytes(raw)) > max_compressed:
            raise ClientError("ENVELOPE_TOO_LARGE")
    if current:
        groups.append(current)
    count = len(groups)
    parts: list[ArchivePart] = []
    for index, group in enumerate(groups):
        raw = canonical_bytes({"schema_version": version, "results": group})
        parts.append(ArchivePart(index, count, version, raw, gzip_bytes(raw)))
    return parts


def preflight(path: Path, config_path: Path | None = None) -> PreflightReport:
    config = load_config(config_path) if config_path else None
    version, envelopes = load_envelopes(path, config)
    parts = partition_batches(version, envelopes)
    return PreflightReport(
        schema_version=version,
        envelope_count=len(envelopes),
        raw_bytes=sum(len(part.raw) for part in parts),
        part_count=len(parts),
        endpoint_path=f"/v{version}/results",
    )

