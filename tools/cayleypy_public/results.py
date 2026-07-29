"""Canonical, privacy-bounded result envelopes for public CayleyPy runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import secrets
import time
import uuid

from jsonschema import Draft202012Validator, ValidationError
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen


SCHEMA_VERSION = 1
MAX_ENVELOPE_BYTES = 256 * 1024
MAX_RESULTS_PER_REQUEST = 100
_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs/cayleypy_results_schema_v1.json"

_KAGGLE_FIELDS = ("kernel_slug", "kernel_version", "notebook_sha256")
_PROOF_FIELDS = (
    "path_valid", "initial_state_sha256", "target_state_sha256",
    "reached_state_sha256", "generators_sha256",
)
_SOLUTION_FIELDS = (
    "puzzle_id", "path", "move_count", "found_depth", "touch_depth", "variant",
    "valid", "source_solution_sha256",
)
_PROFILE_FIELDS = (
    "profile_evidence_version", "profile_power", "model_class", "requested_beam",
    "effective_beam", "world_size",
)
_RUNTIME_FIELDS = (
    "b_micro", "stream1_concurrency", "stream3_ring_slots", "shard_count",
    "shard_capacity_scale_ppm", "stream4_batch_candidates",
    "stream4_trigger_candidates", "stream4_active_sort_slots",
)
_MANIFEST_FIELDS = (
    "state_len", "num_classes", "hd1", "hd2", "nrd", "output_dim", "dtype",
    "normalization", "layout", "batchnorm", "embeddingbag", "embedding",
)
_HARDWARE_FIELDS = ("platform", "accelerator", "accelerator_count", "world_size")
_TIMING_FIELDS = ("solve_seconds", "wall_seconds")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _project(source: Mapping[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {name: source[name] for name in fields if name in source}


def _canonical_bytes(value: object) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("result envelope contains a non-canonical JSON value") from error
    return text.encode("utf-8")


@lru_cache(maxsize=1)
def _schema_validator() -> Draft202012Validator:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _validate_schema(envelope: Mapping[str, object]) -> None:
    try:
        _schema_validator().validate(envelope)
    except ValidationError as error:
        raise ValueError("result envelope violates the exact v1 JSON schema") from error


def _uuid7() -> str:
    timestamp_ms = time.time_ns() // 1_000_000
    if not 0 <= timestamp_ms < (1 << 48):
        raise RuntimeError("current timestamp does not fit UUIDv7")
    random_bits = secrets.randbits(74)
    rand_a = random_bits >> 62
    rand_b = random_bits & ((1 << 62) - 1)
    value = (
        (timestamp_ms << 80)
        | (0x7 << 76)
        | (rand_a << 64)
        | (0b10 << 62)
        | rand_b
    )
    return str(uuid.UUID(int=value))


def _sanitized_payload(
    context: Mapping[str, object], solution: Mapping[str, object],
) -> dict[str, object]:
    kaggle = _project(_mapping(context.get("kaggle"), "context.kaggle"), _KAGGLE_FIELDS)
    proof = _project(
        _mapping(context.get("proof_bundle"), "context.proof_bundle"), _PROOF_FIELDS,
    )
    profile_source = _mapping(context.get("profile"), "context.profile")
    profile = _project(profile_source, _PROFILE_FIELDS)
    profile["runtime"] = _project(
        _mapping(profile_source.get("runtime"), "context.profile.runtime"), _RUNTIME_FIELDS,
    )
    model_source = _mapping(context.get("model"), "context.model")
    model = _project(model_source, ("checkpoint_sha256",))
    model["manifest"] = _project(
        _mapping(model_source.get("manifest"), "context.model.manifest"), _MANIFEST_FIELDS,
    )
    hardware = _project(
        _mapping(context.get("hardware"), "context.hardware"), _HARDWARE_FIELDS,
    )
    timings = _project(_mapping(context.get("timings"), "context.timings"), _TIMING_FIELDS)
    return {
        "schema_version": SCHEMA_VERSION,
        "author": context.get("author"),
        "kaggle": kaggle,
        "proof_bundle": proof,
        "solution": _project(solution, _SOLUTION_FIELDS),
        "profile": profile,
        "model": model,
        "hardware": hardware,
        "timings": timings,
        "solver_commit": context.get("solver_commit"),
    }


def build_result_envelope(
    context: Mapping[str, object], solution: Mapping[str, object],
) -> dict[str, object]:
    """Build one exact-schema envelope without copying unknown or private input fields."""
    context_mapping = _mapping(context, "context")
    solution_mapping = _mapping(solution, "solution")
    payload = _sanitized_payload(context_mapping, solution_mapping)
    envelope = {
        **payload,
        "submission_id": _uuid7(),
        "idempotency_key": sha256(_canonical_bytes(payload)).hexdigest(),
    }
    _validate_schema(envelope)
    if len(_canonical_bytes(envelope)) > MAX_ENVELOPE_BYTES:
        raise ValueError("result envelope exceeds 256 KiB")
    return envelope


@dataclass(frozen=True)
class PublishStatus:
    ok: bool
    retryable: bool
    safe_error: str | None
    status_code: int | None
    result_count: int
    duplicate: bool
    endpoint: str


def _safe_endpoint(url: str) -> str:
    try:
        parsed = urlsplit(url)
        hostname = parsed.hostname or ""
        if not hostname:
            return "<invalid endpoint>"
        host = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            host = f"{host}:{parsed.port}"
        return urlunsplit((parsed.scheme, host, parsed.path, "", ""))
    except (TypeError, ValueError):
        return "<invalid endpoint>"


def _bounded_safe_error(message: str) -> str:
    encoded = message.encode("utf-8")
    if len(encoded) <= 2 * 1024:
        return message
    return encoded[: 2 * 1024].decode("utf-8", errors="ignore")


def _write_publish_status(status: PublishStatus) -> None:
    target = Path.cwd() / "publish_status.json"
    temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    payload = {"schema_version": SCHEMA_VERSION, **asdict(status)}
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _finish(status: PublishStatus) -> PublishStatus:
    try:
        _write_publish_status(status)
    except Exception:
        return PublishStatus(
            ok=False,
            retryable=True,
            safe_error="publish status persistence failed",
            status_code=status.status_code,
            result_count=status.result_count,
            duplicate=False,
            endpoint=status.endpoint,
        )
    return status


def _failure(
    endpoint: str,
    result_count: int,
    message: str,
    *,
    retryable: bool,
    status_code: int | None = None,
) -> PublishStatus:
    return _finish(PublishStatus(
        ok=False,
        retryable=retryable,
        safe_error=_bounded_safe_error(message),
        status_code=status_code,
        result_count=result_count,
        duplicate=False,
        endpoint=endpoint,
    ))


def _validate_publish_envelope(envelope: object) -> dict[str, object]:
    normalized = dict(_mapping(envelope, "envelope"))
    _validate_schema(normalized)
    if len(_canonical_bytes(normalized)) > MAX_ENVELOPE_BYTES:
        raise ValueError("result envelope exceeds 256 KiB")
    semantic_payload = {
        name: normalized[name]
        for name in (
            "schema_version", "author", "kaggle", "proof_bundle", "solution", "profile",
            "model", "hardware", "timings", "solver_commit",
        )
    }
    expected_key = sha256(_canonical_bytes(semantic_payload)).hexdigest()
    if normalized["idempotency_key"] != expected_key:
        raise ValueError("result envelope idempotency key does not match its semantic payload")
    return normalized


def publish_results(
    url: str,
    envelopes: Sequence[dict],
    timeout_s: float = 15.0,
) -> PublishStatus:
    """Best-effort POST that always returns a safe status and persists it locally."""
    endpoint = _safe_endpoint(url)
    try:
        items = list(envelopes)
    except Exception:
        return _failure(endpoint, 0, "publish payload failed validation", retryable=False)
    result_count = len(items)
    if result_count == 0:
        return _failure(endpoint, 0, "publish request has no results", retryable=False)
    if result_count > MAX_RESULTS_PER_REQUEST:
        return _failure(
            endpoint,
            result_count,
            "publish request exceeds 100 results",
            retryable=False,
        )
    if isinstance(timeout_s, bool):
        return _failure(endpoint, result_count, "publish timeout is invalid", retryable=False)
    try:
        timeout = float(timeout_s)
    except (TypeError, ValueError):
        return _failure(endpoint, result_count, "publish timeout is invalid", retryable=False)
    if not math.isfinite(timeout) or timeout <= 0:
        return _failure(endpoint, result_count, "publish timeout is invalid", retryable=False)
    try:
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("unsupported results endpoint")
        validated = [_validate_publish_envelope(envelope) for envelope in items]
        body = _canonical_bytes({"schema_version": SCHEMA_VERSION, "results": validated})
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=timeout) as response:
            status_code = getattr(response, "status", None)
            if status_code is None:
                status_code = response.getcode()
            status_code = int(status_code)
    except HTTPError as error:
        status_code = int(error.code)
        return _failure(
            endpoint,
            result_count,
            f"results endpoint returned HTTP {status_code}",
            retryable=status_code == 429 or status_code >= 500,
            status_code=status_code,
        )
    except (TimeoutError, URLError, OSError):
        return _failure(endpoint, result_count, "results endpoint is temporarily unavailable", retryable=True)
    except (TypeError, ValueError, ValidationError):
        return _failure(endpoint, result_count, "publish payload failed validation", retryable=False)
    except Exception:
        return _failure(endpoint, result_count, "results publish failed safely", retryable=True)

    if status_code == 202:
        return _finish(PublishStatus(
            ok=True,
            retryable=False,
            safe_error=None,
            status_code=202,
            result_count=result_count,
            duplicate=False,
            endpoint=endpoint,
        ))
    if status_code == 200:
        return _finish(PublishStatus(
            ok=True,
            retryable=False,
            safe_error=None,
            status_code=200,
            result_count=result_count,
            duplicate=True,
            endpoint=endpoint,
        ))
    return _failure(
        endpoint,
        result_count,
        f"results endpoint returned HTTP {status_code}",
        retryable=status_code == 429 or status_code >= 500,
        status_code=status_code,
    )
