"""Canonical, privacy-bounded result envelopes for public CayleyPy runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from hashlib import sha256
from ipaddress import ip_address
import json
import math
import os
from pathlib import Path
import secrets
import socket
import time
import uuid

from jsonschema import Draft202012Validator, ValidationError
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from tools.cayleypy_public.paths import invert_path, tokenize_path


SCHEMA_VERSION = 1
MAX_ENVELOPE_BYTES = 256 * 1024
MAX_PUBLISH_REQUEST_BYTES = 4 * 1024 * 1024
MAX_RESULTS_PER_REQUEST = 100

class _NoRedirectHandler(HTTPRedirectHandler):
    """Return the first 3xx response as an HTTPError without following Location."""

    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def urlopen(request: Request, timeout: float):
    """Open exactly one configured endpoint; redirects are deliberately disabled."""
    return build_opener(_NoRedirectHandler()).open(request, timeout=timeout)


def _is_public_resolved_address(address_text: str) -> bool:
    try:
        address = ip_address(address_text)
    except ValueError:
        return False
    return address.is_global and not any((
        address.is_loopback,
        address.is_private,
        address.is_link_local,
        address.is_reserved,
        address.is_unspecified,
        address.is_multicast,
    ))


def _endpoint_resolves_only_to_public_addresses(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        if parsed.hostname is None:
            return False
        answers = socket.getaddrinfo(
            parsed.hostname, parsed.port or 443, type=socket.SOCK_STREAM,
        )
    except (ValueError, socket.gaierror, OSError):
        return False
    addresses: list[str] = []
    for family, _, _, _, sockaddr in answers:
        if family not in {socket.AF_INET, socket.AF_INET6} or not sockaddr:
            return False
        addresses.append(sockaddr[0])
    return bool(addresses) and all(_is_public_resolved_address(address) for address in addresses)

_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "configs/cayleypy_results_schema_v1.json"

_AUTHOR_FIELDS = ("name", "kaggle_username")
_KAGGLE_FIELDS = ("owner", "slug", "version", "run_url", "notebook_sha256")
_PROFILE_FIELDS = (
    "requested_beam", "effective_beam", "alignment_delta", "selected_profile",
    "evidence", "profile_evidence_version", "profile_power", "model_class",
    "world_size",
)
_RUNTIME_FIELDS = (
    "touch_bfs_radius", "solution_mode", "max_depth", "max_collected_solutions",
    "b_micro", "stream1_concurrency", "stream3_ring_slots", "shard_count",
    "shard_capacity_scale_ppm", "stream4_batch_candidates",
    "stream4_trigger_candidates", "stream4_active_sort_slots",
)
_MANIFEST_FIELDS = (
    "state_len", "num_classes", "hd1", "hd2", "nrd", "output_dim", "dtype",
    "normalization", "layout", "batchnorm", "embeddingbag", "embedding",
)
_MODEL_FIELDS = ("filename", "sha256", "format")
_HARDWARE_FIELDS = ("platform", "gpu_names", "accelerator_count", "world_size")
_TIMING_FIELDS = ("solve_us", "wall_us")
_SEMANTIC_TRANSPORT_FIELDS = frozenset({
    "client_submission_id", "idempotency_key", "submitted_at", "run_id",
})


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _project(source: Mapping[str, object], fields: tuple[str, ...]) -> dict[str, object]:
    return {name: source[name] for name in fields if name in source}


def _state_list(value: object, name: str) -> list[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be an integer state array")
    return list(value)


def _validate_state_classes(state: Sequence[object], num_classes: int, name: str) -> None:
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 or value >= num_classes
           for value in state):
        raise ValueError(
            f"{name} values must be integers in [0, num_classes) for the supported permutation contract"
        )


def _generator_map(value: object) -> dict[str, list[object]]:
    source = _mapping(value, "context.proof.generators")
    return {
        name: _state_list(permutation, f"context.proof.generators.{name}")
        for name, permutation in source.items()
    }


def _token_list(value: object, name: str) -> list[str]:
    try:
        return list(tokenize_path(value))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a strict dot-separated path") from error


def _hash_json(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _validate_model_output_contract(
    profile: Mapping[str, object], model: Mapping[str, object], move_count: int,
) -> None:
    manifest = _mapping(model.get("manifest"), "model.manifest")
    model_class = profile.get("model_class")
    output_dim = manifest.get("output_dim")
    if model_class == "output1" and output_dim != 1:
        raise ValueError("output1 profile requires model manifest output_dim=1")
    if model_class == "output_move_count" and output_dim != move_count:
        raise ValueError(
            "output_move_count profile requires output_dim equal to generator count"
        )


def _apply_tokens(
    state: tuple[int, ...], tokens: Sequence[object], generators: Mapping[str, tuple[int, ...]],
) -> tuple[int, ...]:
    current = state
    for token in tokens:
        if not isinstance(token, str) or token not in generators:
            raise ValueError("solution path contains an unknown move token")
        permutation = generators[token]
        current = tuple(current[index] for index in permutation)
    return current


def _validate_replay_contract(envelope: Mapping[str, object]) -> None:
    proof = _mapping(envelope.get("proof"), "envelope.proof")
    initial_state = tuple(_state_list(proof.get("initial_state"), "envelope.proof.initial_state"))
    central_state = tuple(_state_list(proof.get("central_state"), "envelope.proof.central_state"))
    state_len = len(initial_state)
    if not 1 <= state_len <= 120 or len(central_state) != state_len:
        raise ValueError("proof states must share a State128 logical length in 1..120")

    generator_source = _mapping(proof.get("generators"), "envelope.proof.generators")
    generators = {
        name: tuple(_state_list(permutation, f"envelope.proof.generators.{name}"))
        for name, permutation in generator_source.items()
    }
    expected_indices = tuple(range(state_len))
    if any(len(permutation) != state_len or tuple(sorted(permutation)) != expected_indices
           for permutation in generators.values()):
        raise ValueError("every proof generator must be a permutation of range(state_len)")

    manifest = _mapping(
        _mapping(envelope.get("model"), "envelope.model").get("manifest"),
        "envelope.model.manifest",
    )
    if manifest.get("state_len") != state_len:
        raise ValueError("model manifest state_len must equal proof state_len")
    if manifest.get("num_classes") != state_len:
        raise ValueError("model manifest num_classes must equal the supported permutation state_len")

    _validate_state_classes(initial_state, state_len, "proof.initial_state")
    _validate_state_classes(central_state, state_len, "proof.central_state")
    expected_hashes = {
        "initial_state_sha256": _hash_json(list(initial_state)),
        "central_state_sha256": _hash_json(list(central_state)),
        "generators_sha256": _hash_json({name: list(value) for name, value in generators.items()}),
    }
    if any(proof.get(name) != expected for name, expected in expected_hashes.items()):
        raise ValueError("proof hash does not match its canonical replay value")

    solution = _mapping(envelope.get("solution"), "envelope.solution")
    path = solution.get("path")
    if isinstance(path, (str, bytes, bytearray)) or not isinstance(path, Sequence):
        raise ValueError("solution.path must be a token array")
    if solution.get("length") != len(path):
        raise ValueError("solution length does not match its token path")
    reached_state = _apply_tokens(initial_state, path, generators)
    if reached_state != central_state:
        raise ValueError("solution path does not replay from initial state to central state")
    if proof.get("reached_state_sha256") != _hash_json(list(reached_state)):
        raise ValueError("reached state hash does not match replay")

    orientation = _mapping(envelope.get("orientation"), "envelope.orientation")
    if orientation.get("final_orientation") == "reflected":
        source_path = orientation.get("reflected_source_path")
        searched_path = orientation.get("searched_path")
        if not isinstance(source_path, Sequence) or isinstance(source_path, (str, bytes, bytearray)):
            raise ValueError("reflected source path must be a token array")
        if not isinstance(searched_path, Sequence) or isinstance(searched_path, (str, bytes, bytearray)):
            raise ValueError("reflected searched path must be a token array")
        source_text = ".".join(source_path)
        if orientation.get("reflected_source_sha256") != sha256(source_text.encode("utf-8")).hexdigest():
            raise ValueError("reflected source hash does not match its exact source path")
        if _apply_tokens(initial_state, source_path, generators) != central_state:
            raise ValueError("reflected source path is not a valid original solution")
        reflected_state = _apply_tokens(central_state, source_path, generators)
        if _apply_tokens(reflected_state, searched_path, generators) != central_state:
            raise ValueError("reflected searched path does not replay to central state")
        expected_original = tokenize_path(invert_path(".".join(searched_path), generators))
        if tuple(path) != expected_original:
            raise ValueError("reflected searched path does not match original-oriented solution")


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
        _schema_validator().validate({"schema_version": SCHEMA_VERSION, "results": [envelope]})
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


def _submitted_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _semantic_payload(envelope: Mapping[str, object]) -> dict[str, object]:
    return {
        name: value
        for name, value in envelope.items()
        if name not in _SEMANTIC_TRANSPORT_FIELDS
    }


def _sanitized_payload(
    context: Mapping[str, object], solution: Mapping[str, object],
) -> dict[str, object]:
    if solution.get("valid") is not True:
        raise ValueError("solution must be independently validated before publication")

    author = _project(_mapping(context.get("author"), "context.author"), _AUTHOR_FIELDS)
    author["verification"] = "claimed"
    kaggle = _project(_mapping(context.get("kaggle"), "context.kaggle"), _KAGGLE_FIELDS)

    proof_source = _mapping(context.get("proof"), "context.proof")
    initial_state = _state_list(proof_source.get("initial_state"), "context.proof.initial_state")
    central_state = _state_list(proof_source.get("central_state"), "context.proof.central_state")
    generators = _generator_map(proof_source.get("generators"))
    reached_state = _state_list(solution.get("reached_state"), "solution.reached_state")
    proof = {
        "initial_state": initial_state,
        "central_state": central_state,
        "generators": generators,
        "initial_state_sha256": _hash_json(initial_state),
        "central_state_sha256": _hash_json(central_state),
        "generators_sha256": _hash_json(generators),
        "reached_state_sha256": _hash_json(reached_state),
    }

    profile = _project(_mapping(context.get("profile"), "context.profile"), _PROFILE_FIELDS)
    runtime = _project(_mapping(context.get("runtime"), "context.runtime"), _RUNTIME_FIELDS)
    model_source = _mapping(context.get("model"), "context.model")
    model = _project(model_source, _MODEL_FIELDS)
    model["manifest"] = _project(
        _mapping(model_source.get("manifest"), "context.model.manifest"), _MANIFEST_FIELDS,
    )
    _validate_model_output_contract(profile, model, len(generators))

    hardware = _project(
        _mapping(context.get("hardware"), "context.hardware"), _HARDWARE_FIELDS,
    )
    timings = _project(_mapping(context.get("timings"), "context.timings"), _TIMING_FIELDS)

    original_path = _token_list(
        solution.get("original_oriented_path"), "solution.original_oriented_path",
    )
    result_solution = {
        "path": original_path,
        "length": len(original_path),
        "solved_depth": solution.get("found_depth"),
        "validation": "valid",
        "collection_status": solution.get("collection_status"),
    }
    if solution.get("touch_depth") is not None:
        result_solution["touch_depth"] = solution["touch_depth"]
    if solution.get("collection_index") is not None:
        result_solution["collection_index"] = solution["collection_index"]

    final_orientation = solution.get("variant")
    orientation = {
        "search_mode": context.get("search_mode"),
        "final_orientation": final_orientation,
    }
    if final_orientation == "reflected":
        searched_path = _token_list(solution.get("path"), "solution.path")
        source_text = solution.get("reflected_source_path")
        if not isinstance(source_text, str):
            raise ValueError("solution.reflected_source_path must be a string")
        source_path = _token_list(source_text, "solution.reflected_source_path")
        source_sha256 = sha256(source_text.encode("utf-8")).hexdigest()
        provided_source_sha256 = solution.get("source_solution_sha256")
        if provided_source_sha256 is not None and provided_source_sha256 != source_sha256:
            raise ValueError("reflected source hash does not match its exact source path")
        orientation.update({
            "searched_path": searched_path,
            "reflected_source_path": source_path,
            "reflected_source_sha256": source_sha256,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "author": author,
        "kaggle": kaggle,
        "competition": context.get("competition"),
        "puzzle_type": context.get("puzzle_type"),
        "puzzle_id": solution.get("puzzle_id"),
        "proof": proof,
        "orientation": orientation,
        "solution": result_solution,
        "profile": profile,
        "runtime": runtime,
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
        "client_submission_id": _uuid7(),
        "run_id": context_mapping.get("run_id"),
        "submitted_at": _submitted_at_utc(),
    }
    envelope["idempotency_key"] = _hash_json(_semantic_payload(envelope))
    _validate_publish_envelope(envelope)
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
        return urlunsplit((parsed.scheme, host, "", "", ""))
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
    proof = _mapping(normalized.get("proof"), "envelope.proof")
    generators = _mapping(proof.get("generators"), "envelope.proof.generators")
    _validate_replay_contract(normalized)
    _validate_model_output_contract(
        _mapping(normalized.get("profile"), "envelope.profile"),
        _mapping(normalized.get("model"), "envelope.model"),
        len(generators),
    )
    expected_key = _hash_json(_semantic_payload(normalized))
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
        if not _endpoint_resolves_only_to_public_addresses(url):
            return _failure(
                endpoint,
                result_count,
                "results endpoint must resolve only to public IP addresses",
                retryable=False,
            )
        validated = [_validate_publish_envelope(envelope) for envelope in items]
        body = _canonical_bytes({"schema_version": SCHEMA_VERSION, "results": validated})
        if len(body) > MAX_PUBLISH_REQUEST_BYTES:
            return _failure(
                endpoint,
                result_count,
                "publish request exceeds 4 MiB",
                retryable=False,
            )
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "User-Agent": "CayleyPy-Kaggle-Publisher/1.0"},
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
