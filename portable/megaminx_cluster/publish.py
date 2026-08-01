"""Strict standard-library publisher for the CayleyPy SLURM result API v2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from ipaddress import ip_address
import json
import os
from pathlib import Path
import secrets
import socket
import time
import uuid
from typing import Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

SCHEMA_VERSION = 2
MAX_RESULTS = 2000
MAX_BODY_BYTES = 4 * 1024 * 1024
TRANSPORT_FIELDS = frozenset({"client_submission_id", "run_id", "idempotency_key", "submitted_at"})
NATIVE_SMS = frozenset({75, 80, 86, 89, 90, 120})
PROFILE_STATUSES = frozenset({"measured", "bounded_from_measured"})


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        return None


@dataclass(frozen=True)
class Receipt:
    submission_id: str
    idempotency_key: str
    status_url: str


@dataclass(frozen=True)
class PublishResult:
    ok: bool
    retryable: bool
    status_code: int | None
    receipts: tuple[Receipt, ...]
    safe_error: str | None
    endpoint: str


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _project(value: Mapping[str, object], fields: Sequence[str], label: str) -> dict[str, object]:
    missing = [field for field in fields if field not in value]
    if missing:
        raise ValueError(f"{label} missing required field: {missing[0]}")
    return {field: value[field] for field in fields}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash_json(value: object) -> str:
    return sha256(canonical_bytes(value)).hexdigest()


def _uuid7() -> str:
    timestamp_ms = time.time_ns() // 1_000_000
    random_bits = secrets.randbits(74)
    value = (timestamp_ms << 80) | (0x7 << 76) | ((random_bits >> 62) << 64) | (0b10 << 62) | (random_bits & ((1 << 62) - 1))
    return str(uuid.UUID(int=value))


def _submitted_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _semantic(envelope: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in envelope.items() if key not in TRANSPORT_FIELDS}


def _state(value: object, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or not 1 <= len(value) <= 120:
        raise ValueError(f"{label} must be a non-empty state array")
    state = list(value)
    if any(isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255 for item in state):
        raise ValueError(f"{label} contains an invalid state value")
    return state


def _tokens(value: object, label: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a token array")
    tokens = list(value)
    if any(not isinstance(token, str) or not token or len(token) > 128 for token in tokens):
        raise ValueError(f"{label} contains an invalid token")
    return tokens


def _apply(state: Sequence[int], path: Sequence[str], generators: Mapping[str, object]) -> list[int]:
    current = list(state)
    for token in path:
        permutation = _state(generators.get(token), f"generator {token}")
        if len(permutation) != len(current) or sorted(permutation) != list(range(len(current))):
            raise ValueError(f"generator {token} is not a permutation")
        current = [current[index] for index in permutation]
    return current


def _validate_hardware(envelope: Mapping[str, object]) -> None:
    profile = _mapping(envelope["profile"], "profile")
    hardware = _mapping(envelope["hardware"], "hardware")
    provenance = _mapping(envelope["provenance"], "provenance")
    count = hardware.get("accelerator_count")
    world = hardware.get("world_size")
    sm = hardware.get("native_sm")
    if hardware.get("platform") != "slurm" or provenance.get("platform") != "slurm":
        raise ValueError("publication requires SLURM provenance")
    if isinstance(world, bool) or not isinstance(world, int) or not 1 <= world <= 16 or count != world:
        raise ValueError("hardware world size is invalid")
    if sm not in NATIVE_SMS or profile.get("native_sm") != sm or profile.get("world_size") != world:
        raise ValueError("profile does not match exact native hardware")
    if profile.get("gpu_family") not in {str(name).split(" ", 1)[1] if str(name).startswith("NVIDIA ") else str(name) for name in hardware.get("gpu_names", [])}:
        # Family spelling varies in nvidia-smi; exact family is already checked by preflight.
        if not all(str(profile.get("gpu_family")) in str(name) for name in hardware.get("gpu_names", [])):
            raise ValueError("profile GPU family does not match hardware")
    if profile.get("vram_mib") != hardware.get("vram_mib_per_gpu"):
        raise ValueError("profile VRAM does not match hardware")
    if profile.get("profile_status") not in PROFILE_STATUSES:
        raise ValueError("profile is not evidence-backed")
    requested = profile.get("requested_beam")
    effective = profile.get("effective_beam")
    delta = profile.get("alignment_delta")
    if not all(isinstance(x, int) and not isinstance(x, bool) for x in (requested, effective, delta)) or effective - requested != delta or delta < 0:
        raise ValueError("profile beam alignment is invalid")
    if provenance.get("run_id") != envelope.get("run_id"):
        raise ValueError("provenance run id mismatch")


def _validate_replay(envelope: Mapping[str, object]) -> None:
    proof = _mapping(envelope["proof"], "proof")
    solution = _mapping(envelope["solution"], "solution")
    initial = _state(proof.get("initial_state"), "proof.initial_state")
    central = _state(proof.get("central_state"), "proof.central_state")
    generators = _mapping(proof.get("generators"), "proof.generators")
    path = _tokens(solution.get("path"), "solution.path")
    if solution.get("length") != len(path) or solution.get("validation") != "valid":
        raise ValueError("solution metadata is invalid")
    reached = _apply(initial, path, generators)
    if reached != central:
        raise ValueError("solution failed independent replay")
    hashes = {
        "initial_state_sha256": _hash_json(initial),
        "central_state_sha256": _hash_json(central),
        "generators_sha256": _hash_json(dict(generators)),
        "reached_state_sha256": _hash_json(reached),
    }
    if any(proof.get(key) != value for key, value in hashes.items()):
        raise ValueError("proof hash mismatch")


def validate_envelope(envelope: Mapping[str, object]) -> None:
    required = {"schema_version", "client_submission_id", "run_id", "idempotency_key", "submitted_at", "author", "competition", "puzzle_type", "puzzle_id", "proof", "orientation", "solution", "profile", "runtime", "model", "hardware", "timings", "provenance"}
    if set(envelope) != required or envelope.get("schema_version") != 2:
        raise ValueError("result does not match the exact v2 field set")
    if _mapping(envelope["author"], "author").get("verification") != "claimed":
        raise ValueError("author verification must be claimed")
    _validate_hardware(envelope)
    _validate_replay(envelope)
    expected = _hash_json(_semantic(envelope))
    if envelope.get("idempotency_key") != expected:
        raise ValueError("idempotency key mismatch")


def build_result_envelope(context: Mapping[str, object], solution: Mapping[str, object]) -> dict[str, object]:
    """Whitelist fields from a preflight context and independently validated solution."""
    if solution.get("valid") is not True:
        raise ValueError("solution must be independently validated")
    proof_source = _mapping(context.get("proof"), "context.proof")
    initial = _state(proof_source.get("initial_state"), "context.proof.initial_state")
    central = _state(proof_source.get("central_state"), "context.proof.central_state")
    generators = dict(_mapping(proof_source.get("generators"), "context.proof.generators"))
    path = _tokens(solution.get("path"), "solution.path")
    reached = _apply(initial, path, generators)
    proof = {"initial_state": initial, "central_state": central, "generators": generators, "initial_state_sha256": _hash_json(initial), "central_state_sha256": _hash_json(central), "generators_sha256": _hash_json(generators), "reached_state_sha256": _hash_json(reached)}
    orientation = {"search_mode": context.get("search_mode"), "final_orientation": solution.get("search", "original")}
    if orientation["final_orientation"] == "reflected":
        orientation.update({"searched_path": _tokens(solution.get("searched_path"), "solution.searched_path"), "reflected_source_path": _tokens(solution.get("reflected_source_path"), "solution.reflected_source_path"), "reflected_source_sha256": solution.get("reflected_source_sha256")})
    envelope = {
        "schema_version": 2,
        "client_submission_id": _uuid7(),
        "run_id": context.get("run_id"),
        "idempotency_key": "0" * 64,
        "submitted_at": _submitted_at(),
        "author": _project(_mapping(context.get("author"), "context.author"), ("name", "verification"), "author"),
        "competition": context.get("competition"), "puzzle_type": context.get("puzzle_type"), "puzzle_id": solution.get("puzzle_id"),
        "proof": proof, "orientation": orientation,
        "solution": {"path": path, "length": len(path), "solved_depth": solution.get("solved_depth", len(path)), "touch_depth": solution.get("touch_depth", 0), "validation": "valid", "collection_index": solution.get("collection_index", 0), "collection_status": solution.get("collection_status", "first_solution")},
        "profile": dict(_mapping(context.get("profile"), "context.profile")), "runtime": dict(_mapping(context.get("runtime"), "context.runtime")),
        "model": dict(_mapping(context.get("model"), "context.model")), "hardware": dict(_mapping(context.get("hardware"), "context.hardware")),
        "timings": dict(_mapping(context.get("timings"), "context.timings")), "provenance": dict(_mapping(context.get("provenance"), "context.provenance")),
    }
    envelope["idempotency_key"] = _hash_json(_semantic(envelope))
    validate_envelope(envelope)
    return envelope


def _safe_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    return urlunsplit((parsed.scheme, host, "", "", "")) if host else "<invalid endpoint>"


def _public_endpoint(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname:
        return False
    port = parsed.port or 443
    return all(ip_address(item[4][0]).is_global for item in socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM))


def _persist(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(canonical_bytes(value) + b"\n")
    os.replace(temporary, path)


def publish_batch(url: str, envelopes: Sequence[Mapping[str, object]], receipt_path: Path, timeout_s: float = 15.0) -> PublishResult:
    endpoint = _safe_endpoint(url)
    try:
        items = [dict(item) for item in envelopes]
        if not 1 <= len(items) <= MAX_RESULTS or not _public_endpoint(url):
            raise ValueError("invalid publication endpoint or batch size")
        for item in items:
            validate_envelope(item)
        body = canonical_bytes({"schema_version": 2, "results": items})
        if len(body) > MAX_BODY_BYTES:
            raise ValueError("publication body exceeds 4 MiB")
        request = Request(url, data=body, headers={"Content-Type": "application/json", "User-Agent": "CayleyPy-SLURM-Publisher/2.0"}, method="POST")
        with build_opener(_NoRedirect()).open(request, timeout=timeout_s) as response:
            code = int(response.status)
            response_body = json.loads(response.read(256 * 1024))
        if code != 202:
            raise HTTPError(url, code, "unexpected status", {}, None)
        receipts = tuple(Receipt(**item) for item in response_body.get("receipts", []))
        if len(receipts) != len(items) or any(not receipt.status_url.startswith(endpoint + "/") for receipt in receipts):
            raise ValueError("invalid receipt response")
        result = PublishResult(True, False, 202, receipts, None, endpoint)
    except HTTPError as error:
        result = PublishResult(False, error.code == 429 or error.code >= 500, error.code, (), f"results endpoint returned HTTP {error.code}", endpoint)
    except (OSError, TimeoutError, URLError):
        result = PublishResult(False, True, None, (), "results endpoint is temporarily unavailable", endpoint)
    except (TypeError, ValueError, json.JSONDecodeError):
        result = PublishResult(False, False, None, (), "publish payload failed validation", endpoint)
    _persist(receipt_path, {"schema_version": 2, **asdict(result)})
    return result


def poll_receipt(receipt: Receipt, status_path: Path, timeout_s: float = 15.0) -> Mapping[str, object]:
    if not _public_endpoint(receipt.status_url):
        raise ValueError("invalid status endpoint")
    request = Request(receipt.status_url, headers={"Accept": "application/json", "User-Agent": "CayleyPy-SLURM-Publisher/2.0"})
    with build_opener(_NoRedirect()).open(request, timeout=timeout_s) as response:
        if int(response.status) != 200:
            raise ValueError("status endpoint returned a non-200 response")
        status = json.loads(response.read(64 * 1024))
    if status.get("submission_id") != receipt.submission_id or status.get("idempotency_key") != receipt.idempotency_key:
        raise ValueError("status response does not match receipt")
    _persist(status_path, status)
    return status


def publish_existing(run_dir: Path, url: str) -> PublishResult:
    context = json.loads((run_dir / "publication_context.json").read_text(encoding="utf-8-sig"))
    validated = json.loads((run_dir / "validated_results.json").read_text(encoding="utf-8-sig"))
    envelopes = [build_result_envelope(context, {**item, "puzzle_id": validated["puzzle_id"]}) for item in validated["results"]]
    _persist(run_dir / "publication_payload.json", {"schema_version": 2, "results": envelopes})
    return publish_batch(url, envelopes, run_dir / "publish_receipt.json")
