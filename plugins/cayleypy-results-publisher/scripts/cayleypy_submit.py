#!/usr/bin/env python3
"""Safe, dependency-free CayleyPy public result publisher."""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import hashlib
import json
import os
import secrets
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

OFFICIAL_ENDPOINT_BASE: Final[str] = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev"
CLIENT_USER_AGENT: Final[str] = "cayleypy-results-publisher/0.1"
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
        self.detail = detail
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


_SECRET_FIELD_FRAGMENTS: Final[tuple[str, ...]] = ("token", "secret", "password", "authorization", "api_key", "private_key")


def _reject_secret_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            if any(fragment in normalized for fragment in _SECRET_FIELD_FRAGMENTS):
                raise ClientError("INPUT_SECRET_FIELD", f"{path}.{key}")
            _reject_secret_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secret_fields(child, f"{path}[{index}]")

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
    _reject_secret_fields(value)
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
    _reject_secret_fields(value)
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


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        return None


class HttpTransport:
    def __init__(self, timeout_seconds: float = 60.0, max_response_bytes: int = 1024 * 1024) -> None:
        self.timeout_seconds = timeout_seconds
        self.max_response_bytes = max_response_bytes
        self.opener = build_opener(_RejectRedirects())

    def request(self, method: str, url: str, body: bytes | None = None, headers: dict[str, str] | None = None) -> HttpResponse:
        request = Request(url, data=body, headers=headers or {}, method=method)
        try:
            response = self.opener.open(request, timeout=self.timeout_seconds)
            with response:
                payload = response.read(self.max_response_bytes + 1)
                if len(payload) > self.max_response_bytes:
                    raise ClientError("HTTP_RESPONSE_TOO_LARGE")
                return HttpResponse(response.status, dict(response.headers.items()), payload)
        except HTTPError as exc:
            payload = exc.read(self.max_response_bytes + 1)
            if len(payload) > self.max_response_bytes:
                raise ClientError("HTTP_RESPONSE_TOO_LARGE") from None
            return HttpResponse(exc.code, dict(exc.headers.items()), payload)
        except TimeoutError:
            raise ClientError("HTTP_TIMEOUT") from None
        except URLError as exc:
            raise ClientError("HTTP_TRANSPORT", type(exc.reason).__name__) from None


@dataclass(frozen=True)
class SubmitConfig:
    manifest_path: Path
    endpoint_base: str = OFFICIAL_ENDPOINT_BASE
    max_retries: int = 3
    retry_base_seconds: float = 1.0


@dataclass(frozen=True)
class PollConfig:
    timeout_seconds: float = 300.0
    interval_seconds: float = 2.0


@dataclass(frozen=True)
class PollSummary:
    published: int
    duplicate: int
    rejected: int
    unresolved: int


def _safe_segment(value: str) -> str:
    out = "".join(char if char.isascii() and (char.isalnum() or char in "._-") else "-" for char in value.lower()).strip("-")
    while "--" in out:
        out = out.replace("--", "-")
    if not out or len(out) > 128:
        raise ClientError("GITHUB_PATH_INVALID")
    return out


def _github_path(envelope: dict[str, Any], submission_id: str) -> str:
    day = str(envelope.get("submitted_at", ""))[:10]
    competition = _safe_segment(str(envelope.get("competition", "")))
    puzzle_type = _safe_segment(str(envelope.get("puzzle_type", "")))
    if envelope.get("schema_version") == 1:
        return f"results/v1/{competition}/{puzzle_type}/{int(envelope['puzzle_id'])}/{day}/{submission_id}.json"
    return f"data/v2/slurm/{competition}/{puzzle_type}/{day}/{submission_id}.json"


def _github_url(envelope: dict[str, Any], submission_id: str) -> str:
    path = "/".join(quote(part, safe="") for part in _github_path(envelope, submission_id).split("/"))
    return "https://raw.githubusercontent.com/TryDotAtwo/cayleypy-beam-results/refs/heads/ingest/staging/" + path


def _safe_json_response(response: HttpResponse) -> dict[str, Any]:
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError):
        raise ClientError("HTTP_INVALID_JSON") from None
    if not isinstance(value, dict):
        raise ClientError("HTTP_INVALID_JSON")
    return value


def _origin(value: str) -> tuple[str, str, int | None]:
    parsed = urlparse(value)
    return parsed.scheme, parsed.hostname or "", parsed.port


def _validate_endpoint(endpoint: str) -> None:
    parsed = urlparse(endpoint)
    if endpoint == OFFICIAL_ENDPOINT_BASE:
        return
    if parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"} and os.environ.get("CAYLEYPY_ALLOW_TEST_ENDPOINT") == "1":
        return
    raise ClientError("ENDPOINT_UNSAFE")


def _request_with_retry(transport: Any, method: str, url: str, body: bytes | None, headers: dict[str, str], config: SubmitConfig) -> HttpResponse:
    for attempt in range(config.max_retries + 1):
        try:
            response = transport.request(method, url, body, headers)
        except ClientError as exc:
            if exc.code not in {"HTTP_TIMEOUT", "HTTP_TRANSPORT"} or attempt == config.max_retries:
                raise
            response = None
        if response is not None and (response.status not in {429, 500, 502, 503, 504} or attempt == config.max_retries):
            return response
        delay = config.retry_base_seconds * (2 ** attempt)
        if response is not None:
            retry_after = response.headers.get("Retry-After", "").strip()
            if retry_after.isdigit():
                delay = min(float(retry_after), 60.0)
        time.sleep(delay)
    raise AssertionError("unreachable")


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(canonical_bytes(manifest) + b"\n")
    os.replace(temporary, path)


def load_manifest(path: Path) -> dict[str, Any]:
    value = _parse_json(_read_text(path))
    if not isinstance(value, dict) or value.get("manifest_version") != 1:
        raise ClientError("MANIFEST_INVALID")
    if not isinstance(value.get("parts"), list) or not isinstance(value.get("receipts"), list):
        raise ClientError("MANIFEST_INVALID")
    return value


def submit_parts(parts: list[ArchivePart], config: SubmitConfig, transport: Any) -> dict[str, Any]:
    if not parts:
        raise ClientError("INPUT_EMPTY")
    _validate_endpoint(config.endpoint_base)
    input_sha = hashlib.sha256(b"".join(part.raw for part in parts)).hexdigest()
    if config.manifest_path.exists():
        manifest = load_manifest(config.manifest_path)
        if manifest.get("input_sha256") != input_sha or manifest.get("endpoint_origin") != config.endpoint_base:
            raise ClientError("MANIFEST_INPUT_MISMATCH")
    else:
        manifest = {"manifest_version": 1, "endpoint_origin": config.endpoint_base, "input_sha256": input_sha, "parts": [], "receipts": []}
    completed = {item.get("sha256") for item in manifest["parts"] if isinstance(item, dict) and item.get("state") == "accepted"}
    for part in parts:
        digest = hashlib.sha256(part.compressed).hexdigest()
        if digest in completed:
            continue
        envelopes = json.loads(part.raw.decode("utf-8"))["results"]
        response = _request_with_retry(transport, "POST", config.endpoint_base + f"/v{part.version}/results", part.compressed, {
            "Content-Type": "application/gzip", "Accept": "application/json",
            "User-Agent": CLIENT_USER_AGENT,
            "X-CayleyPy-Archive-Index": str(part.index), "X-CayleyPy-Archive-Count": str(part.count),
        }, config)
        if response.status != 202:
            raise ClientError("SUBMIT_HTTP", str(response.status))
        value = _safe_json_response(response)
        receipts, errors = value.get("receipts"), value.get("errors", [])
        if not isinstance(receipts, list) or not isinstance(errors, list):
            raise ClientError("SUBMIT_RESPONSE_INVALID")
        error_indices: set[int] = set()
        for error in errors:
            if not isinstance(error, dict) or not isinstance(error.get("index"), int):
                raise ClientError("SUBMIT_RESPONSE_INVALID")
            error_index = error["index"]
            if error_index < 0 or error_index >= len(envelopes) or error_index in error_indices:
                raise ClientError("SUBMIT_RESPONSE_INVALID")
            error_indices.add(error_index)
        accepted_indices = [index for index in range(len(envelopes)) if index not in error_indices]
        if len(receipts) != len(accepted_indices):
            raise ClientError("SUBMIT_RESPONSE_INVALID")
        for envelope_index, receipt in zip(accepted_indices, receipts, strict=True):
            if not isinstance(receipt, dict):
                raise ClientError("SUBMIT_RESPONSE_INVALID")
            submission_id, idempotency, status_url = receipt.get("submission_id"), receipt.get("idempotency_key"), receipt.get("status_url")
            if not all(isinstance(item, str) for item in (submission_id, idempotency, status_url)):
                raise ClientError("SUBMIT_RESPONSE_INVALID")
            receipt_record = {"submission_id": submission_id, "idempotency_key": idempotency, "status_url": status_url, "github_url": _github_url(envelopes[envelope_index], submission_id), "state": "accepted"}
            existing_receipt = next((item for item in manifest["receipts"] if isinstance(item, dict) and item.get("submission_id") == submission_id), None)
            if existing_receipt is None:
                manifest["receipts"].append(receipt_record)
            else:
                existing_receipt.update(receipt_record)
        part_record = {"index": part.index, "sha256": digest, "compressed_bytes": len(part.compressed), "state": "accepted" if not errors else "partial"}
        existing_part = next((item for item in manifest["parts"] if isinstance(item, dict) and item.get("sha256") == digest), None)
        if existing_part is None:
            manifest["parts"].append(part_record)
        else:
            existing_part.update(part_record)
        save_manifest(config.manifest_path, manifest)
        if errors or len(receipts) != len(envelopes):
            raise ClientError("SUBMIT_PARTIAL")
    return manifest

def _safe_status_url(url: str, endpoint_origin: str) -> None:
    parsed = urlparse(url)
    if _origin(url) != _origin(endpoint_origin) or not parsed.path.startswith("/v1/submissions/"):
        raise ClientError("STATUS_URL_UNSAFE")


def poll_manifest(manifest: dict[str, Any], config: PollConfig, transport: Any) -> PollSummary:
    endpoint, receipts = manifest.get("endpoint_origin"), manifest.get("receipts")
    if not isinstance(endpoint, str) or not isinstance(receipts, list):
        raise ClientError("MANIFEST_INVALID")
    for receipt in receipts:
        if not isinstance(receipt, dict) or not isinstance(receipt.get("status_url"), str):
            raise ClientError("MANIFEST_INVALID")
        _safe_status_url(receipt["status_url"], endpoint)
    deadline = time.monotonic() + max(0.0, config.timeout_seconds)
    while True:
        for receipt in receipts:
            if receipt.get("state") in {"published", "duplicate", "rejected"}:
                continue
            response = transport.request("GET", receipt["status_url"], None, {"Accept": "application/json", "User-Agent": CLIENT_USER_AGENT})
            if response.status == 200:
                state = _safe_json_response(response).get("state")
                if state in {"staged", "published"}:
                    receipt["state"] = "published"
                elif state == "duplicate":
                    receipt["state"] = "duplicate"
                elif state in {"rejected", "dead_letter", "failed"}:
                    receipt["state"] = "rejected"
                elif state not in {"received", "queued", "retryable", "validating", "validated", "publishing"}:
                    raise ClientError("STATUS_UNKNOWN")
            elif response.status == 404:
                github_url = receipt.get("github_url")
                if not isinstance(github_url, str) or not github_url.startswith("https://raw.githubusercontent.com/TryDotAtwo/cayleypy-beam-results/"):
                    raise ClientError("GITHUB_URL_UNSAFE")
                github = transport.request("GET", github_url, None, {"Accept": "application/json", "User-Agent": CLIENT_USER_AGENT})
                if github.status == 200:
                    receipt["state"] = "published"
            else:
                raise ClientError("STATUS_HTTP", str(response.status))
        published = sum(item.get("state") == "published" for item in receipts)
        duplicate = sum(item.get("state") == "duplicate" for item in receipts)
        rejected = sum(item.get("state") == "rejected" for item in receipts)
        unresolved = len(receipts) - published - duplicate - rejected
        if unresolved == 0 or time.monotonic() >= deadline:
            return PollSummary(published, duplicate, rejected, unresolved)
        time.sleep(max(0.0, config.interval_seconds))


def init_config(source: Literal["kaggle", "native"], output: Path) -> None:
    if output.exists():
        raise ClientError("CONFIG_EXISTS")
    script_path = Path(__file__).resolve()
    candidates = (
        script_path.parent / "templates" / f"publisher-config-{source}.json",
        script_path.parents[1] / "templates" / f"publisher-config-{source}.json",
    )
    template = next((candidate for candidate in candidates if candidate.is_file()), None)
    if template is None:
        raise ClientError("TEMPLATE_MISSING")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(template.read_bytes())


def _print_summary(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Publish CayleyPy results through the official Cloudflare ingest service")
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init", help="write a fill-once publisher config")
    init_parser.add_argument("--source", choices=("kaggle", "native"), required=True)
    init_parser.add_argument("--output", type=Path, default=Path("publisher-config.json"))
    pre = sub.add_parser("preflight", help="validate and size input without network access")
    pre.add_argument("input", type=Path); pre.add_argument("--config", type=Path)
    send = sub.add_parser("submit", help="submit input to the pinned official endpoint")
    send.add_argument("input", type=Path); send.add_argument("--config", type=Path)
    send.add_argument("--manifest", type=Path, default=Path("cayleypy-receipts.json")); send.add_argument("--wait", action="store_true"); send.add_argument("--timeout", type=float, default=300.0)
    send.add_argument("--test-endpoint", help=argparse.SUPPRESS)
    poll = sub.add_parser("poll", help="poll a saved receipt manifest")
    poll.add_argument("--manifest", type=Path, default=Path("cayleypy-receipts.json")); poll.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            init_config(args.source, args.output); _print_summary({"status": "created", "config": str(args.output)}); return 0
        if args.command == "preflight":
            report = preflight(args.input, args.config); _print_summary(report.__dict__); return 0
        if args.command == "submit":
            publisher_config = load_config(args.config) if args.config else None
            version, envelopes = load_envelopes(args.input, publisher_config)
            parts = partition_batches(version, envelopes)
            endpoint = args.test_endpoint or OFFICIAL_ENDPOINT_BASE
            manifest = submit_parts(parts, SubmitConfig(args.manifest, endpoint), HttpTransport())
            if args.wait:
                summary = poll_manifest(manifest, PollConfig(args.timeout), HttpTransport()); save_manifest(args.manifest, manifest)
                _print_summary({**summary.__dict__, "manifest": str(args.manifest), "accepted": len(manifest["receipts"])})
                return 0 if summary.rejected == 0 and summary.unresolved == 0 else 4
            _print_summary({"accepted": len(manifest["receipts"]), "parts": len(parts), "manifest": str(args.manifest)}); return 0
        manifest = load_manifest(args.manifest)
        summary = poll_manifest(manifest, PollConfig(args.timeout), HttpTransport()); save_manifest(args.manifest, manifest)
        _print_summary({**summary.__dict__, "manifest": str(args.manifest)})
        return 0 if summary.rejected == 0 and summary.unresolved == 0 else 4
    except ClientError as exc:
        failure = {"status": "failed", "code": exc.code}
        if exc.code == "SUBMIT_HTTP" and exc.detail.isdigit():
            failure["http_status"] = int(exc.detail)
        _print_summary(failure)
        return 3 if exc.code.startswith(("HTTP_", "SUBMIT_")) else 2


if __name__ == "__main__":
    raise SystemExit(main())