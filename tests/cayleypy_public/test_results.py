from __future__ import annotations

from io import BytesIO
import socket
import json
from pathlib import Path
from urllib.error import HTTPError, URLError
import uuid

from jsonschema import Draft202012Validator
import pytest

import tools.cayleypy_public.results as results_module
import torch

try:
    from tools.cayleypy_public.results import MAX_ENVELOPE_BYTES, build_result_envelope
except ModuleNotFoundError:
    MAX_ENVELOPE_BYTES = 256 * 1024
    build_result_envelope = None

try:
    from tools.cayleypy_public.results import PublishStatus, publish_results
except ImportError:
    PublishStatus = None
    publish_results = None


SCHEMA_PATH = Path("configs/cayleypy_results_schema_v1.json")
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _context() -> dict[str, object]:
    return {
        "author": "alice",
        "kaggle": {
            "kernel_slug": "alice/cayleypy-public",
            "kernel_version": 7,
            "notebook_sha256": SHA_A,
            "unknown_kaggle_field": "drop-me",
        },
        "proof_bundle": {
            "path_valid": True,
            "initial_state_sha256": SHA_A,
            "target_state_sha256": SHA_B,
            "reached_state_sha256": SHA_B,
            "generators_sha256": SHA_C,
            "private_trace": "drop-me",
        },
        "profile": {
            "profile_evidence_version": 1,
            "profile_power": 16,
            "model_class": "output_move_count",
            "requested_beam": 65_536,
            "effective_beam": 65_536,
            "world_size": 2,
            "runtime": {
                "b_micro": 2_048,
                "stream1_concurrency": 4,
                "stream3_ring_slots": 4,
                "shard_count": 4,
                "shard_capacity_scale_ppm": 1_050_000,
                "stream4_batch_candidates": 98_304,
                "stream4_trigger_candidates": 98_304,
                "stream4_active_sort_slots": 4,
                "environment": {"SECRET": "drop-me"},
            },
        },
        "model": {
            "checkpoint_sha256": SHA_D,
            "manifest": {
                "state_len": 120,
                "num_classes": 120,
                "hd1": 256,
                "hd2": 256,
                "nrd": 4,
                "output_dim": 24,
                "dtype": "fp16",
                "normalization": "layernorm",
                "layout": "row-major input activations times weight_hxk",
                "source_weights": "C:/private/checkpoints/secret-model.pt",
                "api_token": "model-token-must-not-appear",
                "tensor": torch.tensor([42]),
            },
        },
        "hardware": {
            "platform": "kaggle",
            "accelerator": "Tesla T4",
            "accelerator_count": 2,
            "world_size": 2,
            "driver_details": "drop-me",
        },
        "timings": {
            "solve_seconds": 0.095335,
            "wall_seconds": 4.72,
            "raw_timing_log": "drop-me",
        },
        "solver_commit": "e" * 40,
        "checkpoint_path": Path("C:/private/checkpoints/secret-model.pt"),
        "token": "top-level-token-must-not-appear",
        "environment": {"KAGGLE_KEY": "environment-secret"},
        "tensor": torch.tensor([7]),
        "unknown": "drop-me",
    }


def _solution() -> dict[str, object]:
    return {
        "puzzle_id": 1,
        "path": "BR",
        "move_count": 1,
        "found_depth": 1,
        "touch_depth": 0,
        "variant": "original",
        "valid": True,
        "raw_state_tensor": torch.tensor([1, 2, 3]),
        "unknown": "drop-me",
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def test_build_result_envelope_has_exact_schema_and_required_provenance() -> None:
    assert callable(build_result_envelope), "Task 6 result envelope builder is missing"
    assert SCHEMA_PATH.is_file(), "Task 6 result schema is missing"

    envelope = build_result_envelope(_context(), _solution())
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(envelope)

    assert set(envelope) == {
        "schema_version", "submission_id", "idempotency_key", "author", "kaggle",
        "proof_bundle", "solution", "profile", "model", "hardware", "timings",
        "solver_commit",
    }
    assert envelope["schema_version"] == 1
    assert envelope["author"] == "alice"
    assert envelope["kaggle"]["kernel_slug"] == "alice/cayleypy-public"
    assert envelope["proof_bundle"]["path_valid"] is True
    assert envelope["solution"]["path"] == "BR"
    assert envelope["profile"]["runtime"]["shard_count"] == 4
    assert envelope["model"]["checkpoint_sha256"] == SHA_D
    assert envelope["hardware"]["accelerator_count"] == 2
    assert envelope["timings"]["solve_seconds"] == 0.095335
    assert envelope["solver_commit"] == "e" * 40
    assert uuid.UUID(envelope["submission_id"]).version == 7
    assert len(_canonical_bytes(envelope)) <= MAX_ENVELOPE_BYTES


def test_build_result_envelope_has_deterministic_semantic_idempotency() -> None:
    assert callable(build_result_envelope), "Task 6 result envelope builder is missing"
    context = _context()
    solution = _solution()

    first = build_result_envelope(context, solution)
    second = build_result_envelope(
        dict(reversed(tuple(context.items()))), dict(reversed(tuple(solution.items())))
    )
    changed = build_result_envelope(context, {**solution, "path": "BR.BR", "move_count": 2})

    assert first["submission_id"] != second["submission_id"]
    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["idempotency_key"] != changed["idempotency_key"]


def test_build_result_envelope_drops_unknown_sensitive_and_non_json_fields() -> None:
    assert callable(build_result_envelope), "Task 6 result envelope builder is missing"

    envelope = build_result_envelope(_context(), _solution())
    serialized = _canonical_bytes(envelope).decode("utf-8")

    for forbidden in (
        "secret-model.pt", "model-token-must-not-appear", "top-level-token-must-not-appear",
        "environment-secret", "drop-me", "source_weights", "api_token", "tensor",
        "checkpoint_path", "environment", "unknown", "raw_state_tensor",
    ):
        assert forbidden not in serialized


def test_build_result_envelope_rejects_payload_over_256_kib() -> None:
    assert callable(build_result_envelope), "Task 6 result envelope builder is missing"
    solution = _solution()
    solution["path"] = "U" * (MAX_ENVELOPE_BYTES - 128)

    with pytest.raises(ValueError, match="256 KiB"):
        build_result_envelope(_context(), solution)


class _FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status
        self.headers = {"Authorization": "Bearer response-header-secret"}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False

    def read(self) -> bytes:
        raise AssertionError("publisher must not read response bodies")


def test_publish_results_posts_canonical_request_and_persists_202_status(
    monkeypatch, tmp_path: Path,
) -> None:
    assert callable(publish_results), "Task 6 publisher is missing"
    observed: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        observed.update(request=request, timeout=timeout)
        return _FakeResponse(202)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", fake_urlopen, raising=False)
    envelope = build_result_envelope(_context(), _solution())

    status = publish_results(
        "https://endpoint-user:endpoint-password@example.test/ingest?token=query-secret",
        [envelope],
        timeout_s=2.5,
    )

    assert isinstance(status, PublishStatus)
    assert status.ok is True
    assert status.retryable is False
    assert status.safe_error is None
    assert status.status_code == 202
    assert status.result_count == 1
    assert status.duplicate is False
    assert status.endpoint == "https://example.test/ingest"
    request = observed["request"]
    assert observed["timeout"] == 2.5
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.data == _canonical_bytes({"schema_version": 1, "results": [envelope]})
    persisted = json.loads(Path("publish_status.json").read_text(encoding="utf-8"))
    assert persisted == {
        "duplicate": False,
        "endpoint": "https://example.test/ingest",
        "ok": True,
        "result_count": 1,
        "retryable": False,
        "safe_error": None,
        "schema_version": 1,
        "status_code": 202,
    }


def test_publish_results_treats_http_200_as_duplicate_success(monkeypatch, tmp_path: Path) -> None:
    assert callable(publish_results), "Task 6 publisher is missing"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        results_module, "urlopen", lambda request, timeout: _FakeResponse(200), raising=False,
    )

    status = publish_results("https://results.example/ingest", [build_result_envelope(_context(), _solution())])

    assert status.ok is True
    assert status.duplicate is True
    assert status.status_code == 200
    assert Path("publish_status.json").is_file()


@pytest.mark.parametrize(
    ("error", "expected_status_code"),
    [
        (TimeoutError("endpoint-password query-secret timeout"), None),
        (URLError(socket.gaierror("dns-secret")), None),
        (
            HTTPError(
                "https://example.test/ingest", 429, "response-header-secret", {"Authorization": "Bearer response-header-secret"},
                BytesIO(b"response-body-secret"),
            ),
            429,
        ),
        (
            HTTPError(
                "https://example.test/ingest", 500, "response-header-secret", {"Authorization": "Bearer response-header-secret"},
                BytesIO(b"response-body-secret"),
            ),
            500,
        ),
    ],
)
def test_publish_results_returns_safe_retryable_status_for_network_failures(
    monkeypatch, tmp_path: Path, error: BaseException, expected_status_code: int | None,
) -> None:
    assert callable(publish_results), "Task 6 publisher is missing"

    def fail_urlopen(request, timeout):
        raise error

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", fail_urlopen, raising=False)
    status = publish_results(
        "https://endpoint-user:endpoint-password@example.test/ingest?token=query-secret",
        [build_result_envelope(_context(), _solution())],
    )

    assert status.ok is False
    assert status.retryable is True
    assert status.status_code == expected_status_code
    assert status.safe_error
    assert len(status.safe_error.encode("utf-8")) <= 2 * 1024
    persisted = Path("publish_status.json").read_text(encoding="utf-8")
    for forbidden in (
        "endpoint-user", "endpoint-password", "query-secret", "dns-secret",
        "response-header-secret", "response-body-secret", "Authorization", "Bearer",
    ):
        assert forbidden not in status.safe_error
        assert forbidden not in persisted


def test_publish_results_rejects_schema_drift_before_http(monkeypatch, tmp_path: Path) -> None:
    assert callable(publish_results), "Task 6 publisher is missing"
    envelope = build_result_envelope(_context(), _solution())
    envelope["unknown"] = "must-not-post"

    def unexpected_http(request, timeout):
        raise AssertionError("invalid envelopes must not reach HTTP")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", unexpected_http, raising=False)
    status = publish_results("https://results.example/ingest", [envelope])

    assert status.ok is False
    assert status.retryable is False
    assert status.status_code is None
    assert status.safe_error == "publish payload failed validation"
    assert Path("publish_status.json").is_file()


def test_publish_results_rejects_more_than_100_envelopes_before_http(
    monkeypatch, tmp_path: Path,
) -> None:
    assert callable(publish_results), "Task 6 publisher is missing"
    envelope = build_result_envelope(_context(), _solution())

    def unexpected_http(request, timeout):
        raise AssertionError("oversized requests must not reach HTTP")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", unexpected_http, raising=False)
    status = publish_results("https://results.example/ingest", [envelope] * 101)

    assert status.ok is False
    assert status.retryable is False
    assert status.result_count == 101
    assert status.safe_error == "publish request exceeds 100 results"
    assert Path("publish_status.json").is_file()
