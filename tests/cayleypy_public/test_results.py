from __future__ import annotations

from io import BytesIO
from copy import deepcopy
import hashlib
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
    from tools.cayleypy_public.results import MAX_PUBLISH_REQUEST_BYTES
except ImportError:
    MAX_PUBLISH_REQUEST_BYTES = 4 * 1024 * 1024

try:
    from tools.cayleypy_public.results import PublishStatus, publish_results
except ImportError:
    PublishStatus = None
    publish_results = None


_REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = _REPO_ROOT / "configs/cayleypy_results_schema_v1.json"
GOLDEN_PATH = _REPO_ROOT / "configs/cayleypy_results_v1_golden.json"
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64



@pytest.fixture(autouse=True)
def public_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep publisher tests deterministic without live DNS."""
    monkeypatch.setattr(
        results_module.socket,
        "getaddrinfo",
        lambda host, port, *, type: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.16.0.1", port)),
        ],
    )

def _context() -> dict[str, object]:
    context, _ = _producer_inputs(_golden_case("original_unicode_author"))
    return context


def _solution() -> dict[str, object]:
    _, solution = _producer_inputs(_golden_case("original_unicode_author"))
    return solution


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _recompute_idempotency(envelope: dict[str, object]) -> None:
    semantic = {
        key: value
        for key, value in envelope.items()
        if key not in {
            "client_submission_id", "idempotency_key", "submitted_at", "run_id",
        }
    }
    envelope["idempotency_key"] = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()


def test_canonical_v1_batch_schema_and_shared_goldens_are_present() -> None:
    assert GOLDEN_PATH.is_file(), "canonical v1 shared golden fixtures are missing"
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    assert schema["required"] == ["schema_version", "results"]
    assert schema["$defs"]["state"]["maxItems"] == 120
    assert schema["$defs"]["manifest"]["properties"]["state_len"]["maximum"] == 120
    result_schema = schema["$defs"]["result"]
    assert "client_submission_id" in result_schema["required"]
    assert "submission_id" not in result_schema["properties"]
    assert set(result_schema["properties"]["proof"]["required"]) == {
        "initial_state", "central_state", "generators", "initial_state_sha256",
        "central_state_sha256", "generators_sha256", "reached_state_sha256",
    }
    assert [case["name"] for case in golden["cases"]] == [
        "original_unicode_author", "reflected", "empty_path_source",
    ]
    unicode_author = golden["cases"][0]["envelope"]["author"]["name"]
    assert unicode_author == "\u0410\u043b\u0438\u0441\u0430 \u0394"
    assert 63 not in map(ord, unicode_author)
    assert any(ord(character) > 127 for character in unicode_author)
    assert golden["cases"][0]["envelope"]["puzzle_type"] == "cube_3/3/3"
    assert result_schema["properties"]["puzzle_type"]["$ref"] == "#/$defs/puzzle_type"
    assert golden["semantic_excludes"] == [
        "client_submission_id", "idempotency_key", "submitted_at", "run_id",
    ]


    for case in golden["cases"]:
        envelope = case["envelope"]
        Draft202012Validator(schema).validate({"schema_version": 1, "results": [envelope]})
        assert _canonical_bytes(envelope).decode("utf-8") == case["canonical_json"]


def _golden_case(name: str) -> dict[str, object]:
    fixtures = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    return next(case for case in fixtures["cases"] if case["name"] == name)


def _producer_inputs(case: dict[str, object]) -> tuple[dict[str, object], dict[str, object]]:
    envelope = case["envelope"]
    proof = envelope["proof"]
    orientation = envelope["orientation"]
    expected_solution = envelope["solution"]
    model = dict(envelope["model"])
    model["manifest"] = {
        **model["manifest"],
        "source_weights": "C:/private/checkpoints/secret-model.pt",
        "api_token": "model-token-must-not-appear",
        "tensor": torch.tensor([42]),
    }
    context = {
        "run_id": envelope["run_id"],
        "author": {**envelope["author"], "unknown_author_field": "drop-me"},
        "kaggle": {**envelope["kaggle"], "unknown_kaggle_field": "drop-me"},
        "competition": envelope["competition"],
        "puzzle_type": envelope["puzzle_type"],
        "proof": {
            "initial_state": proof["initial_state"],
            "central_state": proof["central_state"],
            "generators": proof["generators"],
            "initial_state_sha256": "0" * 64,
            "private_trace": "drop-me",
        },
        "search_mode": orientation["search_mode"],
        "profile": {**envelope["profile"], "unknown_profile_field": "drop-me"},
        "runtime": {**envelope["runtime"], "environment": {"SECRET": "drop-me"}},
        "model": model,
        "hardware": {**envelope["hardware"], "driver_details": "drop-me"},
        "timings": {**envelope["timings"], "raw_timing_log": "drop-me"},
        "solver_commit": envelope["solver_commit"],
        "checkpoint_path": Path("C:/private/checkpoints/secret-model.pt"),
        "token": "top-level-token-must-not-appear",
        "tensor": torch.tensor([7]),
    }
    searched_path = orientation.get("searched_path", expected_solution["path"])
    solution = {
        "puzzle_id": envelope["puzzle_id"],
        "path": ".".join(searched_path),
        "original_oriented_path": ".".join(expected_solution["path"]),
        "found_depth": expected_solution["solved_depth"],
        "touch_depth": expected_solution.get("touch_depth"),
        "variant": orientation["final_orientation"],
        "valid": True,
        "reached_state": proof["central_state"],
        "collection_status": expected_solution["collection_status"],
        "raw_state_tensor": torch.tensor([1, 2, 3]),
    }
    if "collection_index" in expected_solution:
        solution["collection_index"] = expected_solution["collection_index"]
    if orientation["final_orientation"] == "reflected":
        solution["reflected_source_path"] = ".".join(orientation["reflected_source_path"])
        solution["source_solution_sha256"] = orientation["reflected_source_sha256"]
    return context, solution


@pytest.mark.parametrize(
    "case_name", ["original_unicode_author", "reflected", "empty_path_source"],
)
def test_build_result_envelope_matches_shared_canonical_goldens(
    monkeypatch, case_name: str,
) -> None:
    case = _golden_case(case_name)
    context, solution = _producer_inputs(case)
    expected = case["envelope"]
    expected_bytes = case["canonical_json"].encode("utf-8")

    assert hashlib.sha256(expected_bytes).hexdigest() == case["canonical_sha256"]
    assert expected["idempotency_key"] == case["semantic_sha256"]
    monkeypatch.setattr(results_module, "_uuid7", lambda: expected["client_submission_id"])
    monkeypatch.setattr(
        results_module, "_submitted_at_utc", lambda: expected["submitted_at"], raising=False,
    )

    envelope = build_result_envelope(context, solution)

    assert envelope == expected
    assert _canonical_bytes(envelope) == expected_bytes
    assert hashlib.sha256(_canonical_bytes(envelope)).hexdigest() == case["canonical_sha256"]


def test_build_result_envelope_has_exact_schema_and_required_provenance() -> None:
    assert callable(build_result_envelope), "Task 6 result envelope builder is missing"
    envelope = build_result_envelope(_context(), _solution())
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate({"schema_version": 1, "results": [envelope]})
    assert set(envelope) == {
        "schema_version", "client_submission_id", "run_id", "idempotency_key",
        "submitted_at", "author", "kaggle", "competition", "puzzle_type",
        "puzzle_id", "proof", "orientation", "solution", "profile", "runtime",
        "model", "hardware", "timings", "solver_commit",
    }
    assert envelope["author"] == {
        "name": "\u0410\u043b\u0438\u0441\u0430 \u0394", "kaggle_username": "alice-k", "verification": "claimed",
    }
    assert envelope["kaggle"]["notebook_sha256"] == SHA_A
    assert envelope["proof"]["initial_state"] == [2, 0, 1]
    assert envelope["proof"]["central_state"] == [0, 1, 2]
    assert envelope["solution"]["path"] == ["clockwise"]
    assert envelope["orientation"] == {"search_mode": "off", "final_orientation": "original"}
    assert envelope["runtime"]["shard_count"] == 4
    assert envelope["model"]["sha256"] == SHA_D
    assert envelope["hardware"]["gpu_names"] == ["Tesla T4", "Tesla T4"]
    assert envelope["timings"] == {"solve_us": 95_335, "wall_us": 4_720_000}
    assert uuid.UUID(envelope["client_submission_id"]).version == 7
    assert envelope["submitted_at"].endswith("Z")
    assert len(_canonical_bytes(envelope)) <= MAX_ENVELOPE_BYTES


def test_build_result_envelope_has_deterministic_semantic_idempotency() -> None:
    assert callable(build_result_envelope), "Task 6 result envelope builder is missing"
    context = _context()
    solution = _solution()

    first = build_result_envelope(context, solution)
    second = build_result_envelope(
        {**dict(reversed(tuple(context.items()))), "run_id": "different-run-id"},
        dict(reversed(tuple(solution.items()))),
    )
    changed = build_result_envelope(
        {**context, "timings": {**context["timings"], "solve_us": 95_336}}, solution,
    )

    assert first["client_submission_id"] != second["client_submission_id"]
    assert first["run_id"] != second["run_id"]
    assert first["idempotency_key"] == second["idempotency_key"]
    assert first["idempotency_key"] != changed["idempotency_key"]


@pytest.mark.parametrize(
    ("model_class", "output_dim", "message"),
    [
        ("output1", 2, "output1 profile requires model manifest output_dim=1"),
        (
            "output_move_count",
            1,
            "output_move_count profile requires output_dim equal to generator count",
        ),
    ],
)
def test_build_result_envelope_rejects_model_output_contract_drift(
    model_class: str, output_dim: int, message: str,
) -> None:
    context = _context()
    context["profile"] = {**context["profile"], "model_class": model_class}
    context["model"] = {
        **context["model"],
        "manifest": {**context["model"]["manifest"], "output_dim": output_dim},
    }
    with pytest.raises(ValueError, match=message):
        build_result_envelope(context, _solution())


@pytest.mark.parametrize("corruption", ["manifest_state_len", "generator_permutation", "solution_replay"])
def test_build_result_envelope_rejects_replay_invalid_source_context(corruption: str) -> None:
    context = deepcopy(_context())
    solution = deepcopy(_solution())
    if corruption == "manifest_state_len":
        context["model"]["manifest"]["state_len"] = 121
    elif corruption == "generator_permutation":
        context["proof"]["generators"]["clockwise"] = [0, 0, 2]
    elif corruption == "solution_replay":
        solution["path"] = "counterclockwise"
        solution["original_oriented_path"] = "counterclockwise"
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(corruption)

    with pytest.raises(ValueError):
        build_result_envelope(context, solution)


@pytest.mark.parametrize(
    ("state_name", "invalid_value"),
    [
        ("initial_state", -1),
        ("initial_state", 3),
        ("central_state", -1),
        ("central_state", 3),
    ],
)
def test_build_result_envelope_rejects_negative_or_overflow_state_labels(
    state_name: str, invalid_value: int,
) -> None:
    envelope = deepcopy(build_result_envelope(_context(), _solution()))
    envelope["proof"][state_name][0] = invalid_value

    with pytest.raises(
        ValueError,
        match=rf"proof\.{state_name} values must be integers in \[0, num_classes\)",
    ):
        results_module._validate_replay_contract(envelope)


def test_build_result_envelope_rejects_manifest_num_classes_mismatch() -> None:
    context = deepcopy(_context())
    context["model"]["manifest"]["num_classes"] = 4

    with pytest.raises(
        ValueError,
        match="model manifest num_classes must equal the supported permutation state_len",
    ):
        build_result_envelope(context, _solution())


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
    context = _context()
    solution = _solution()
    long_move = "U" * 64
    context["proof"] = {
        "initial_state": [0, 1, 2],
        "central_state": [0, 1, 2],
        "generators": {long_move: [0, 1, 2], "V" * 64: [0, 1, 2]},
    }
    solution.update({
        "path": ".".join([long_move] * 4_096),
        "original_oriented_path": ".".join([long_move] * 4_096),
        "found_depth": 4_096,
        "reached_state": [0, 1, 2],
    })

    with pytest.raises(ValueError, match="256 KiB"):
        build_result_envelope(context, solution)


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
        (
            "https://endpoint-user:endpoint-password@example.test/"
            "private/path-secret?token=query-secret#fragment-secret"
        ),
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
    assert status.endpoint == "https://example.test"
    request = observed["request"]
    assert observed["timeout"] == 2.5
    assert request.get_method() == "POST"
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("User-agent") == "CayleyPy-Kaggle-Publisher/1.0"
    assert request.data == _canonical_bytes({"schema_version": 1, "results": [envelope]})
    persisted = json.loads(Path("publish_status.json").read_text(encoding="utf-8"))
    assert persisted == {
        "duplicate": False,
        "endpoint": "https://example.test",
        "ok": True,
        "result_count": 1,
        "retryable": False,
        "safe_error": None,
        "schema_version": 1,
        "status_code": 202,
    }
    persisted_text = Path("publish_status.json").read_text(encoding="utf-8")
    for forbidden in (
        "endpoint-user", "endpoint-password", "path-secret", "query-secret",
        "fragment-secret",
    ):
        assert forbidden not in status.endpoint
        assert forbidden not in persisted_text


def test_publish_results_allows_100_small_envelopes_within_request_limit(
    monkeypatch, tmp_path: Path,
) -> None:
    assert callable(publish_results), "Task 6 publisher is missing"
    observed: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        observed.update(request=request, timeout=timeout)
        return _FakeResponse(202)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", fake_urlopen, raising=False)
    envelopes = [build_result_envelope(_context(), _solution()) for _ in range(100)]

    status = publish_results("https://results.example/ingest", envelopes)

    assert status.ok is True
    assert status.result_count == 100
    assert len(observed["request"].data) <= MAX_PUBLISH_REQUEST_BYTES
    assert observed["request"].data == _canonical_bytes(
        {"schema_version": 1, "results": envelopes}
    )


def test_publish_results_rejects_100_near_limit_envelopes_before_http(
    monkeypatch, tmp_path: Path,
) -> None:
    assert callable(publish_results), "Task 6 publisher is missing"
    context = _context()
    solution = _solution()
    long_move = "U" * 54
    context["proof"] = {
        "initial_state": [0, 1, 2],
        "central_state": [0, 1, 2],
        "generators": {long_move: [0, 1, 2], "V" * 54: [0, 1, 2]},
    }
    long_path = ".".join([long_move] * 4_096)
    solution.update({
        "path": long_path,
        "original_oriented_path": long_path,
        "found_depth": 4_096,
        "reached_state": [0, 1, 2],
    })
    envelope = build_result_envelope(context, solution)
    envelopes = [envelope] * 100
    request_bytes = _canonical_bytes({"schema_version": 1, "results": envelopes})

    assert MAX_ENVELOPE_BYTES - 32 * 1_024 < len(_canonical_bytes(envelope)) <= MAX_ENVELOPE_BYTES
    assert len(request_bytes) > MAX_PUBLISH_REQUEST_BYTES

    def unexpected_http(request, timeout):
        raise AssertionError("oversized canonical request must not reach HTTP")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", unexpected_http, raising=False)

    status = publish_results("https://results.example/ingest", envelopes)

    assert status.ok is False
    assert status.retryable is False
    assert status.status_code is None
    assert status.result_count == 100
    assert status.safe_error == "publish request exceeds 4 MiB"
    assert Path("publish_status.json").is_file()


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
    "location",
    [
        "https://127.0.0.1/private",
        "http://results.example/private",
    ],
)
def test_publish_results_does_not_follow_redirects(
    monkeypatch, tmp_path: Path, location: str,
) -> None:
    class RedirectingOpener:
        calls = 0

        def open(self, request, timeout):
            self.calls += 1
            raise HTTPError(
                request.full_url,
                302,
                "redirect blocked",
                {"Location": location},
                None,
            )

    opener = RedirectingOpener()

    def fake_build_opener(*handlers):
        assert any(
            isinstance(handler, results_module._NoRedirectHandler)
            for handler in handlers
        )
        return opener

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "build_opener", fake_build_opener)
    status = publish_results(
        "https://results.example/ingest",
        [build_result_envelope(_context(), _solution())],
    )

    assert opener.calls == 1
    assert status.ok is False
    assert status.retryable is False
    assert status.status_code == 302
    assert status.safe_error == "results endpoint returned HTTP 302"


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



def test_publish_results_rejects_model_output_semantic_drift_before_http(
    monkeypatch, tmp_path: Path,
) -> None:
    envelope = build_result_envelope(_context(), _solution())
    envelope["model"]["manifest"]["output_dim"] = 1
    semantic = {
        key: value
        for key, value in envelope.items()
        if key not in {
            "client_submission_id", "idempotency_key", "submitted_at", "run_id",
        }
    }
    envelope["idempotency_key"] = hashlib.sha256(_canonical_bytes(semantic)).hexdigest()

    def unexpected_http(request, timeout):
        raise AssertionError("semantic model drift must not reach HTTP")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", unexpected_http, raising=False)

    status = publish_results("https://results.example/ingest", [envelope])

    assert status.ok is False
    assert status.retryable is False
    assert status.safe_error == "publish payload failed validation"


@pytest.mark.parametrize(
    "corruption",
    [
        "initial_hash",
        "generator_permutation",
        "state_length",
        "manifest_state_len",
        "manifest_num_classes",
        "solution_replay",
        "reached_hash",
    ],
)
def test_publish_results_rejects_replay_contract_corruption_even_with_fresh_idempotency(
    monkeypatch, tmp_path: Path, corruption: str,
) -> None:
    envelope = deepcopy(build_result_envelope(_context(), _solution()))
    proof = envelope["proof"]
    manifest = envelope["model"]["manifest"]
    if corruption == "initial_hash":
        proof["initial_state_sha256"] = SHA_A
    elif corruption == "generator_permutation":
        proof["generators"]["clockwise"] = [0, 0, 2]
        proof["generators_sha256"] = hashlib.sha256(
            _canonical_bytes(proof["generators"])
        ).hexdigest()
    elif corruption == "state_length":
        proof["central_state"] = [0, 1]
        proof["central_state_sha256"] = hashlib.sha256(
            _canonical_bytes(proof["central_state"])
        ).hexdigest()
        proof["reached_state_sha256"] = proof["central_state_sha256"]
    elif corruption == "manifest_state_len":
        manifest["state_len"] = 121
    elif corruption == "manifest_num_classes":
        manifest["num_classes"] = 4
    elif corruption == "solution_replay":
        envelope["solution"]["path"] = ["counterclockwise"]
    elif corruption == "reached_hash":
        proof["reached_state_sha256"] = SHA_B
    else:  # pragma: no cover - parametrization is exhaustive.
        raise AssertionError(corruption)
    _recompute_idempotency(envelope)

    def unexpected_http(request, timeout):
        raise AssertionError("replay-invalid envelope must not reach HTTP")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", unexpected_http, raising=False)

    status = publish_results("https://results.example/ingest", [envelope])

    assert status.ok is False
    assert status.retryable is False
    assert status.safe_error == "publish payload failed validation"


def test_publish_results_rejects_reflection_provenance_tamper_before_http(
    monkeypatch, tmp_path: Path,
) -> None:
    case = _golden_case("reflected")
    context, solution = _producer_inputs(case)
    envelope = build_result_envelope(context, solution)
    envelope["orientation"]["reflected_source_sha256"] = SHA_C
    _recompute_idempotency(envelope)

    def unexpected_http(request, timeout):
        raise AssertionError("reflection-invalid envelope must not reach HTTP")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(results_module, "urlopen", unexpected_http, raising=False)
    status = publish_results("https://results.example/ingest", [envelope])

    assert status.ok is False
    assert status.retryable is False
    assert status.safe_error == "publish payload failed validation"


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

@pytest.mark.parametrize(
    "resolved_address",
    [
        "127.0.0.1", "10.0.0.1", "169.254.1.1", "240.0.0.1", "0.0.0.0", "224.0.0.1",
        "::1", "fc00::1", "fe80::1", "2001:db8::1", "::", "ff00::1",
    ],
)
def test_publish_results_rejects_nonpublic_dns_answers_before_http_without_leaking_host(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, resolved_address: str,
) -> None:
    family = socket.AF_INET6 if ":" in resolved_address else socket.AF_INET
    monkeypatch.setattr(
        results_module.socket,
        "getaddrinfo",
        lambda host, port, *, type: [
            (family, socket.SOCK_STREAM, 6, "", (resolved_address, port)),
        ],
        raising=False,
    )
    monkeypatch.setattr(
        results_module,
        "urlopen",
        lambda request, timeout: (_ for _ in ()).throw(AssertionError("private DNS answer reached HTTP")),
    )
    monkeypatch.chdir(tmp_path)

    status = publish_results(
        "https://dns-secret.example/ingest",
        [build_result_envelope(_context(), _solution())],
    )

    assert status.ok is False
    assert status.retryable is False
    assert status.safe_error == "results endpoint must resolve only to public IP addresses"
    assert "dns-secret" not in status.safe_error


def test_publish_results_accepts_global_ipv4_and_ipv6_dns_answers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        results_module.socket,
        "getaddrinfo",
        lambda host, port, *, type: [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.16.0.1", port)),
            (socket.AF_INET6, socket.SOCK_STREAM, 6, "", ("2606:4700:4700::1111", port, 0, 0)),
        ],
        raising=False,
    )
    monkeypatch.setattr(results_module, "urlopen", lambda request, timeout: _FakeResponse(202))
    monkeypatch.chdir(tmp_path)

    status = publish_results(
        "https://results.cloudflare.example/ingest",
        [build_result_envelope(_context(), _solution())],
    )

    assert status.ok is True
