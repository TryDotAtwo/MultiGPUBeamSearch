import json
from pathlib import Path
import socket
from urllib.error import HTTPError

import pytest

import portable.megaminx_cluster.publish as module


def context():
    return {
        "run_id": "slurm-32633", "author": {"name": "cluster-user", "verification": "claimed"},
        "competition": "santa-2023", "puzzle_type": "megaminx", "search_mode": "off",
        "proof": {"initial_state": [1, 0], "central_state": [0, 1], "generators": {"swap": [1, 0]}},
        "profile": {"requested_beam": 1000, "effective_beam": 1024, "alignment_delta": 24, "profile_power": 10, "profile_anchor_beam": 1024, "profile_status": "measured", "profile_evidence_id": "h100x4-p10", "gpu_family": "H100", "vram_mib": 81559, "native_sm": 90, "world_size": 4, "backend": "mlp", "model_class": "output_move_count"},
        "runtime": {"touch_bfs_radius": 0, "solution_mode": "first", "max_depth": 120, "max_collected_solutions": 16, "b_micro": 2048, "stream1_concurrency": 4, "stream3_ring_slots": 4, "shard_count": 4, "shard_capacity_scale_ppm": 1050000, "stream4_batch_candidates": 98304, "stream4_trigger_candidates": 98304, "stream4_active_sort_slots": 4},
        "model": {"filename": "megaminx.pt", "sha256": "d" * 64, "format": "resmlp-layernorm", "manifest": {"state_len": 2, "num_classes": 2, "hd1": 256, "hd2": 256, "nrd": 4, "output_dim": 12, "dtype": "fp16", "normalization": "layernorm", "layout": "row-major"}},
        "hardware": {"platform": "slurm", "gpu_names": ["NVIDIA H100 80GB HBM3"] * 4, "accelerator_count": 4, "world_size": 4, "native_sm": 90, "vram_mib_per_gpu": 81559, "driver_version": "570.133.20"},
        "timings": {"solve_us": 100, "wall_us": 200},
        "provenance": {"platform": "slurm", "cluster_name": "basis", "slurm_job_id": "32633", "slurm_array_task_id": None, "run_id": "slurm-32633", "release_tag": "v1", "release_asset": "megaminx-sm90-linux-x86_64.tar.zst", "release_manifest_sha256": "a" * 64, "solver_commit": "b" * 40},
    }


@pytest.fixture(autouse=True)
def public_dns(monkeypatch):
    monkeypatch.setattr(module.socket, "getaddrinfo", lambda host, port, *, type: [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("104.16.0.1", port))])


def envelope():
    return module.build_result_envelope(context(), {"valid": True, "search": "original", "path": ["swap"], "puzzle_id": 900})


def test_v2_envelope_has_slurm_provenance_and_stable_semantic_idempotency():
    first = envelope()
    second = envelope()
    assert first["schema_version"] == 2
    assert first["provenance"]["platform"] == "slurm"
    assert "kaggle" not in first
    assert first["idempotency_key"] == second["idempotency_key"]


@pytest.mark.parametrize("field", ["token", "logs", "absolute_path", "weights"])
def test_unknown_or_private_fields_are_rejected(field):
    item = envelope()
    item[field] = "secret"
    with pytest.raises(ValueError, match="exact v2"):
        module.validate_envelope(item)


def test_cross_hardware_profile_is_rejected():
    item = envelope()
    item["profile"]["native_sm"] = 80
    item["idempotency_key"] = module._hash_json(module._semantic(item))
    with pytest.raises(ValueError, match="native hardware"):
        module.validate_envelope(item)


class Response:
    status = 202
    def __enter__(self): return self
    def __exit__(self, *args): return False
    def read(self, limit):
        return json.dumps({"receipts": [{"submission_id": "019fbdc0-9347-7639-895c-d27e703694ad", "idempotency_key": envelope()["idempotency_key"], "status_url": "https://worker.example/v1/submissions/019fbdc0-9347-7639-895c-d27e703694ad"}]}).encode()


class Opener:
    def open(self, request, timeout):
        assert request.full_url.endswith("/v2/results")
        assert request.headers["Content-type"] == "application/json"
        return Response()


def test_202_receipt_is_persisted(monkeypatch, tmp_path):
    item = envelope()
    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    result = module.publish_batch("https://worker.example/v2/results", [item], tmp_path / "receipt.json")
    assert result.ok and result.status_code == 202
    assert json.loads((tmp_path / "receipt.json").read_text())["receipts"][0]["submission_id"].startswith("019f")


@pytest.mark.parametrize(("code", "retryable"), [(400, False), (429, True), (503, True), (302, False)])
def test_http_failures_are_safe(monkeypatch, tmp_path, code, retryable):
    class Failing:
        def open(self, request, timeout): raise HTTPError(request.full_url, code, "failure", {}, None)
    monkeypatch.setattr(module, "build_opener", lambda *args: Failing())
    result = module.publish_batch("https://worker.example/v2/results", [envelope()], tmp_path / "receipt.json")
    assert not result.ok and result.retryable is retryable
    assert result.safe_error == f"results endpoint returned HTTP {code}"


def test_publish_existing_writes_only_sanitized_payload(monkeypatch, tmp_path):
    (tmp_path / "publication_context.json").write_text(json.dumps({**context(), "private_token": "ghp_secret"}))
    (tmp_path / "validated_results.json").write_text(json.dumps({"puzzle_id": 900, "results": [{"valid": True, "search": "original", "path": ["swap"]}]}))
    monkeypatch.setattr(module, "build_opener", lambda *args: Opener())
    result = module.publish_existing(tmp_path, "https://worker.example/v2/results")
    assert result.ok
    payload = (tmp_path / "publication_payload.json").read_text()
    assert "ghp_secret" not in payload and "private_token" not in payload and "kaggle" not in payload
