from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from portable.megaminx_cluster.autotune.probe import (
    ProbeRequest,
    build_probe_command,
    classify_metrics,
    run_probe,
    _failure_status,
    _runtime_env,
)


RUNTIME = {
    "b_micro": 262144,
    "stream1_concurrency": 2,
    "stream3_ring_slots": 2,
    "shard_count": 8,
    "shard_capacity_scale_ppm": 1200000,
    "stream4_batch_candidates": 1048576,
    "stream4_trigger_candidates": 2097152,
    "stream4_active_sort_slots": 2,
    "final_materialize_chunk_candidates": 262144,
}


def request(tmp_path, **changes):
    values = dict(
        archive_root=tmp_path,
        candidate_dir=tmp_path / "candidate",
        world_size=8,
        puzzle_id=900,
        depth=120,
        requested_beam=30_000_000,
        runtime=RUNTIME,
        timeout_seconds=60,
        rendezvous_id="tune-a1",
        total_vram_mib=(40960,) * 8,
        required_scratch_bytes=100,
        provenance={"solver_commit": "abc", "manifest_digest": "def"},
    )
    values.update(changes)
    return ProbeRequest(**values)


def test_builds_validating_workflow_with_one_world_size(tmp_path):
    command = build_probe_command(request(tmp_path))
    assert command[:3] == ["python3", "-m", "portable.megaminx_cluster.workflow"]
    assert "--benchmark-depth" in command
    assert command[command.index("--world-size") + 1] == "8"
    assert command[command.index("--puzzle") + 1] == "900"
    assert command[command.index("--beam") + 1] == "30000000"


def valid_metrics():
    return {
        "requested_beam": 30_000_000,
        "effective_beam": 30_015_488,
        "wall_us": 2_000_000,
        "solve_us": 1_900_000,
        "setup_us": 100_000,
        "throughput": 1000,
        "host_ram_bytes": 100,
        "scratch_bytes": 100,
        "peak_vram_mib": [34000] * 8,
        "total_vram_mib": [40960] * 8,
        "replay_ok": True,
        "exactness_digest": "a" * 64,
        "cuda_ok": True,
        "nccl_ok": True,
        "provenance": {"solver_commit": "abc", "manifest_digest": "def"},
    }


def test_classify_metrics_accepts_complete_row_with_ten_percent_margin():
    result = classify_metrics(valid_metrics(), world_size=8, required_scratch_bytes=100)
    assert result == (True, "stable")


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"replay_ok": False}, "replay_failed"),
        ({"exactness_digest": ""}, "missing_metric"),
        ({"cuda_ok": False}, "cuda_error"),
        ({"nccl_ok": False}, "nccl_error"),
        ({"scratch_bytes": 99}, "scratch_capacity"),
        ({"peak_vram_mib": [35000] * 8}, "vram_margin"),
        ({"peak_vram_mib": [35000] * 7}, "missing_metric"),
    ],
)
def test_classify_metrics_rejects_any_failed_gate(change, status):
    payload = valid_metrics()
    payload.update(change)
    assert classify_metrics(payload, 8, 100) == (False, status)


def test_run_probe_records_timeout_without_losing_logs(tmp_path):
    def timeout_runner(command, **kwargs):
        raise subprocess.TimeoutExpired(command, kwargs["timeout"], output="partial", stderr="late")

    result = run_probe(request(tmp_path), run_command=timeout_runner)
    assert not result.stable
    assert result.status == "timeout"
    assert (tmp_path / "candidate" / "stdout.log").read_text() == "partial"


def test_run_probe_replays_workflow_result_and_records_digest(tmp_path):
    def successful_runner(command, **kwargs):
        candidate = tmp_path / "candidate"
        (candidate / "validated_results.json").write_text(json.dumps({
            "puzzle_id": 900,
            "reflect": "off",
            "results": [{"search": "original", "path": ["A", "B"], "valid": True}],
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "puzzle_solved=1", "")

    result = run_probe(
        request(tmp_path),
        run_command=successful_runner,
        peak_vram_mib=(34000,) * 8,
        effective_beam=30_015_488,
        throughput=1000,
    )
    assert result.stable
    assert result.metrics["replay_ok"] is True
    assert len(result.metrics["exactness_digest"]) == 64
    assert result.metrics["provenance"]["solver_commit"] == "abc"

def test_default_runner_uses_concurrent_peak_vram_monitor(tmp_path, monkeypatch):
    import portable.megaminx_cluster.autotune.probe as probe_module

    def monitored(command, **kwargs):
        candidate = tmp_path / "candidate"
        (candidate / "validated_results.json").write_text(json.dumps({
            "puzzle_id": 900,
            "reflect": "off",
            "results": [{"search": "original", "path": ["A"], "valid": True}],
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", ""), (34000,) * 8

    monkeypatch.setattr(probe_module, "_run_with_vram_monitor", monitored)
    result = run_probe(request(tmp_path), effective_beam=30_015_488)
    assert result.stable
    assert result.metrics["peak_vram_mib"] == [34000] * 8

def test_probe_derives_effective_beam_and_shard_capacity_env(tmp_path):
    captured = {}

    def successful_runner(command, **kwargs):
        captured.update(kwargs["env"])
        candidate = tmp_path / "candidate"
        (candidate / "validated_results.json").write_text(json.dumps({
            "puzzle_id": 900, "reflect": "off",
            "results": [{"search": "original", "path": ["A"], "valid": True}],
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = run_probe(
        request(tmp_path), run_command=successful_runner,
        peak_vram_mib=(34000,) * 8,
    )
    assert result.metrics["effective_beam"] == 30_015_488
    assert int(captured["BEAM_SHARD_CAPACITY_CANDIDATES"]) >= 30_015_488 // 8 // 8
    assert "BEAM_SHARD_CAPACITY_SCALE_PPM" not in captured


def test_failure_status_does_not_treat_nccl_in_directory_name_as_nccl_error():
    assert _failure_status(250, 'workflow_failed=/scratch/megaminx-clean-nccl-20260802/original.log') == 'process_error'
    assert _failure_status(250, 'ncclCommInitRank: unhandled cuda error') == 'nccl_error'


def test_run_probe_classifies_nested_solver_cuda_log(tmp_path):
    def failed_runner(command, **kwargs):
        candidate = tmp_path / "candidate"
        (candidate / "logs/original.log").write_text(
            "cuda stream fatal error: phase=stream3_remote_recv_collect flag=3002",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 250, "", "workflow_failed=original")

    result = run_probe(request(tmp_path), run_command=failed_runner)
    assert result.status == "cuda_error"

def test_run_probe_scores_only_depth8_benchmark_metrics(tmp_path):
    def successful_runner(command, **kwargs):
        candidate = tmp_path / "candidate"
        (candidate / "benchmark_metrics.json").write_text(json.dumps({
            "benchmark_depth": 8,
            "depth_sec": 2.0,
            "frontier_size": 30_015_488,
            "rank_samples": 1,
            "frontier_full": True,
        }), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "ok", "")

    result = run_probe(
        request(tmp_path, depth=8),
        run_command=successful_runner,
        peak_vram_mib=(34000,) * 8,
    )
    assert result.stable
    assert result.metrics["benchmark_depth"] == 8
    assert result.metrics["wall_us"] == 2_000_000
    assert result.metrics["throughput"] == pytest.approx(15_007_744)
    assert result.metrics["frontier_full"] is True

def test_runtime_env_scales_final_exchange_by_world_size(tmp_path):
    runtime = {
        "b_micro": 8192, "stream1_concurrency": 1, "stream3_ring_slots": 1,
        "shard_count": 8, "shard_capacity_scale_ppm": 1250000,
        "stream4_batch_candidates": 65536, "stream4_trigger_candidates": 131072,
        "stream4_active_sort_slots": 1, "final_materialize_chunk_candidates": 32768,
    }
    request = ProbeRequest(tmp_path, tmp_path / "run", 4, 900, 8, 30_000_000, runtime, 60, "job", (40000,) * 4, 1, {})
    env, _ = _runtime_env(request)
    assert env["BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM"] == "4000000"


def test_capacity_probe_can_measure_up_to_100_percent_gate():
    payload = valid_metrics()
    payload["peak_vram_mib"] = [40140] * 8
    assert classify_metrics(payload, 8, 100, vram_limit_percent=100) == (True, "stable")
    assert classify_metrics(payload, 8, 100) == (False, "vram_margin")
