from __future__ import annotations

import json
from pathlib import Path
import subprocess

import pandas as pd
import pytest

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.paths import SolutionRecord
from tools.cayleypy_public.results import PublishStatus
from tools.cayleypy_public.runner import PublicSearchRunError
from tools.cayleypy_public.runner import RunArtifacts
import tools.run_cayleypy_public as public_cli


def test_public_cli_uses_only_config_and_output_and_materializes_fake_run(
    tmp_path: Path, monkeypatch,
) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "author_name": "alice", "checkpoint_path": str(tmp_path / "model.pth"),
        "puzzle_info_json": str(tmp_path / "puzzle_info.json"),
        "test_csv": str(tmp_path / "test.csv"),
        "sample_submission_csv": str(tmp_path / "sample_submission.csv"),
        "puzzle_id_start": 7, "puzzle_id_end": 7, "beam_width": 2**16,
        "max_depth": 4, "reflect_mode": "off", "reflect_source_csv": None,
        "solution_mode": "first", "collect_until_depth": 4,
        "max_collected_solutions": 2, "touch_bfs_radius": 1,
        "publish_results": False, "results_ingest_url": "",
    }), encoding="utf-8")
    output = tmp_path / "out"

    contract = type("Contract", (), {
        "state_len": 3, "num_classes": 3, "move_count": 2,
        "central_state": (0, 1, 2), "generators": {"a": (0, 1, 2), "b": (0, 1, 2)},
        "initial_states": {7: (0, 1, 2)},
    })()
    model = type("Model", (), {"format": "batchnorm-folded", "backend": "mlp", "dtype": "fp16", "checkpoint_sha256": "a" * 64,
        "manifest": {"state_len": 3, "num_classes": 3, "output_dim": 1, "normalization": "batchnorm_folded"}})()
    plan = type("Plan", (), {"requested_beam": 2**16, "effective_beam": 2**16, "alignment_delta": 0,
        "profile_power": 16, "model_class": "output1", "runtime": {"b_micro": 2},
        "local_beam": 2**15, "parent_batch": 1, "stream3_batch_candidates": 2,
        "shard_capacity_candidates": 1024, "cross_puzzle_profile_note": ""})()
    record = SolutionRecord(7, "original", "", "", 0, 0, None, True, (0, 1, 2))
    artifacts = RunArtifacts((record,), pd.DataFrame({"initial_state_id": [7], "path": [""]}), (), (), (0,), (0.1,), ("first_solution",))

    monkeypatch.setattr(public_cli, "validate_t4_hardware", lambda: ["Tesla T4", "Tesla T4"])
    monkeypatch.setattr(public_cli, "load_puzzle_contract", lambda *args: contract)
    monkeypatch.setattr(public_cli, "export_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(public_cli, "select_profile", lambda *args: {"profile_power": 16, "validation_status": "measured", "hardware": "kaggle_2xt4", "runtime": {}})
    monkeypatch.setattr(public_cli, "derive_runtime", lambda *args: plan)
    monkeypatch.setattr(public_cli, "serialize_preflight", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(public_cli, "locate_or_build_runner", lambda *_, **__: tmp_path / "production_runner")
    monkeypatch.setattr(public_cli, "run_public_search", lambda *args, **kwargs: artifacts)

    assert public_cli.main(["--config-json", str(config_path), "--output-dir", str(output)]) == 0
    for relative in (
        "selected_profile.json", "preflight.json", "export/manifest.json", "beam_run_results.csv",
        "solutions/all_solutions.csv", "submission.csv", "run_summary.json", "logs",
    ):
        assert (output / relative).exists()

    parser = public_cli._parser()
    assert {action.dest for action in parser._actions if action.dest != "help"} == {"config_json", "output_dir"}



def test_validate_t4_hardware_accepts_only_exact_two_t4s(monkeypatch) -> None:
    def completed(names: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["nvidia-smi"], 0, stdout=names, stderr="")

    monkeypatch.setattr(public_cli.subprocess, "run", lambda *args, **kwargs: completed("Tesla T4\nNVIDIA T4\n"))
    assert public_cli.validate_t4_hardware() == ["Tesla T4", "NVIDIA T4"]
    monkeypatch.setattr(public_cli.subprocess, "run", lambda *args, **kwargs: completed("Tesla T4\nA100\n"))
    with pytest.raises(RuntimeError, match="exactly two Tesla T4"):
        public_cli.validate_t4_hardware()
    monkeypatch.setattr(public_cli.subprocess, "run", lambda *args, **kwargs: completed("Tesla T4\n"))
    with pytest.raises(RuntimeError, match="exactly two Tesla T4"):
        public_cli.validate_t4_hardware()


def test_sanitize_log_redacts_paths_credentials_and_tokens(tmp_path: Path, monkeypatch) -> None:
    fake_home = tmp_path / "private-home"
    private_build = tmp_path / "private-build"
    monkeypatch.setattr(public_cli.Path, "home", staticmethod(lambda: fake_home))
    token_fixtures = ("ghp_" + "a" * 40, "github_pat_" + "b" * 40, "cf_api_token_" + "c" * 12)
    text = (f"home={fake_home} repo={public_cli._REPO_ROOT} build={private_build} "
            f"url=https://alice:secret@example.test/repo {' '.join(token_fixtures)}")
    sanitized = public_cli._sanitize_log(text, (private_build,))
    for secret in (str(fake_home), str(public_cli._REPO_ROOT), str(private_build), "alice:secret",
                   *token_fixtures):
        assert secret not in sanitized
    assert "<redacted-path>" in sanitized
    assert "<redacted-credentials>@" in sanitized
    assert "<redacted-secret>" in sanitized


def test_locate_or_build_runner_pins_cutlass_and_sm75_without_subprocess_build(tmp_path: Path, monkeypatch) -> None:
    cutlass = tmp_path / "cutlass"
    (cutlass / ".git").mkdir(parents=True)
    build = tmp_path / "build"
    commands: list[tuple[object, ...]] = []
    monkeypatch.setenv("CAYLEYPY_CUTLASS_DIR", str(cutlass))
    monkeypatch.setenv("CAYLEYPY_BUILD_DIR", str(build))
    monkeypatch.setattr(public_cli, "_git_stdout", lambda *args, **kwargs: public_cli._CUTLASS_REV)
    monkeypatch.setattr(public_cli, "_pytorch_nccl_cmake_args", lambda **_: [
        "-DNCCL_LIBRARY=/torch/nccl/lib/libnccl.so.2",
        "-DNCCL_INCLUDE_DIR=/torch/nccl/include",
    ])

    def fake_run_logged(command, log_path, **kwargs) -> None:
        commands.append(tuple(command))
        if command[:3] == ["cmake", "--build", build]:
            build.mkdir(parents=True, exist_ok=True)
            (build / "production_runner").write_text("fake", encoding="utf-8")

    monkeypatch.setattr(public_cli, "_run_logged", fake_run_logged)
    runner = public_cli.locate_or_build_runner(tmp_path / "out", tmp_path / "puzzle_info.json")
    assert runner == (build / "production_runner").resolve()
    assert public_cli._CUTLASS_REV == "afa1772203677c5118fcd82537a9c8fefbcc7008"
    configure = next(command for command in commands if command[0] == "cmake" and "-S" in command)
    assert "-DBEAM_CUDA_ARCHITECTURES=75" in configure
    assert f"-DCUTLASS_DIR={cutlass}" in configure
    assert "-DNCCL_LIBRARY=/torch/nccl/lib/libnccl.so.2" in configure
    assert "-DNCCL_INCLUDE_DIR=/torch/nccl/include" in configure
    assert any(command[:4] == ("cmake", "--build", build, "--target") for command in commands)


def test_build_parallel_jobs_defaults_and_validates_environment(monkeypatch) -> None:
    monkeypatch.delenv("CAYLEYPY_BUILD_JOBS", raising=False)
    assert public_cli._build_parallel_jobs() == 2
    monkeypatch.setenv("CAYLEYPY_BUILD_JOBS", "1")
    assert public_cli._build_parallel_jobs() == 1
    monkeypatch.setenv("CAYLEYPY_BUILD_JOBS", "0")
    with pytest.raises(ValueError, match="CAYLEYPY_BUILD_JOBS"):
        public_cli._build_parallel_jobs()

def _publishing_config(tmp_path: Path) -> PublicRunConfig:
    return PublicRunConfig.from_mapping({"author_name": "alice", "checkpoint_path": str(tmp_path / "model.pth"), "puzzle_info_json": str(tmp_path / "puzzle_info.json"), "test_csv": str(tmp_path / "test.csv"), "sample_submission_csv": str(tmp_path / "sample_submission.csv"), "puzzle_id_start": 7, "puzzle_id_end": 7, "beam_width": 2**16, "max_depth": 4, "reflect_mode": "off", "reflect_source_csv": None, "solution_mode": "first", "collect_until_depth": 4, "max_collected_solutions": 2, "touch_bfs_radius": 1, "publish_results": True, "results_ingest_url": "https://ingest.example.test/results", "competition": "comp", "kaggle_owner": "owner", "kaggle_slug": "slug", "kaggle_version": 1, "solver_commit": "a" * 40, "kaggle_notebook_sha256": "b" * 64})


def _one_solution_artifacts() -> RunArtifacts:
    record = SolutionRecord(7, "original", "", "", 0, 0, None, True, (0, 1, 2))
    return RunArtifacts((record,), pd.DataFrame({"initial_state_id": [7], "path": [""]}), (), (), (0,), (0.1,), ("first_solution",))


def test_best_effort_publication_failure_keeps_solve_artifacts(tmp_path: Path, monkeypatch) -> None:
    config = _publishing_config(tmp_path)
    artifacts = _one_solution_artifacts()
    contract = type("Contract", (), {"state_len": 3, "move_count": 2, "central_state": (0, 1, 2), "generators": {"a": (0, 1, 2), "b": (0, 1, 2)}, "initial_states": {7: (0, 1, 2)}})()
    model = type("Model", (), {"format": "batchnorm-folded", "checkpoint_sha256": "c" * 64, "manifest": {"state_len": 3, "num_classes": 3, "output_dim": 1}})()
    plan = type("Plan", (), {"requested_beam": 2**16, "effective_beam": 2**16, "alignment_delta": 0, "profile_power": 16, "model_class": "output1", "runtime": {}})()
    failure = PublishStatus(False, True, "results endpoint is temporarily unavailable", None, 1, False, "https://ingest.example.test")
    monkeypatch.setattr(public_cli, "_publication_envelopes", lambda *args, **kwargs: [{"client_submission_id": "x"}])
    monkeypatch.setattr(public_cli, "publish_results", lambda *args, **kwargs: failure)
    materialized = public_cli._materialize_run_artifacts(artifacts, tmp_path)
    status = public_cli._publish_best_effort(config, contract, model, {"profile_registry_schema_version": 1}, plan, ["Tesla T4", "Tesla T4"], artifacts, tmp_path, 0.1)
    assert materialized["solution_count"] == 1
    assert (tmp_path / "submission.csv").is_file()
    assert (tmp_path / "solutions" / "all_solutions.csv").is_file()
    assert status["state"] == "failed"
    assert status["ok"] is False
    assert json.loads((tmp_path / "publish_status.json").read_text(encoding="utf-8"))["state"] == "failed"


def test_public_search_error_retains_partial_artifacts(tmp_path: Path) -> None:
    error = PublicSearchRunError("rank one failed", _one_solution_artifacts())
    summary = public_cli._materialize_run_artifacts(error.partial_artifacts, tmp_path)
    assert summary["solution_count"] == 1
    assert summary["invocation_count"] == 1
    assert (tmp_path / "submission.csv").is_file()
    assert (tmp_path / "beam_run_results.csv").is_file()
    assert (tmp_path / "solutions" / "solutions.csv").is_file()


@pytest.mark.parametrize("publish_enabled", (False, True))
def test_main_failed_search_materializes_partial_and_always_writes_publish_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, publish_enabled: bool,
) -> None:
    config_data = {
        "author_name": "alice", "checkpoint_path": str(tmp_path / "model.pth"),
        "puzzle_info_json": str(tmp_path / "puzzle_info.json"), "test_csv": str(tmp_path / "test.csv"),
        "sample_submission_csv": str(tmp_path / "sample_submission.csv"), "puzzle_id_start": 7,
        "puzzle_id_end": 7, "beam_width": 2**16, "max_depth": 4, "reflect_mode": "off",
        "reflect_source_csv": None, "solution_mode": "first", "collect_until_depth": 4,
        "max_collected_solutions": 2, "touch_bfs_radius": 1, "publish_results": publish_enabled,
        "results_ingest_url": "https://ingest.example.test/results" if publish_enabled else "",
    }
    if publish_enabled:
        config_data.update({"competition": "comp", "kaggle_owner": "owner", "kaggle_slug": "slug",
                            "kaggle_version": 1, "solver_commit": "a" * 40,
                            "kaggle_notebook_sha256": "b" * 64})
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config_data), encoding="utf-8")
    contract = type("Contract", (), {"state_len": 3, "num_classes": 3, "move_count": 2,
        "central_state": (0, 1, 2), "generators": {"a": (0, 1, 2), "b": (0, 1, 2)},
        "initial_states": {7: (0, 1, 2)}})()
    model = type("Model", (), {"format": "batchnorm-folded", "backend": "mlp", "dtype": "fp16", "checkpoint_sha256": "a" * 64,
        "manifest": {"state_len": 3, "num_classes": 3, "output_dim": 1}})()
    plan = type("Plan", (), {"requested_beam": 2**16, "effective_beam": 2**16, "alignment_delta": 0,
        "profile_power": 16, "model_class": "output1", "runtime": {"b_micro": 2}, "local_beam": 2**15,
        "parent_batch": 1, "stream3_batch_candidates": 2, "shard_capacity_candidates": 1024,
        "cross_puzzle_profile_note": ""})()
    partial = _one_solution_artifacts()
    monkeypatch.setattr(public_cli, "validate_t4_hardware", lambda: ["Tesla T4", "Tesla T4"])
    monkeypatch.setattr(public_cli, "load_puzzle_contract", lambda *args: contract)
    monkeypatch.setattr(public_cli, "export_checkpoint", lambda *args, **kwargs: model)
    monkeypatch.setattr(public_cli, "select_profile", lambda *args: {"profile_power": 16, "validation_status": "measured", "hardware": "kaggle_2xt4", "runtime": {}})
    monkeypatch.setattr(public_cli, "derive_runtime", lambda *args: plan)
    monkeypatch.setattr(public_cli, "serialize_preflight", lambda *args, **kwargs: {"ok": True})
    monkeypatch.setattr(public_cli, "locate_or_build_runner", lambda *_, **__: tmp_path / "production_runner")
    monkeypatch.setattr(public_cli, "run_public_search", lambda *args, **kwargs: (_ for _ in ()).throw(PublicSearchRunError("rank failed", partial)))
    if publish_enabled:
        monkeypatch.setattr(public_cli, "_publication_envelopes", lambda *args, **kwargs: [{"client_submission_id": "partial"}])
        monkeypatch.setattr(public_cli, "build_result_archives", lambda items: [b"archive"])
        monkeypatch.setattr(public_cli, "publish_result_archive", lambda *args, **kwargs: PublishStatus(True, False, None, 202, 1, False, "https://ingest.example.test"))
    calls: list[RunArtifacts] = []
    original_publish = public_cli._publish_best_effort
    def observe_publish(*args, **kwargs):
        calls.append(args[6])
        return original_publish(*args, **kwargs)
    monkeypatch.setattr(public_cli, "_publish_best_effort", observe_publish)

    output = tmp_path / "out"
    assert public_cli.main(["--config-json", str(config_path), "--output-dir", str(output)]) == 2
    summary = json.loads((output / "run_summary.json").read_text(encoding="utf-8"))
    status = json.loads((output / "publish_status.json").read_text(encoding="utf-8"))
    assert summary["status"] == "failed" and summary["solution_count"] == 1
    assert calls == [partial]
    assert status["state"] == ("published" if publish_enabled else "skipped")
    assert status["reason"] == "disabled" if not publish_enabled else status["result_count"] == 1


def _publication_record(puzzle_id: int, path: str, *, variant: str = "original", valid: bool = True) -> SolutionRecord:
    tokens = tuple(token for token in path.split(".") if token)
    return SolutionRecord(
        puzzle_id, variant, path, path, len(tokens), 0, None, valid, (0, 1, 2)
    )


def test_first_mode_publishes_only_shortest_valid_solution_per_puzzle() -> None:
    records = (
        _publication_record(7, "a.a"),
        _publication_record(7, "b"),
        _publication_record(7, "", variant="source"),
        _publication_record(7, "a", valid=False),
        _publication_record(8, "a.b"),
    )

    selected = public_cli._publication_records("first", records)

    assert [(record.puzzle_id, record.path) for record in selected] == [(7, "b"), (8, "a.b")]


def test_collect_mode_publishes_every_valid_discovered_solution() -> None:
    records = (
        _publication_record(7, "a.a"),
        _publication_record(7, "b"),
        _publication_record(7, "", variant="source"),
        _publication_record(7, "a", valid=False),
        _publication_record(8, "a.b"),
    )

    selected = public_cli._publication_records("collect", records)

    assert [(record.puzzle_id, record.path) for record in selected] == [
        (7, "a.a"), (7, "b"), (8, "a.b"),
    ]

def test_history_budget_is_derived_before_preflight_and_keeps_headroom():
    ram, disk = public_cli._derive_history_budgets(
        available_ram_bytes=29_257_576_448,
        tmp_free_bytes=1_000_000_000_000,
    )

    assert ram == 27_757_576_448
    assert disk == 50 * 1024**3

    with pytest.raises(ValueError, match="/tmp"):
        public_cli._derive_history_budgets(
            available_ram_bytes=29_257_576_448,
            tmp_free_bytes=50 * 1024**3 - 1,
        )

def test_best_effort_sends_archives_sequentially_with_one_request_each(tmp_path: Path, monkeypatch) -> None:
    config = _publishing_config(tmp_path)
    artifacts = _one_solution_artifacts()
    contract = type("Contract", (), {"state_len": 3, "move_count": 2, "central_state": (0, 1, 2), "generators": {"a": (0, 1, 2), "b": (0, 1, 2)}, "initial_states": {7: (0, 1, 2)}})()
    model = type("Model", (), {"format": "batchnorm-folded", "checkpoint_sha256": "c" * 64, "manifest": {"state_len": 3, "num_classes": 3, "output_dim": 1}})()
    plan = type("Plan", (), {"requested_beam": 2**16, "effective_beam": 2**16, "alignment_delta": 0, "profile_power": 16, "model_class": "output1", "runtime": {}})()
    envelopes = [{"client_submission_id": str(index)} for index in range(3)]
    calls: list[tuple[bytes, int, int, int]] = []

    monkeypatch.setattr(public_cli, "_publication_envelopes", lambda *args, **kwargs: envelopes)
    monkeypatch.setattr(public_cli, "build_result_archives", lambda items: [b"archive-0", b"archive-1"], raising=False)

    def publish(url, archive, *, result_count, archive_index, archive_count):
        calls.append((archive, result_count, archive_index, archive_count))
        return PublishStatus(True, False, None, 202, result_count, False, "https://ingest.example.test")

    monkeypatch.setattr(public_cli, "publish_result_archive", publish, raising=False)

    status = public_cli._publish_best_effort(
        config, contract, model, {"profile_registry_schema_version": 1}, plan,
        ["Tesla T4", "Tesla T4"], artifacts, tmp_path, 0.1,
    )

    assert calls == [
        (b"archive-0", 3, 0, 2),
        (b"archive-1", 3, 1, 2),
    ]
    assert status["ok"] is True
    assert status["archive_count"] == 2
