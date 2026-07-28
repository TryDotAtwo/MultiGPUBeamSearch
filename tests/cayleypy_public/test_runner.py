from __future__ import annotations

from pathlib import Path
import io
from types import SimpleNamespace

import pandas as pd
import pytest

from tools.cayleypy_public.config import PublicRunConfig
from tools.cayleypy_public.data import PuzzleContract
from tools.cayleypy_public.model import ExportedModel
from tools.cayleypy_public.profile import RuntimePlan
import tools.cayleypy_public.runner as runner


GENERATORS = {
    "clockwise": (1, 2, 0),
    "counterclockwise": (2, 0, 1),
}


def _config(tmp_path: Path, *, solution_mode: str = "first", reflect_mode: str = "off",
            source: Path | None = None, end: int = 7) -> PublicRunConfig:
    return PublicRunConfig.from_mapping({
        "author_name": "alice", "checkpoint_path": tmp_path / "model.pth",
        "puzzle_info_json": tmp_path / "puzzle_info.json", "test_csv": tmp_path / "test.csv",
        "sample_submission_csv": tmp_path / "submission.csv", "puzzle_id_start": 7,
        "puzzle_id_end": end, "beam_width": 2**21, "max_depth": 100,
        "reflect_mode": reflect_mode, "reflect_source_csv": source, "solution_mode": solution_mode,
        "collect_until_depth": 12, "max_collected_solutions": 3, "touch_bfs_radius": 4,
        "publish_results": False, "results_ingest_url": "https://results.example/",
    })


def _plan(*, local_beam: int = 2**20) -> RuntimePlan:
    return RuntimePlan(2**21, 2**21, 0, 21, "output1", local_beam, 2048, 196608, 262144,
                       {"b_micro": 49152, "stream1_concurrency": 2, "stream3_ring_slots": 4,
                        "shard_count": 8, "shard_capacity_scale_ppm": 1000000,
                        "stream4_batch_candidates": 98304, "stream4_trigger_candidates": 98304,
                        "stream4_active_sort_slots": 2}, "")


def _contract(*, two_puzzles: bool = False) -> PuzzleContract:
    initial_states = {7: (1, 2, 0)}
    if two_puzzles:
        initial_states[8] = (1, 2, 0)
    return PuzzleContract(
        central_state=(0, 1, 2), generators=GENERATORS, initial_states=initial_states,
        sample_submission=pd.DataFrame({
            "initial_state_id": list(initial_states), "path": ["old.old"] * len(initial_states),
        }), state_len=3, num_classes=3,
    )


def _model() -> ExportedModel:
    return ExportedModel("batchnorm-folded", "fp16", "a" * 64, {"output_dim": 1})


def _successful_execution(invocation: runner.RunnerInvocation, output: str) -> runner.InvocationExecution:
    invocation.combined_log.write_text(output, encoding="utf-8")
    for rank, rank_log in enumerate(invocation.rank_logs):
        rank_log.write_text(f"rank={rank}\n{output}", encoding="utf-8")
    return runner.InvocationExecution(invocation, 0, 0.25, output)


def test_first_invocation_uses_exact_two_rank_torchrun_and_runtime_contract(tmp_path: Path) -> None:
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 24, 7, "original", tmp_path / "weights", tmp_path
    )

    assert invocation.command[:4] == ("python", "-m", "torch.distributed.run", "--nproc-per-node=2")
    assert "--redirects=3" in invocation.command and "--tee=0:3" in invocation.command
    assert f"--log-dir={invocation.torchrun_log_dir}" in invocation.command
    assert invocation.command[-3:] == ("7", "100", str(2**21))
    assert invocation.env["BEAM_SOLVED_NEIGHBORHOOD_RADIUS"] == "4"
    assert invocation.env["BEAM_REPAIR_K1_RADIUS"] == "4"
    assert invocation.env["BEAM_REPAIR_K2_RADIUS"] == "0"
    assert invocation.env["BEAM_SHARD_CAPACITY_CANDIDATES"] == "262144"
    assert {
        key: invocation.env[key]
        for key in (
            "BEAM_RUNTIME_CONFIG_MODE",
            "BEAM_SHARD_BUFFER_COUNT",
            "BEAM_GLOBAL_SPILL_CAPACITY",
            "BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM",
            "BEAM_GPU_HEADROOM_BYTES",
        )
    } == {
        "BEAM_RUNTIME_CONFIG_MODE": "manual",
        "BEAM_SHARD_BUFFER_COUNT": "2",
        "BEAM_GLOBAL_SPILL_CAPACITY": "0",
        "BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM": "1000000",
        "BEAM_GPU_HEADROOM_BYTES": str(768 * 1024**2),
    }
    assert {
        key: invocation.env[key]
        for key in (
            "BEAM_HISTORY_MODE",
            "BEAM_HISTORY_SLOT_COUNT",
            "BEAM_HISTORY_WORKERS",
            "BEAM_HISTORY_RAM_BYTES",
            "BEAM_HISTORY_DISK_BYTES",
            "BEAM_HISTORY_DISK_PATH",
            "BEAM_STREAM2_SUFFIX_RADIUS",
        )
    } == {
        "BEAM_HISTORY_MODE": "static_hybrid",
        "BEAM_HISTORY_SLOT_COUNT": "2",
        "BEAM_HISTORY_WORKERS": "1",
        "BEAM_HISTORY_RAM_BYTES": str(28 * 1024**3),
        "BEAM_HISTORY_DISK_BYTES": str(32 * 1024**3),
        "BEAM_HISTORY_DISK_PATH": invocation.env["BEAM_HISTORY_DIR"],
        "BEAM_STREAM2_SUFFIX_RADIUS": "0",
    }
    assert invocation.env["BEAM_HISTORY_DIR"].startswith("/tmp/beam_history_public/")
    assert not {"WORLD_SIZE", "RANK", "LOCAL_RANK"}.intersection(invocation.env)


def test_history_defaults_preflight_budget_and_tmp_space(tmp_path: Path) -> None:
    safe = runner.preflight_history_runtime(
        _config(tmp_path), _plan(local_beam=2**24), move_count=24,
        history_dir=Path("/tmp/beam_history_public/run/7/original"),
        tmp_free_bytes=32 * 1024**3,
    )

    assert safe.required_history_bytes <= safe.usable_history_bytes
    with pytest.raises(ValueError, match="/tmp free disk"):
        runner.preflight_history_runtime(
            _config(tmp_path), _plan(local_beam=128), move_count=24,
            history_dir=Path("/tmp/beam_history_public/run/7/original"),
            tmp_free_bytes=32 * 1024**3 - 1,
        )
    with pytest.raises(ValueError, match="under /tmp/beam_history_public"):
        runner.preflight_history_runtime(
            _config(tmp_path), _plan(local_beam=128), move_count=24,
            history_dir=tmp_path / "unsafe",
            tmp_free_bytes=32 * 1024**3,
        )


def test_collect_capacity_is_full_local_depth_not_host_solution_limit(tmp_path: Path) -> None:
    invocation = runner.build_runner_invocation(
        _config(tmp_path, solution_mode="collect"), _plan(local_beam=128), 24, 7, "original",
        tmp_path / "weights", tmp_path,
    )

    assert {key: invocation.env[key] for key in (
        "BEAM_SOLVE_BUCKET_MODE", "BEAM_SOLVE_BUCKET_STOP_DEPTH",
        "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS", "BEAM_SOLVED_RESULT_CAPACITY",
    )} == {"BEAM_SOLVE_BUCKET_MODE": "1", "BEAM_SOLVE_BUCKET_STOP_DEPTH": "12",
          "BEAM_SOLVE_BUCKET_MAX_SOLUTIONS": "3", "BEAM_SOLVED_RESULT_CAPACITY": "3072"}


def test_solved_snapshot_capacity_fails_closed_on_uint32_or_t4_memory_overflow() -> None:
    with pytest.raises(ValueError, match="uint32"):
        runner.derive_solved_result_capacity(_plan(local_beam=2**31), 24)
    with pytest.raises(ValueError, match="device memory"):
        runner.derive_solved_result_capacity(_plan(local_beam=400_000_000), 2)
    with pytest.raises(ValueError, match="snapshot plus current frontier"):
        runner.derive_solved_result_capacity(_plan(local_beam=2**24), 24)


def test_full_depth_capacity_uses_bounded_multi_chunk_gather_plan() -> None:
    plan = runner.derive_gather_chunk_plan(frontier_states=128, capacity=128 * 24, world_size=2)

    assert plan.records_per_chunk == 85
    assert plan.chunk_count == 37
    assert plan.records_per_chunk < plan.capacity
    large = runner.derive_gather_chunk_plan(frontier_states=2**20, capacity=2**20 * 24, world_size=2)
    assert large.records_per_chunk == 65_536
    assert large.chunk_count == 384


def test_real_tsv_schema_prefers_solution_path_and_maps_depths_adversarially(tmp_path: Path) -> None:
    result_tsv = tmp_path / "results.tsv"
    result_tsv.write_text(
        "puzzle_id\tdepth_index\tfound_depth\ttotal_depth\tknown_length\tdelta\towner_rank"
        "\tsolution\tsolution_path\n"
        "7\t8\t2\t4\t0\t0\t1\tWRONG\tclockwise.clockwise.clockwise.clockwise\n",
        encoding="utf-8",
    )

    parsed = runner.parse_runner_output("collection_status=capacity_reached\n", result_tsv, 7, "original")

    assert parsed.collection_status == "capacity_reached"
    assert len(parsed.records) == 1
    assert parsed.records[0].path == "clockwise.clockwise.clockwise.clockwise"
    assert parsed.records[0].found_depth == 2
    assert parsed.records[0].touch_depth == 2


def test_torchrun_rank_logs_are_real_stdout_and_stderr_for_both_ranks(tmp_path: Path) -> None:
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 2, 7, "original", tmp_path / "weights", tmp_path
    )
    for rank in (0, 1):
        rank_dir = invocation.torchrun_log_dir / "run" / "attempt_0" / str(rank)
        rank_dir.mkdir(parents=True)
        (rank_dir / "stdout.log").write_text(f"stdout-rank-{rank}\n", encoding="utf-8")
        (rank_dir / "stderr.log").write_text(f"stderr-rank-{rank}\n", encoding="utf-8")

    runner.collect_torchrun_rank_logs(invocation)

    assert invocation.rank_logs[0].read_text(encoding="utf-8") == (
        "[stdout]\nstdout-rank-0\n[stderr]\nstderr-rank-0\n"
    )
    assert invocation.rank_logs[1].read_text(encoding="utf-8") == (
        "[stdout]\nstdout-rank-1\n[stderr]\nstderr-rank-1\n"
    )


def test_missing_rank_log_is_a_hard_failure_not_an_empty_placeholder(tmp_path: Path) -> None:
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 2, 7, "original", tmp_path / "weights", tmp_path
    )
    rank_dir = invocation.torchrun_log_dir / "run" / "attempt_0" / "0"
    rank_dir.mkdir(parents=True)
    (rank_dir / "stdout.log").write_text("rank0", encoding="utf-8")
    (rank_dir / "stderr.log").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="rank 1"):
        runner.collect_torchrun_rank_logs(invocation)
    assert not invocation.rank_logs[1].exists()


def test_after_original_runs_original_then_reflected_and_updates_shortest_submission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[runner.RunnerInvocation] = []

    def execute(invocation: runner.RunnerInvocation, extra_env: dict[str, str]) -> runner.InvocationExecution:
        calls.append(invocation)
        if invocation.variant == "original":
            return _successful_execution(invocation, "solution_path=clockwise.clockwise\n")
        reflected = pd.read_csv(invocation.env["BEAM_TEST_CSV"])
        assert reflected.to_dict("records") == [{"initial_state_id": 7, "initial_state": "2,0,1"}]
        return _successful_execution(invocation, "solution_path=clockwise\n")

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, reflect_mode="after_original"), _contract(), _model(), _plan(local_beam=128),
        tmp_path / "weights", tmp_path / "artifacts",
    )

    assert [call.variant for call in calls] == ["original", "reflected"]
    assert [record.original_oriented_path for record in artifacts.solution_records] == [
        "clockwise.clockwise", "counterclockwise",
    ]
    assert artifacts.submission.loc[artifacts.submission["initial_state_id"] == 7, "path"].item() == "counterclockwise"
    assert len(artifacts.combined_logs) == 2
    assert len(artifacts.rank_logs) == 2


def test_only_prevalidates_source_before_gpu_and_never_runs_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": ["clockwise.clockwise"]}).to_csv(source, index=False)
    variants: list[str] = []

    def execute(invocation: runner.RunnerInvocation, extra_env: dict[str, str]) -> runner.InvocationExecution:
        variants.append(invocation.variant)
        return _successful_execution(invocation, "solution_path=clockwise\n")

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, reflect_mode="only", source=source), _contract(), _model(),
        _plan(local_beam=128), tmp_path / "weights", tmp_path / "artifacts",
    )

    assert variants == ["reflected"]
    assert artifacts.solution_records[0].original_oriented_path == "counterclockwise"


def test_invalid_reflection_source_fails_before_any_gpu_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": ["unknown"]}).to_csv(source, index=False)
    launches = 0

    def execute(invocation: runner.RunnerInvocation, extra_env: dict[str, str]) -> runner.InvocationExecution:
        nonlocal launches
        launches += 1
        raise AssertionError("must not launch")

    monkeypatch.setattr(runner, "_run_one", execute)
    with pytest.raises(ValueError, match="source.*puzzle 7"):
        runner.run_public_search(
            _config(tmp_path, reflect_mode="only", source=source), _contract(), _model(),
            _plan(local_beam=128), tmp_path / "weights", tmp_path / "artifacts",
        )
    assert launches == 0


def test_nonzero_rank_group_exit_is_hard_failure_with_prior_artifacts_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def execute(invocation: runner.RunnerInvocation, extra_env: dict[str, str]) -> runner.InvocationExecution:
        nonlocal calls
        calls += 1
        if calls == 1:
            return _successful_execution(invocation, "solution_path=counterclockwise\n")
        invocation.combined_log.write_text("solution_path=counterclockwise\nfatal\n", encoding="utf-8")
        for rank_log in invocation.rank_logs:
            rank_log.write_text("fatal\n", encoding="utf-8")
        return runner.InvocationExecution(invocation, 9, 0.5, "solution_path=counterclockwise\nfatal\n")

    monkeypatch.setattr(runner, "_run_one", execute)
    with pytest.raises(runner.PublicSearchRunError, match="exit code 9") as caught:
        runner.run_public_search(
            _config(tmp_path, end=8), _contract(two_puzzles=True), _model(), _plan(local_beam=128),
            tmp_path / "weights", tmp_path / "artifacts",
        )

    partial = caught.value.partial_artifacts
    assert [record.puzzle_id for record in partial.solution_records] == [7]
    assert partial.return_codes == (0, 9)
    assert len(partial.combined_logs) == 2
    assert len(partial.rank_logs) == 2


def test_final_records_use_semantic_solution_deduplication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    def execute(invocation: runner.RunnerInvocation, extra_env: dict[str, str]) -> runner.InvocationExecution:
        assert invocation.result_tsv is not None
        invocation.result_tsv.write_text(
            "puzzle_id\tdepth_index\tfound_depth\ttotal_depth\tknown_length\tdelta\towner_rank\tsolution_path\n"
            "7\t0\t1\t1\t0\t0\t0\tcounterclockwise\n"
            "7\t0\t1\t1\t0\t0\t1\tcounterclockwise\n",
            encoding="utf-8",
        )
        return _successful_execution(invocation, "collection_status=depth_reached\n")

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, solution_mode="collect"), _contract(), _model(), _plan(local_beam=128),
        tmp_path / "weights", tmp_path / "artifacts",
    )

    assert [record.original_oriented_path for record in artifacts.solution_records] == ["counterclockwise"]
    assert artifacts.submission.loc[artifacts.submission["initial_state_id"] == 7, "path"].item() == "counterclockwise"


def test_rank0_output_streams_live_and_keeps_bounded_parser_tail(tmp_path: Path) -> None:
    class FakeProcess:
        stdout = iter(["rank0-start\n", "solution_path=counterclockwise\n"])

        @staticmethod
        def wait() -> int:
            return 0

    live = io.StringIO()
    combined = tmp_path / "combined.log"
    return_code, tail = runner.stream_process_output(
        FakeProcess(), combined, live_stream=live, max_tail_chars=24,
    )

    assert return_code == 0
    assert live.getvalue() == "rank0-start\nsolution_path=counterclockwise\n"
    assert combined.read_text(encoding="utf-8") == live.getvalue()
    assert tail == "n_path=counterclockwise\n"


@pytest.mark.parametrize(
    ("failure_stage", "expected_error"),
    (("popen", OSError), ("stream", RuntimeError), ("log_capture", runner.InvocationLogCaptureError)),
)
def test_history_cleanup_runs_on_popen_stream_and_log_capture_exceptions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_error: type[Exception],
) -> None:
    history_root = tmp_path / "beam_history_public"
    monkeypatch.setattr(runner, "_HISTORY_ROOT", runner.PurePosixPath(history_root.as_posix()))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3))
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 2, 7, "original",
        tmp_path / "weights", tmp_path / "artifacts",
    )

    class BrokenStream:
        def __iter__(self) -> "BrokenStream":
            return self

        def __next__(self) -> str:
            raise RuntimeError("stream failed")

    def popen(*_: object, env: dict[str, str], **__: object) -> SimpleNamespace:
        history_dir = Path(env["BEAM_HISTORY_DIR"])
        history_dir.mkdir(parents=True)
        (history_dir / "rank_0_history_static_arena.bin").write_bytes(b"arena")
        if failure_stage == "popen":
            raise OSError("spawn failed")
        stdout: object = BrokenStream() if failure_stage == "stream" else iter(["rank0\n"])
        return SimpleNamespace(stdout=stdout, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(expected_error):
        runner._run_one(invocation, {})

    assert not invocation.history_dir.exists()


def test_sequential_success_and_failure_cleanup_only_their_unique_history_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    return_codes = iter((0, 9))

    def popen(command: tuple[str, ...], *, env: dict[str, str], **_: object) -> SimpleNamespace:
        history_dir = Path(env["BEAM_HISTORY_DIR"])
        history_dir.mkdir(parents=True)
        for rank in (0, 1):
            (history_dir / f"rank_{rank}_history_static_arena.bin").write_bytes(b"arena")
        log_arg = next(part for part in command if str(part).startswith("--log-dir="))
        log_dir = Path(str(log_arg).split("=", 1)[1])
        for rank in (0, 1):
            rank_dir = log_dir / "run" / "attempt_0" / str(rank)
            rank_dir.mkdir(parents=True)
            (rank_dir / "stdout.log").write_text(f"rank {rank}\n", encoding="utf-8")
            (rank_dir / "stderr.log").write_text("", encoding="utf-8")
        return_code = next(return_codes)
        return SimpleNamespace(stdout=iter([f"return_code={return_code}\n"]), wait=lambda: return_code)

    history_root = tmp_path / "beam_history_public"
    monkeypatch.setattr(runner, "_HISTORY_ROOT", runner.PurePosixPath(history_root.as_posix()))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3))
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    sentinel = tmp_path / "history-sentinel"
    sentinel.write_text("keep", encoding="utf-8")
    try:
        invocations = [
            runner.build_runner_invocation(
                _config(tmp_path), _plan(local_beam=128), 2, puzzle_id, "original",
                tmp_path / "weights", tmp_path / "artifacts",
            )
            for puzzle_id in (7, 8)
        ]
        executions = [runner._run_one(invocation, {}) for invocation in invocations]

        assert [execution.return_code for execution in executions] == [0, 9]
        assert all(not invocation.history_dir.exists() for invocation in invocations)
        assert sentinel.read_text(encoding="utf-8") == "keep"
    finally:
        sentinel.unlink(missing_ok=True)


def test_missing_real_torchrun_logs_become_hard_run_error_with_combined_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailedProcess:
        stdout = iter(["fatal before rank redirect\n"])

        @staticmethod
        def wait() -> int:
            return 9

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: FailedProcess())

    with pytest.raises(runner.PublicSearchRunError, match="exit code 9.*log capture") as caught:
        runner.run_public_search(
            _config(tmp_path), _contract(), _model(), _plan(local_beam=128),
            tmp_path / "weights", tmp_path / "artifacts",
        )

    partial = caught.value.partial_artifacts
    assert partial.return_codes == (9,)
    assert partial.combined_logs[0].read_text(encoding="utf-8") == "fatal before rank redirect\n"
    assert all(not path.exists() for path in partial.rank_logs[0])


def test_collection_stop_contract_syncs_host_reason_and_explicit_depth_beats_legacy() -> None:
    source = Path("tools/production_runner.cu").read_text(encoding="utf-8")

    assert "propagate_stop_flag(memory, streams, nccl_runtime.comm, world_size)" in source
    assert "solve_bucket_stop_depth == 0U" in source
    assert "bucket_global_stop_reason" in source
    assert "bucket_global_stop_reason == 2U" in source
    assert "solve_bucket_gather_records_per_chunk" in source
    assert "global_stored" in source
    assert "collection_status=depth_reached" in source
    assert "solve bucket overflow: increase BEAM_SOLVED_RESULT_CAPACITY" in source
