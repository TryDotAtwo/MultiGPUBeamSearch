from __future__ import annotations

from pathlib import Path
import hashlib
import io
import os
import subprocess
import sys
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


def test_child_environment_drops_inherited_beam_and_torchrun_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_root = tmp_path / "beam_history_public"
    monkeypatch.setattr(runner, "_HISTORY_ROOT", runner.PurePosixPath(history_root.as_posix()))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3))
    poison = {
        "BEAM_REPAIR_SOLUTIONS_CSV": "repair.csv",
        "BEAM_START_STATE_TEXT": "poison-start",
        "BEAM_TARGET_STATE_FILE": "poison-target",
        "BEAM_RING_COUNT": "999",
        "BEAM_B_MICRO": "poison-manual",
        "RANK": "8",
        "LOCAL_RANK": "7",
        "WORLD_SIZE": "16",
        "LOCAL_WORLD_SIZE": "8",
        "GROUP_RANK": "5",
        "ROLE_RANK": "4",
        "GROUP_WORLD_SIZE": "3",
        "ROLE_WORLD_SIZE": "2",
        "ROLE_NAME": "poison-role",
        "MASTER_ADDR": "poison-master",
        "MASTER_PORT": "1",
        "TORCHELASTIC_RESTART_COUNT": "9",
        "TORCHELASTIC_MAX_RESTARTS": "9",
        "TORCHELASTIC_RUN_ID": "poison-run",
        "TORCHELASTIC_USE_AGENT_STORE": "1",
        "TORCHELASTIC_ERROR_FILE": "poison-error",
        "TORCHELASTIC_FUTURE_RESERVED": "poison-future",
        "TORCH_NCCL_ASYNC_ERROR_HANDLING": "9",
    }
    for name, value in poison.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    original_path = os.environ.get("PATH", "")
    captured: dict[str, str] = {}

    def popen(command: tuple[str, ...], *, env: dict[str, str], **_: object) -> SimpleNamespace:
        captured.update(env)
        log_arg = next(part for part in command if str(part).startswith("--log-dir="))
        log_dir = Path(str(log_arg).split("=", 1)[1])
        for rank in (0, 1):
            rank_dir = log_dir / "run" / "attempt_0" / str(rank)
            rank_dir.mkdir(parents=True)
            (rank_dir / "stdout.log").write_text("", encoding="utf-8")
            (rank_dir / "stderr.log").write_text("", encoding="utf-8")
        return SimpleNamespace(stdout=iter(["puzzle_solved=0\n"]), wait=lambda *args, **kwargs: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 2, 7, "original",
        tmp_path / "weights", tmp_path / "artifacts",
    )
    runner._run_one(invocation)

    assert captured["PATH"] == original_path
    assert captured["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert captured["BEAM_B_MICRO"] == invocation.env["BEAM_B_MICRO"]
    assert not ({name for name in poison if name.startswith("BEAM_")} - set(invocation.env)).intersection(captured)
    assert not ({name for name in poison if not name.startswith("BEAM_")}).intersection(captured)


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


def test_collection_batch_limit_is_bounded_by_remaining_unique_target() -> None:
    assert runner.derive_collection_batch_limit(max_solutions=3, accepted_count=0) == 3
    assert runner.derive_collection_batch_limit(max_solutions=3, accepted_count=1) == 2
    assert runner.derive_collection_batch_limit(max_solutions=3, accepted_count=3) == 0
    assert runner.derive_collection_batch_limit(max_solutions=0, accepted_count=9) == 65_536


def test_large_collection_plan_never_allocates_capacity_sized_host_metadata() -> None:
    plan = runner.derive_gather_chunk_plan(
        frontier_states=2**20, capacity=2**21 * 24, world_size=2,
    )

    assert plan.capacity == 2**21 * 24
    assert plan.records_per_chunk == 65_536
    assert plan.chunk_count == 768
    assert runner.derive_collection_batch_limit(max_solutions=3, accepted_count=0) == 3


def test_collection_cursor_model_advances_over_duplicate_first_batch() -> None:
    records = [
        ((4, 0, 1, 10, 1, 1, 1, 0), "same"),
        ((4, 0, 2, 10, 1, 1, 1, 0), "same"),
        ((4, 1, 1, 10, 1, 1, 1, 0), "unique-b"),
        ((5, 0, 1, 11, 1, 1, 1, 0), "unique-c"),
    ]
    accepted: set[str] = set()
    cursor: tuple[int, ...] | None = None
    batch_sizes: list[int] = []
    while len(accepted) < 3:
        remaining = runner.derive_collection_batch_limit(3, len(accepted))
        batch = [(key, path) for key, path in sorted(records) if cursor is None or key > cursor][:remaining]
        if not batch:
            break
        batch_sizes.append(len(batch))
        accepted.update(path for _, path in batch)
        cursor = batch[-1][0]

    assert batch_sizes == [3, 1]
    assert accepted == {"same", "unique-b", "unique-c"}
    assert sorted(records)[0][0][2] < sorted(records)[1][0][2]  # found_depth participates in the key.


def test_collection_cuda_source_uses_header_only_bounded_cursor_batches() -> None:
    source = Path("tools/production_runner.cu").read_text(encoding="utf-8")
    gather = source.split("SolveBucketRecordBatch gather_solve_bucket_record_batch_distributed", 1)[1]
    gather = gather.split("std::uint32_t propagate_solved_flag", 1)[0]
    comparator = source.split("struct SolveBucketRecordLess", 1)[1]
    comparator = comparator.split("};", 1)[0]

    assert "read_solved_snapshot_header" in source
    assert "memory.solved_meta_list + base" in gather
    assert "memory.solved_depth_list + base" in gather
    assert "memory.solved_suffix_list + base" in gather
    assert "std::vector<SolveBucketRecord> records;" not in gather
    assert "max_selection_batch_records = 65'536ULL" in gather
    assert "selected_records.erase(std::prev(selected_records.end()))" in gather
    comparator_fields = [
        "a.total_depth", "a.owner_rank", "a.found_depth", "a.meta.parent_idx",
        "a.meta.route_packed", "a.meta.hash.lo", "a.meta.hash.hi", "a.suffix_id",
    ]
    positions = [comparator.index(field) for field in comparator_fields]
    assert positions == sorted(positions)
    assert "synchronize_rank0_accepted_count" in source
    assert "scan_cursor = batch.records.back()" in source
    scan_loop = source.split("std::optional<SolveBucketRecord> scan_cursor", 1)[1]
    scan_loop = scan_loop.split("reset_solved_buffers(memory)", 1)[0]
    assert scan_loop.index("synchronized_accepted_count = synchronize_rank0_accepted_count") < scan_loop.index(
        "gather_solve_bucket_record_batch_distributed"
    )

def test_release_first_mode_fixture_parses_without_debug_solution_path(tmp_path: Path) -> None:
    fixture = Path("tests/cayleypy_public/fixtures/fake_production_runner.py")
    environment = os.environ.copy()
    environment.pop("BEAM_SOLVE_BUCKET_RESULT_TSV", None)
    completed = subprocess.run(
        [sys.executable, str(fixture)],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )

    assert "solution_path=" not in completed.stdout
    parsed = runner.parse_runner_output(completed.stdout, None, 7, "original")
    assert parsed.collection_status == "first_solution"
    assert [(record.path, record.found_depth, record.touch_depth) for record in parsed.records] == [
        ("counterclockwise", 1, 0),
    ]


def test_release_first_mode_parser_supports_empty_and_legacy_lines() -> None:
    empty = runner.parse_runner_output(
        "puzzle_solved=1 puzzle_id=7 seconds=0 solution_length=0 "
        "found_depth=0 touch_depth=0 solution=\n",
        None,
        7,
        "original",
    )
    legacy = runner.parse_runner_output(
        "puzzle_solved=1 puzzle_id=7 seconds=1.5 solution_length=1 solution=counterclockwise\n",
        None,
        7,
        "original",
    )

    assert [(record.path, record.found_depth, record.touch_depth) for record in empty.records] == [("", 0, 0)]
    assert [(record.path, record.found_depth, record.touch_depth) for record in legacy.records] == [
        ("counterclockwise", 1, 0),
    ]


@pytest.mark.parametrize(
    ("line", "message"),
    (
        (
            "puzzle_solved=1 puzzle_id=8 seconds=1 solution_length=1 "
            "found_depth=1 touch_depth=0 solution=counterclockwise\n",
            "puzzle id",
        ),
        (
            "puzzle_solved=1 puzzle_id=7 seconds=1 solution_length=2 "
            "found_depth=1 touch_depth=1 solution=counterclockwise\n",
            "length",
        ),
    ),
)
def test_release_first_mode_parser_rejects_wrong_puzzle_or_length(line: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        runner.parse_runner_output(line, None, 7, "original")


def test_debug_solution_path_compatibility_is_anchored_to_a_complete_line() -> None:
    unrelated = runner.parse_runner_output(
        "[default0]:message=do not parse solution_path=counterclockwise as a result\n",
        None,
        7,
        "original",
    )
    compatible = runner.parse_runner_output("solution_path=counterclockwise\n", None, 7, "original")
    tee_compatible = runner.parse_runner_output(
        "[default0]:solution_path=counterclockwise\n", None, 7, "original"
    )

    assert unrelated.records == ()
    assert [record.path for record in compatible.records] == ["counterclockwise"]
    assert [record.path for record in tee_compatible.records] == ["counterclockwise"]


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

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
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


def test_only_uses_validated_external_source_when_reflected_search_has_no_hit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": ["counterclockwise"]}).to_csv(source, index=False)

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
        return _successful_execution(
            invocation,
            "puzzle_solved=0 puzzle_id=7 seconds=0.1 solution_length=-1 solution=\n",
        )

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, reflect_mode="only", source=source), _contract(), _model(),
        _plan(local_beam=128), tmp_path / "weights", tmp_path / "artifacts",
    )

    assert [(record.variant, record.original_oriented_path) for record in artifacts.solution_records] == [
        ("source", "counterclockwise"),
    ]
    assert artifacts.solution_records[0].source_solution_sha256 == hashlib.sha256(
        b"counterclockwise"
    ).hexdigest()
    assert artifacts.submission.loc[
        artifacts.submission["initial_state_id"] == 7, "path"
    ].item() == "counterclockwise"


def test_discovered_reflected_duplicate_retains_solver_provenance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": ["counterclockwise"]}).to_csv(source, index=False)

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
        return _successful_execution(invocation, "solution_path=clockwise\n")

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, reflect_mode="only", source=source), _contract(), _model(),
        _plan(local_beam=128), tmp_path / "weights", tmp_path / "artifacts",
    )

    assert [(record.variant, record.original_oriented_path) for record in artifacts.solution_records] == [
        ("reflected", "counterclockwise"),
    ]

def test_short_external_source_beats_longer_reflected_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": ["counterclockwise"]}).to_csv(source, index=False)

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
        return _successful_execution(
            invocation,
            "solution_path=clockwise.clockwise.clockwise.clockwise\n",
        )

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, reflect_mode="only", source=source), _contract(), _model(),
        _plan(local_beam=128), tmp_path / "weights", tmp_path / "artifacts",
    )

    assert {record.variant for record in artifacts.solution_records} == {"source", "reflected"}
    assert artifacts.submission.loc[
        artifacts.submission["initial_state_id"] == 7, "path"
    ].item() == "counterclockwise"


def test_after_original_uses_deterministic_union_of_external_and_discovered_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    external_path = "counterclockwise"
    discovered_path = ".".join(["counterclockwise"] * 4)
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": [external_path]}).to_csv(source, index=False)
    calls: list[runner.RunnerInvocation] = []

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
        calls.append(invocation)
        output = f"solution_path={discovered_path}\n" if invocation.variant == "original" else "no-hit\n"
        return _successful_execution(invocation, output)

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, reflect_mode="after_original", source=source), _contract(), _model(),
        _plan(local_beam=128), tmp_path / "weights", tmp_path / "artifacts",
    )

    assert [call.variant for call in calls] == ["original", "reflected", "reflected"]
    assert [call.source_solution_sha256 for call in calls[1:]] == [
        hashlib.sha256(external_path.encode("utf-8")).hexdigest(),
        hashlib.sha256(discovered_path.encode("utf-8")).hexdigest(),
    ]
    assert {(record.variant, record.original_oriented_path) for record in artifacts.solution_records} == {
        ("source", external_path),
        ("original", discovered_path),
    }
    assert artifacts.submission.loc[
        artifacts.submission["initial_state_id"] == 7, "path"
    ].item() == external_path


def test_reflection_inverse_closure_fails_before_any_subprocess(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = PuzzleContract(
        central_state=(0, 1, 2),
        generators={"clockwise": (1, 2, 0)},
        initial_states={7: (1, 2, 0)},
        sample_submission=pd.DataFrame({"initial_state_id": [7], "path": ["old"]}),
        state_len=3,
        num_classes=3,
    )
    launches = 0

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
        nonlocal launches
        launches += 1
        return _successful_execution(invocation, "no-hit\n")

    monkeypatch.setattr(runner, "_run_one", execute)
    with pytest.raises(ValueError, match="unique inverse"):
        runner.run_public_search(
            _config(tmp_path, reflect_mode="after_original"), broken, _model(), _plan(local_beam=128),
            tmp_path / "weights", tmp_path / "artifacts",
        )
    assert launches == 0


def test_only_prevalidates_source_before_gpu_and_never_runs_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": ["clockwise.clockwise"]}).to_csv(source, index=False)
    variants: list[str] = []

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
        variants.append(invocation.variant)
        return _successful_execution(invocation, "solution_path=clockwise\n")

    monkeypatch.setattr(runner, "_run_one", execute)
    artifacts = runner.run_public_search(
        _config(tmp_path, reflect_mode="only", source=source), _contract(), _model(),
        _plan(local_beam=128), tmp_path / "weights", tmp_path / "artifacts",
    )

    assert variants == ["reflected"]
    assert [
        (record.variant, record.original_oriented_path)
        for record in artifacts.solution_records
    ] == [
        ("source", "clockwise.clockwise"),
        ("reflected", "counterclockwise"),
    ]


def test_invalid_reflection_source_fails_before_any_gpu_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.csv"
    pd.DataFrame({"initial_state_id": [7], "path": ["unknown"]}).to_csv(source, index=False)
    launches = 0

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
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

    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
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
    def execute(invocation: runner.RunnerInvocation) -> runner.InvocationExecution:
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
        runner._run_one(invocation)

    assert not invocation.history_dir.exists()


def test_stream_failure_stops_process_before_cleanup_and_retains_partial_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_root = tmp_path / "beam_history_public"
    monkeypatch.setattr(runner, "_HISTORY_ROOT", runner.PurePosixPath(history_root.as_posix()))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3))
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 2, 7, "original",
        tmp_path / "weights", tmp_path / "artifacts",
    )
    events: list[str] = []

    class BrokenAfterOne:
        def __init__(self) -> None:
            self.first = True

        def __iter__(self) -> "BrokenAfterOne":
            return self

        def __next__(self) -> str:
            if self.first:
                self.first = False
                return "partial rank0\n"
            raise RuntimeError("stream failed")

    class LiveProcess:
        stdout = BrokenAfterOne()

        def __init__(self) -> None:
            self.alive = True
            self.return_code = 143

        def poll(self) -> int | None:
            return None if self.alive else self.return_code

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            events.append("wait")
            self.alive = False
            return self.return_code

        def kill(self) -> None:
            events.append("kill")

    process = LiveProcess()

    def popen(*_: object, env: dict[str, str], **__: object) -> LiveProcess:
        Path(env["BEAM_HISTORY_DIR"]).mkdir(parents=True)
        rank_dir = invocation.torchrun_log_dir / "run" / "attempt_0" / "0"
        rank_dir.mkdir(parents=True)
        (rank_dir / "stdout.log").write_text("rank0 partial\n", encoding="utf-8")
        (rank_dir / "stderr.log").write_text("", encoding="utf-8")
        return process

    real_cleanup = runner.cleanup_history_runtime

    def cleanup(path: Path) -> None:
        assert not process.alive
        events.append("cleanup")
        real_cleanup(path)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "cleanup_history_runtime", cleanup)
    with pytest.raises(RuntimeError, match="stream failed") as caught:
        runner._run_one(invocation)

    assert caught.value.execution.return_code == 143
    assert events == ["terminate", "wait", "cleanup"]
    assert invocation.combined_log.read_text(encoding="utf-8") == "partial rank0\n"
    assert invocation.rank_logs[0].read_text(encoding="utf-8") == "[stdout]\nrank0 partial\n[stderr]\n"
    assert not invocation.rank_logs[1].exists()
    assert not invocation.history_dir.exists()


def test_stream_failure_uses_bounded_kill_fallback_before_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_root = tmp_path / "beam_history_public"
    monkeypatch.setattr(runner, "_HISTORY_ROOT", runner.PurePosixPath(history_root.as_posix()))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3))
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 2, 7, "original",
        tmp_path / "weights", tmp_path / "artifacts",
    )
    events: list[str] = []

    class LiveProcess:
        stdout = iter(())

        def __init__(self) -> None:
            self.killed = False
            self.alive = True

        def poll(self) -> int | None:
            return None if self.alive else 137

        def terminate(self) -> None:
            events.append("terminate")

        def wait(self, timeout: float | None = None) -> int:
            assert timeout is not None
            events.append(f"wait:{timeout}")
            if not self.killed:
                raise subprocess.TimeoutExpired("torchrun", timeout)
            self.alive = False
            return 137

        def kill(self) -> None:
            events.append("kill")
            self.killed = True

    process = LiveProcess()

    def popen(*_: object, env: dict[str, str], **__: object) -> LiveProcess:
        Path(env["BEAM_HISTORY_DIR"]).mkdir(parents=True)
        return process

    def fail_stream(*_: object, **__: object) -> tuple[int, str]:
        raise RuntimeError("stream failed")

    real_cleanup = runner.cleanup_history_runtime

    def cleanup(path: Path) -> None:
        assert not process.alive
        events.append("cleanup")
        real_cleanup(path)

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    monkeypatch.setattr(runner, "stream_process_output", fail_stream)
    monkeypatch.setattr(runner, "cleanup_history_runtime", cleanup)
    with pytest.raises(RuntimeError, match="stream failed"):
        runner._run_one(invocation)

    assert events == ["terminate", "wait:5.0", "kill", "wait:5.0", "cleanup"]


def test_cleanup_failure_adds_note_without_masking_stream_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocation = runner.build_runner_invocation(
        _config(tmp_path), _plan(local_beam=128), 2, 7, "original",
        tmp_path / "weights", tmp_path / "artifacts",
    )

    class LiveProcess:
        stdout = iter(())
        alive = True

        def poll(self) -> int | None:
            return None if self.alive else 143

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            self.alive = False
            return 143

        def kill(self) -> None:
            raise AssertionError("kill not expected")

    def fail_stream(*_: object, **__: object) -> tuple[int, str]:
        raise RuntimeError("primary stream failure")

    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: LiveProcess())
    monkeypatch.setattr(runner, "stream_process_output", fail_stream)
    monkeypatch.setattr(
        runner, "cleanup_history_runtime", lambda _: (_ for _ in ()).throw(OSError("cleanup failed")),
    )

    with pytest.raises(RuntimeError, match="primary stream failure") as caught:
        runner._run_one(invocation)

    assert type(caught.value) is RuntimeError
    assert any("history scratch cleanup failed: cleanup failed" in note for note in caught.value.__notes__)


def test_run_search_reports_partial_artifacts_after_stream_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    history_root = tmp_path / "beam_history_public"
    monkeypatch.setattr(runner, "_HISTORY_ROOT", runner.PurePosixPath(history_root.as_posix()))
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda _: SimpleNamespace(free=32 * 1024**3))

    class BrokenStream:
        def __iter__(self) -> "BrokenStream":
            return self

        def __next__(self) -> str:
            raise RuntimeError("stream failed")

    class LiveProcess:
        stdout = BrokenStream()
        alive = True

        def poll(self) -> int | None:
            return None if self.alive else 143

        def terminate(self) -> None:
            pass

        def wait(self, timeout: float | None = None) -> int:
            self.alive = False
            return 143

        def kill(self) -> None:
            raise AssertionError("kill not expected")

    def popen(*_: object, env: dict[str, str], **__: object) -> LiveProcess:
        Path(env["BEAM_HISTORY_DIR"]).mkdir(parents=True)
        return LiveProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    with pytest.raises(runner.PublicSearchRunError, match="stream failed") as caught:
        runner.run_public_search(
            _config(tmp_path), _contract(), _model(), _plan(local_beam=128),
            tmp_path / "weights", tmp_path / "artifacts",
        )

    partial = caught.value.partial_artifacts
    assert partial.return_codes == (143,)
    assert len(partial.combined_logs) == 1
    assert partial.combined_logs[0].exists()

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
        executions = [runner._run_one(invocation) for invocation in invocations]

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


def test_collection_exchanges_only_rank_local_next_k_candidates() -> None:
    source = Path("tools/production_runner.cu").read_text(encoding="utf-8")
    gather = source.split("SolveBucketRecordBatch gather_solve_bucket_record_batch_distributed", 1)[1]
    gather = gather.split("std::uint32_t propagate_solved_flag", 1)[0]
    local_scan = gather.split("std::set<SolveBucketRecord, SolveBucketRecordLess> local_selected_records", 1)[1]
    local_scan = local_scan.split("// Exactness:", 1)[0]
    exchange = gather.split("// Exactness:", 1)[1]

    assert "ncclAllGather" not in local_scan
    assert "local_selected_records.size() > batch_limit" in local_scan
    assert "global next-K" in gather and "rank-local next-K" in gather
    assert "base < batch_limit" in exchange
    assert "local_selected_vector" in exchange
    assert "ncclAllGather" in exchange
    assert "global_stored" not in gather

def test_collection_rank0_processing_errors_are_synchronized_before_next_reconstruction() -> None:
    source = Path("tools/production_runner.cu").read_text(encoding="utf-8")
    record_loop = source.split("for (SolveBucketRecord record : batch.records)", 1)[1]
    record_loop = record_loop.split("scan_cursor = batch.records.back()", 1)[0]

    rank0_processing = record_loop.split(
        "if (rank == 0U && !rank0_processing_exception)", 1
    )[1].split("const std::uint32_t global_processing_error", 1)[0]
    try_pos = rank0_processing.index("try {")
    validation_pos = rank0_processing.index("solve bucket CPU solution validation failed")
    path_pos = rank0_processing.index("moves_to_path_text")
    catch_pos = rank0_processing.index("catch (...)")
    sync_pos = record_loop.index("global_processing_error = propagate_host_stop_value")
    rethrow_pos = record_loop.index("std::rethrow_exception(rank0_processing_exception)")
    assert try_pos < validation_pos < catch_pos
    assert try_pos < path_pos < catch_pos
    assert record_loop.index("rank0_processing_exception = std::current_exception()") < sync_pos < rethrow_pos
    assert "continue;" not in record_loop

def test_collection_stop_contract_syncs_host_reason_and_explicit_depth_beats_legacy() -> None:
    source = Path("tools/production_runner.cu").read_text(encoding="utf-8")

    assert "propagate_stop_flag(memory, streams, nccl_runtime.comm, world_size)" in source
    assert "solve_bucket_stop_depth == 0U" in source
    assert "bucket_global_stop_reason" in source
    assert "bucket_global_stop_reason == 2U" in source
    assert "solve_bucket_gather_records_per_chunk" in source
    assert "local_selected_records" in source
    assert "collection_status=depth_reached" in source
    assert "solve bucket overflow: increase BEAM_SOLVED_RESULT_CAPACITY" in source
