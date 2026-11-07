"""Actual CPU subprocess fixtures validate protocol only, never native CUDA."""
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from cayleypy_native.backend import child_environment, parse_terminal, run_process, runtime_devices
from cayleypy_native.errors import NativeBackendError, NativeUnavailable
from cayleypy_native.options import NativeOptions


class ReplayContract:
    move_count = 2

    def replay(self, path):
        return tuple(path) == (1,)


def invoke_fixture(tmp_path, text, *, code=0):
    """Explicitly fake producer, run by the real subprocess lifetime helper."""
    log = tmp_path / "fake-native.log"
    script = "import sys; print(" + repr(text) + "); sys.exit(" + str(code) + ")"
    run_process([sys.executable, "-c", script], cwd=tmp_path, env=dict(os.environ), timeout=10, log_path=log)
    return log


def test_real_subprocess_valid_path_parses_and_replays(tmp_path):
    log = invoke_fixture(tmp_path, "GLOBAL_BEAM_WIDTH_EFFECTIVE=1024\nB_MICRO=682\nSTREAM1_ROWS_PER_JOB=2046\nSTREAM3_BATCH_CANDIDATES=2046\npuzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 found_depth=1 touch_depth=0 solution=m1")
    path, width, metadata = parse_terminal(log, ReplayContract())
    assert (path, width) == ((1,), 1024)
    assert metadata["native_seconds"] == 0.1
    assert metadata["observed_parent_batch"] == 682
    assert metadata["observed_inference_rows"] == 2046
    assert json.loads(log.with_suffix(".log.command.json").read_text())[0] == sys.executable


def test_real_subprocess_not_found_is_distinct_from_crash(tmp_path):
    log = invoke_fixture(tmp_path, "puzzle_solved=0 puzzle_id=0 seconds=0.2 solution_length=-1 solution=")
    assert parse_terminal(log, ReplayContract())[:2] == (None, None)
    assert parse_terminal(log, ReplayContract())[2]["observed_parent_batch"] is None
    with pytest.raises(NativeBackendError, match="rc=7"):
        invoke_fixture(tmp_path, "puzzle_solved=0 puzzle_id=0 seconds=0 solution_length=-1 solution=", code=7)


@pytest.mark.parametrize("record,reason", [
    ("startup only", "without a terminal"),
    ("puzzle_solved=1 bad", "malformed"),
    ("puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m0", "replay"),
    ("puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=2 solution=m1", "length"),
    ("puzzle_solved=1 puzzle_id=0 seconds=nan solution_length=1 solution=m1", "time"),
    ("puzzle_solved=1 puzzle_id=0 seconds=bad solution_length=1 solution=m1", "time"),
    ("puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=weird-name", "synthetic"),
    ("puzzle_solved=1 puzzle_id=0 seconds=0 solution_length=1 solution=m3", "IDs"),
    ("puzzle_solved=0 puzzle_id=0 seconds=0 solution_length=-1 solution=m1", "inconsistent"),
])
def test_bad_protocol_never_looks_like_not_found(tmp_path, record, reason):
    log = invoke_fixture(tmp_path, record)
    with pytest.raises(NativeBackendError, match=reason):
        parse_terminal(log, ReplayContract())


def test_timeout_terminates_real_process(tmp_path):
    with pytest.raises(NativeBackendError, match="timed out"):
        run_process([sys.executable, "-c", "import time; print('fake waiting', flush=True); time.sleep(30)"],
                    cwd=tmp_path, env=dict(os.environ), timeout=0.2, log_path=tmp_path / "timeout.log")


def test_child_environment_isolated_and_maps_visible_indices(tmp_path):
    original = {"PATH": "safe", "BEAM_REPAIR_SOLUTIONS_CSV": "unrelated.csv", "BEAM_WEIGHT_DIR": "wrong",
                "WORLD_SIZE": "8", "RANK": "3", "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c",
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_LAUNCH_BLOCKING": "1", "NCCL_DEBUG": "TRACE",
                "STREAM4_BATCH_ALIGNMENT": "999", "TORCHELASTIC_RUN_ID": "inherited", "PUBLISH_RESULTS": "1"}
    env = child_environment((2, 0), tmp_path, original)
    assert env["CUDA_VISIBLE_DEVICES"] == "GPU-c,GPU-a"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
    assert env["BEAM_NCCL_ID_FILE"] == str(tmp_path / "nccl-id.bin")
    assert not any(k in env for k in ("RANK", "WORLD_SIZE", "BEAM_WEIGHT_DIR", "BEAM_REPAIR_SOLUTIONS_CSV",
                                     "CUDA_LAUNCH_BLOCKING", "NCCL_DEBUG", "STREAM4_BATCH_ALIGNMENT", "PUBLISH_RESULTS"))
    assert original["WORLD_SIZE"] == "8"


@pytest.mark.parametrize("moves,outputs,world,beam", [
    (3, 1, 2, 7), (24, 24, 2, 7), (24, 1, 2, 7), (256, 256, 8, 1), (3, 1, 1, 32768),
])
def test_small_beam_microbatch_fits_native_auto_storage(moves, outputs, world, beam):
    from cayleypy_native.backend import microbatch_environment
    env, metadata = microbatch_environment(SimpleNamespace(move_count=moves),
        SimpleNamespace(manifest={"output_dim": outputs}), beam, world)
    # Baseline native's independent rejection condition (runtime_config.cpp:420)
    # is Stream3's one-slot child count > physical shard capacity.
    rows_per_parent = moves if outputs == 1 else 1
    configured_rows = int(env.get("BEAM_B_MICRO", "8192"))
    assert 1 <= configured_rows <= 8192
    assert configured_rows // rows_per_parent * moves <= metadata["reference_shard_storage"]
    assert set(env) <= {"BEAM_B_MICRO"}  # no semantic beam/shard controls
    if (moves, outputs, world, beam) == (3, 1, 2, 7):
        assert 8192 // 3 * 3 == 8190 > metadata["reference_shard_storage"] == 2048
        assert metadata["derived_parent_batch"] == 682
        assert env == {"BEAM_B_MICRO": "2046"}


@pytest.mark.parametrize("moves,outputs,beam", [(3, 1, 65536), (24, 1, 65536), (24, 24, 2**24)])
def test_fitting_beam_keeps_native_default_microbatch(moves, outputs, beam):
    from cayleypy_native.backend import microbatch_environment
    env, metadata = microbatch_environment(SimpleNamespace(move_count=moves),
        SimpleNamespace(manifest={"output_dim": outputs}), beam, 2)
    assert env == {}
    assert metadata["configured_row_budget"] == 8192
    assert not metadata["adjusted_for_small_beam"]


def test_runtime_unavailable_before_process_creation(monkeypatch, tmp_path):
    import cayleypy_native.backend as backend
    monkeypatch.setattr(backend.platform, "system", lambda: "Windows")
    with pytest.raises(NativeUnavailable, match="Linux"):
        runtime_devices(object(), NativeOptions(cache_dir=tmp_path))
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(NativeUnavailable, match="torchrun"):
        runtime_devices(object(), NativeOptions(cache_dir=tmp_path))


def test_runtime_devices_supports_pypi_graph_device_and_explicit_override(monkeypatch, tmp_path):
    import torch
    import cayleypy_native.backend as backend
    import cayleypy_native.build as build
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(build, "prerequisites", lambda options: None)
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(key, raising=False)
    options = NativeOptions(cache_dir=tmp_path)
    assert runtime_devices(SimpleNamespace(device=torch.device("cuda:1")), options) == (1,)
    with pytest.raises(NativeUnavailable, match="does not select CUDA"):
        runtime_devices(SimpleNamespace(device=torch.device("cpu")), options)
    assert runtime_devices(SimpleNamespace(device=torch.device("cpu")),
                           NativeOptions(cache_dir=tmp_path, devices=(0,))) == (0,)


def test_touch_bfs_capability_is_checked_during_preparation(tmp_path):
    from cayleypy_native.backend import prepare_runtime
    with pytest.raises(NativeUnavailable, match="32 generators"):
        prepare_runtime(SimpleNamespace(move_count=33), object(),
                        NativeOptions(cache_dir=tmp_path, touch_bfs_radius=1), tmp_path, (0,))


def test_bf16_requires_sm80_before_build_or_launch(monkeypatch, tmp_path):
    import torch
    from cayleypy_native.backend import prepare_runtime
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7, 5) if d == 1 else (8, 6))
    with pytest.raises(NativeUnavailable, match="BF16.*SM80"):
        prepare_runtime(SimpleNamespace(move_count=24), SimpleNamespace(backend="mlp", manifest={"dtype": "bf16", "output_dim": 1}),
                        NativeOptions(cache_dir=tmp_path), tmp_path, (0, 1))


def test_fp16_requires_sm75_before_build_or_launch(monkeypatch, tmp_path):
    import torch
    from cayleypy_native.backend import prepare_runtime
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (6, 1))
    model = SimpleNamespace(backend="mlp", manifest={"dtype": "fp16", "output_dim": 1})
    with pytest.raises(NativeUnavailable, match="SM75"):
        prepare_runtime(SimpleNamespace(move_count=3), model, NativeOptions(cache_dir=tmp_path), tmp_path, (0,))


@pytest.mark.parametrize("outputs", [2, 3, 18])
def test_q_head_alignment_is_unavailable_before_build_or_launch(tmp_path, outputs):
    from cayleypy_native.backend import prepare_runtime
    model = SimpleNamespace(backend="mlp", manifest={"dtype": "fp16", "output_dim": outputs})
    with pytest.raises(NativeUnavailable, match="Q-head.*multiple of 8"):
        prepare_runtime(SimpleNamespace(move_count=outputs), model,
                        NativeOptions(cache_dir=tmp_path), tmp_path, (0,))


@pytest.mark.parametrize("outputs", [1, 8, 24])
def test_scalar_and_aligned_q_heads_reach_build_preparation(monkeypatch, tmp_path, outputs):
    import torch
    import cayleypy_native.build as build
    from cayleypy_native.backend import prepare_runtime
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (8, 6))
    monkeypatch.setattr(build, "ensure_runner", lambda *args: (tmp_path / "test-only-runner", {"test_only": True}))
    model = SimpleNamespace(backend="mlp", manifest={"dtype": "fp16", "output_dim": outputs})
    runtime = prepare_runtime(SimpleNamespace(move_count=24 if outputs == 1 else outputs), model,
                              NativeOptions(cache_dir=tmp_path), tmp_path, (0,))
    assert runtime.architectures == (86,)
    assert runtime.build_metadata == {"test_only": True}


@pytest.mark.skipif(os.name != "posix", reason="Linux process-group behavior requires POSIX")
def test_timeout_kills_descendant_process(tmp_path):
    import time
    marker = tmp_path / "child-survived"
    child = f"import time; from pathlib import Path; time.sleep(1.2); Path({str(marker)!r}).write_text('survived')"
    parent = f"import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',{child!r}]); time.sleep(30)"
    with pytest.raises(NativeBackendError, match="timed out"):
        run_process([sys.executable, "-c", parent], cwd=tmp_path, env=dict(os.environ), timeout=0.3,
                    log_path=tmp_path / "tree-timeout.log")
    time.sleep(1.3)
    assert not marker.exists()


def test_run_native_writes_unquoted_id_quoted_state_and_preserves_move_order(monkeypatch, tmp_path):
    import torch
    import cayleypy_native.backend as backend
    import cayleypy_native.build as build
    contract = SimpleNamespace(state_len=2, move_count=2, start=(1, 0), center=(0, 1), graph_hash="graph",
        replay=lambda path: tuple(path) == (1,),
        to_puzzle_info=lambda: {"central_state": [0, 1], "generators": {"m0": [0, 1], "m1": [1, 0]}})
    model = SimpleNamespace(weights_dir=tmp_path / "weights", backend="mlp", artifact_hash="model", manifest={"dtype": "fp16", "output_dim": 1})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7, 5))
    nccl_dir = tmp_path / "test-only-nccl"
    nccl_dir.mkdir()
    nccl_library = nccl_dir / "libnccl.so.2"
    nccl_library.write_bytes(b"test-only library bytes")
    nccl_sha = build.file_sha256(nccl_library)
    monkeypatch.setattr(build, "ensure_runner", lambda *args: (tmp_path / "test-only-runner",
                        {"test_only": True, "nccl": {"library": str(nccl_library), "library_sha256": nccl_sha}}))
    monkeypatch.setenv("LD_LIBRARY_PATH", "/unrelated/library/path")
    observed = {}

    def fake_process(command, **kwargs):
        assert (kwargs["cwd"] / "test_results").is_dir(), "native solution logging requires private test_results"
        observed.update(command=command, **kwargs)
        kwargs["log_path"].write_text("GLOBAL_BEAM_WIDTH_EFFECTIVE=1024\npuzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m1\n")
        return 0.1

    monkeypatch.setattr(backend, "run_process", fake_process)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    outcome = backend.run_native(contract, model, NativeOptions(cache_dir=tmp_path), 1000, 4, tmp_path / "run", (0,))
    assert outcome.path == (1,)
    assert (tmp_path / "run/test.csv").read_text() == 'initial_state_id,initial_state\n0,"1,0"\n'
    assert list(json.loads((tmp_path / "run/puzzle_info.json").read_text())["generators"]) == ["m0", "m1"]
    assert observed["command"][-5:] == ["0", "4", "1000", "1", "0"]
    assert observed["env"]["BEAM_RUNTIME_CONFIG_MODE"] == "auto"
    # Native auto uses 1024 logical slots and 2048 physical slots at this beam.
    # Its unbounded default of 8192 child rows rejects every candidate.
    assert int(observed["env"]["BEAM_B_MICRO"]) <= 2048
    assert observed["env"]["BEAM_HISTORY_MODE"] == "disk"
    assert observed["env"]["LD_LIBRARY_PATH"] == str(nccl_dir) + os.pathsep + "/unrelated/library/path"
    nccl_library.write_bytes(b"test-only replaced library")
    monkeypatch.setattr(backend, "run_process", lambda *args, **kwargs: pytest.fail("changed NCCL must prevent launch"))
    with pytest.raises(NativeBackendError, match="NCCL library changed"):
        backend.run_native(contract, model, NativeOptions(cache_dir=tmp_path), 1000, 4, tmp_path / "run-replaced", (0,))


def _rank_stream_fixture(root, rank, *, terminal=False):
    directory = root / "test-session" / "attempt_0" / str(rank)
    directory.mkdir(parents=True)
    stdout = f"WORLD_SIZE=2\nLOCAL_RANK={rank}\nCUDA_DEVICE_LOCAL_RANK={rank}\nGLOBAL_BEAM_WIDTH_EFFECTIVE=8192\n"
    if terminal:
        stdout += "puzzle_solved=1 puzzle_id=0 seconds=0.1 solution_length=1 solution=m1\n"
    (directory / "stdout.log").write_text(stdout)
    (directory / "stderr.log").write_text(f"rank {rank} diagnostic\n")


@pytest.mark.parametrize("mode", ["success", "failed_partial", "successful_missing_rank"])
def test_multirank_redirects_merge_even_on_failure(monkeypatch, tmp_path, mode):
    import torch
    import cayleypy_native.backend as backend
    import cayleypy_native.build as build
    contract = SimpleNamespace(state_len=2, move_count=2, start=(1, 0), center=(0, 1), graph_hash="graph",
        replay=lambda path: tuple(path) == (1,),
        to_puzzle_info=lambda: {"central_state": [0, 1], "generators": {"m0": [0, 1], "m1": [1, 0]}})
    model = SimpleNamespace(weights_dir=tmp_path / "weights", backend="mlp", artifact_hash="model",
                            manifest={"dtype": "fp16", "output_dim": 1})
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda d: (7, 5))
    monkeypatch.setattr(build, "ensure_runner", lambda *args: (tmp_path / "test-only-runner", {"test_only": True}))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    run = tmp_path / "run"

    def fake_process(command, **kwargs):
        assert (kwargs["cwd"] / "test_results").is_dir()
        assert f"--log-dir={run / 'worker-logs'}" in command
        assert "--redirects=3" in command and "--max-restarts=0" in command
        assert kwargs["log_path"] == run / "launcher.log"
        kwargs["log_path"].write_text("test-only launcher diagnostic\n")
        _rank_stream_fixture(run / "worker-logs", 0, terminal=True)
        if mode == "success":
            _rank_stream_fixture(run / "worker-logs", 1)
        if mode == "failed_partial":
            raise NativeBackendError("native process failed rc=7")
        return 0.2

    monkeypatch.setattr(backend, "run_process", fake_process)
    if mode == "success":
        result = backend.run_native(contract, model, NativeOptions(cache_dir=tmp_path), 7, 1, run, (0, 1))
        assert result.path == (1,) and result.metadata["worker_logs_complete"]
        assert len(result.metadata["worker_streams"]) == 4
    else:
        reason = "rc=7" if mode == "failed_partial" else "incomplete native worker logs"
        with pytest.raises(NativeBackendError, match=reason):
            backend.run_native(contract, model, NativeOptions(cache_dir=tmp_path), 7, 1, run, (0, 1))
    combined = (run / "native.log").read_text()
    assert "[launcher]" in combined and "test-only launcher diagnostic" in combined
    assert "[rank 0 stdout.log]" in combined and "[rank 0 stderr.log]" in combined
    assert "rank 0 diagnostic" in combined
    if mode != "success":
        assert "[log collection diagnostics]" in combined


def test_rank_log_collection_rejects_unexpected_rank_and_does_not_read_it(tmp_path):
    from cayleypy_native.backend import collect_worker_logs
    root = tmp_path / "worker-logs"
    for rank in range(3):
        _rank_stream_fixture(root, rank)
    (root / "test-session" / "attempt_0" / "2" / "stdout.log").write_text("UNEXPECTED_LOG_MUST_NOT_BE_READ")
    launcher, combined = tmp_path / "launcher.log", tmp_path / "native.log"
    launcher.write_text("launcher\n")
    with pytest.raises(NativeBackendError, match="rank directory set"):
        collect_worker_logs(root, launcher, combined, 2, strict=True)
    assert "UNEXPECTED_LOG_MUST_NOT_BE_READ" not in combined.read_text()


def test_rank_log_collection_does_not_follow_stream_symlink(tmp_path):
    from cayleypy_native.backend import collect_worker_logs
    root = tmp_path / "worker-logs"
    _rank_stream_fixture(root, 0)
    _rank_stream_fixture(root, 1)
    outside = tmp_path / "unrelated.log"
    outside.write_text("UNRELATED_LOG_MUST_NOT_BE_READ")
    redirected = root / "test-session" / "attempt_0" / "1" / "stdout.log"
    redirected.unlink()
    try:
        redirected.symlink_to(outside)
    except OSError:
        pytest.skip("creating a symlink is not permitted on this host")
    launcher, combined = tmp_path / "launcher.log", tmp_path / "native.log"
    launcher.write_text("launcher\n")
    with pytest.raises(NativeBackendError, match="unsafe rank stream"):
        collect_worker_logs(root, launcher, combined, 2, strict=True)
    assert "UNRELATED_LOG_MUST_NOT_BE_READ" not in combined.read_text()
