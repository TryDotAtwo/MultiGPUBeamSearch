"""Compare the externally scheduled Rust supervisor with the original loop."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest
from multigpubeamsearch import cluster
from multigpubeamsearch.errors import NativeUnavailable
from multigpubeamsearch.rust_control import extension


try:
    runtime = extension()
    if getattr(runtime, "CLUSTER_RANK_VERSION", None) != 1:
        raise NativeUnavailable("missing Rust cluster supervisor")
except NativeUnavailable:
    if os.environ.get("REQUIRE_RUST_CLUSTER") == "1":
        raise
    pytest.skip("Rust cluster supervisor is not installed", allow_module_level=True)


def torch_env(rank, world=2, local_world=2):
    env = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(world),
               LOCAL_RANK=str(rank % local_world), LOCAL_WORLD_SIZE=str(local_world),
               GROUP_RANK=str(rank // local_world), TORCHELASTIC_RUN_ID="cluster-test",
               CUDA_VISIBLE_DEVICES="0,1")
    env.pop("TORCHELASTIC_RESTART_COUNT", None)
    return env


@pytest.mark.parametrize("changes", [
    {}, {"LOCAL_WORLD_SIZE": "0"}, {"RANK": "2"}, {"LOCAL_RANK": "2"},
    {"WORLD_SIZE": "129"}, {"GROUP_RANK": "1"}, {"LOCAL_RANK": "1.0"},
    {"WORLD_SIZE": "00000000002"},
])
def test_scheduler_validation_matches_python(changes):
    from dataclasses import asdict

    env = torch_env(0)
    env.update(changes)
    try:
        expected = cluster._scheduler_rank_python(env)
    except ValueError as error:
        with pytest.raises(ValueError, match=__import__("re").escape(str(error))):
            runtime.resolve_scheduler_rank(env)
        with pytest.raises(ValueError, match=__import__("re").escape(str(error))):
            runtime.validate_cluster_rank(env)
    else:
        expected_fields = {
            "global_rank", "node_rank", "local_rank", "world_size", "local_world_size",
        }
        assert asdict(expected).keys() == expected_fields
        assert json.loads(runtime.resolve_scheduler_rank(env)) == asdict(expected)
        runtime.validate_cluster_rank(env)

@pytest.mark.parametrize("engine", ["python", "rust"])
def test_two_ranks_share_id_and_preserve_native_environment(tmp_path, engine):
    probe = """
import json, os, pathlib
names = ('RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','WORLD_SIZE','GROUP_RANK',
         'CUDA_VISIBLE_DEVICES','BEAM_NCCL_RUN_ID','BEAM_NCCL_ID_FILE',
         'BEAM_CLUSTER_LAUNCH','BEAM_PUZZLE_INFO_JSON','BEAM_HISTORY_MODE')
pathlib.Path('probe.json').write_text(json.dumps({key:os.environ[key] for key in names}))
"""
    def run(rank):
        return cluster.run_cluster_rank([sys.executable, "-c", probe], shared_dir=tmp_path,
                                        environ=torch_env(rank), timeout=20, engine=engine)
    with ThreadPoolExecutor(2) as pool:
        directories = list(pool.map(run, range(2)))
    rows = [json.loads((directory / "probe.json").read_text()) for directory in directories]
    assert [row["RANK"] for row in rows] == ["0", "1"]
    assert [row["LOCAL_RANK"] for row in rows] == ["0", "1"]
    assert {row["WORLD_SIZE"] for row in rows} == {"2"}
    assert {row["BEAM_NCCL_RUN_ID"] for row in rows} == {"cluster-test"}
    assert rows[0]["BEAM_NCCL_ID_FILE"] == rows[1]["BEAM_NCCL_ID_FILE"]
    assert all(row["CUDA_VISIBLE_DEVICES"] == "0,1" and row["BEAM_HISTORY_MODE"] == "disk" for row in rows)
    assert all(row["BEAM_PUZZLE_INFO_JSON"] == str(Path("data/puzzle_info.json").resolve()) for row in rows)
    for rank, directory in enumerate(directories):
        launch = json.loads((directory / "launch.json").read_text())
        assert (launch["rank"], launch["world_size"], launch["local_rank"]) == (rank, 2, rank)
        assert (directory / "native.log").is_file()
    with pytest.raises(FileExistsError):
        run(0)


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_peer_failure_stops_owned_child(tmp_path, engine):
    marker = tmp_path / "child-ready"
    script = "import pathlib,time; pathlib.Path(%r).write_text('ready'); time.sleep(60)" % str(marker)
    def run(rank, command):
        return cluster.run_cluster_rank([sys.executable, "-c", command], shared_dir=tmp_path,
                                        environ=torch_env(rank), timeout=20, engine=engine)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(run, 0, script)
        deadline = time.monotonic() + 10
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        assert marker.exists(), "the first rank did not start"
        second = pool.submit(run, 1, "raise SystemExit(7)")
        with pytest.raises(RuntimeError, match="7"):
            second.result(timeout=10)
        with pytest.raises(RuntimeError, match="peer"):
            first.result(timeout=10)
    assert "rank 1" in (tmp_path / "cluster-test/attempt-0/FAILED").read_text()


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_timeout_records_failure(tmp_path, engine):
    with pytest.raises(RuntimeError, match="timed out"):
        cluster.run_cluster_rank([sys.executable, "-c", "import time; time.sleep(60)"],
                                shared_dir=tmp_path, environ=torch_env(0, 1, 1),
                                timeout=.5, engine=engine)
    assert "timed out" in (tmp_path / "cluster-test/attempt-0/FAILED").read_text()


@pytest.mark.parametrize("engine", ["python", "rust"])
def test_slurm_step_rank_cli(tmp_path, engine):
    env = dict(os.environ, SLURM_PROCID="12", SLURM_LOCALID="4", SLURM_NODEID="1",
               SLURM_STEP_NUM_TASKS="14", SLURM_STEP_TASKS_PER_NODE="8,6",
               SLURM_JOB_ID="99", SLURM_STEP_ID="2", CUDA_VISIBLE_DEVICES="GPU-four")
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT"):
        env.pop(name, None)
    probe = "import json,os;print(json.dumps([os.environ[k] for k in ('RANK','GROUP_RANK','LOCAL_RANK','LOCAL_WORLD_SIZE','CUDA_VISIBLE_DEVICES')]))"
    worker = subprocess.run([sys.executable, "-m", "multigpubeamsearch.cluster_worker",
        "--shared-dir", str(tmp_path), "--engine", engine, "--", sys.executable, "-c", probe],
        env=env, capture_output=True, text=True, timeout=20)
    assert worker.returncode == 0, worker.stderr
    log = tmp_path / "slurm-99-2/attempt-0/rank-12/native.log"
    assert json.loads(log.read_text()) == ["12", "1", "4", "6", "GPU-four"]


@pytest.mark.parametrize("engine", ["python", "rust"])
@pytest.mark.parametrize("local_rank_flag", ["--local-rank", "--local_rank"])
def test_torchrun_local_rank_flag_aliases_preserve_scheduler_assignment(tmp_path, engine, local_rank_flag):
    env = torch_env(1)
    probe = "import json,os;print(json.dumps([os.environ['RANK'],os.environ['LOCAL_RANK'],os.environ['WORLD_SIZE']]))"
    worker = subprocess.run(
        [sys.executable, "-m", "multigpubeamsearch.cluster_worker",
         "--shared-dir", str(tmp_path), local_rank_flag, "1",
         "--engine", engine, "--", sys.executable, "-c", probe],
        env=env, capture_output=True, text=True, timeout=20,
    )
    assert worker.returncode == 0, worker.stderr
    log = tmp_path / "cluster-test/attempt-0/rank-1/native.log"
    assert json.loads(log.read_text()) == ["1", "1", "2"]


@pytest.mark.skipif(os.name != "posix", reason="POSIX SIGTERM semantics")
@pytest.mark.parametrize("engine", ["python", "rust"])
def test_worker_sigterm_publishes_failure_and_reaps_child(tmp_path, engine):
    ready = tmp_path / "ready"
    terminated = tmp_path / "terminated"
    probe = """
import pathlib, signal, sys, time
def stop(*_):
    pathlib.Path(sys.argv[2]).write_text('terminated')
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
pathlib.Path(sys.argv[1]).write_text('ready')
while True: time.sleep(.02)
"""
    worker = subprocess.Popen([sys.executable, "-m", "multigpubeamsearch.cluster_worker",
        "--shared-dir", str(tmp_path), "--engine", engine, "--", sys.executable,
        "-c", probe, str(ready), str(terminated)], env=torch_env(0, 1, 1),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and time.monotonic() < deadline and worker.poll() is None:
            time.sleep(.02)
        assert ready.exists(), worker.communicate(timeout=2) if worker.poll() is not None else "worker did not start"
        worker.send_signal(signal.SIGTERM)
        worker.communicate(timeout=10)
        assert worker.returncode == 128 + signal.SIGTERM
        assert terminated.read_text() == "terminated"
        assert (tmp_path / "cluster-test/attempt-0/FAILED").is_file()
    finally:
        if worker.poll() is None:
            worker.kill()
            worker.wait()

def test_installed_rust_scheduler_rank_resolves_torchrun_and_slurm():
    resolver = getattr(runtime, "resolve_scheduler_rank", None)
    assert callable(resolver), "the installed Rust runtime does not return its scheduler rank"

    torchrun = torch_env(1)
    assert json.loads(resolver(torchrun)) == {
        "global_rank": 1,
        "node_rank": 0,
        "local_rank": 1,
        "world_size": 2,
        "local_world_size": 2,
    }

    slurm = {
        "SLURM_PROCID": "12",
        "SLURM_LOCALID": "4",
        "SLURM_NODEID": "1",
        "SLURM_STEP_NUM_TASKS": "14",
        "SLURM_STEP_TASKS_PER_NODE": "8,6",
    }
    assert json.loads(resolver(slurm)) == {
        "global_rank": 12,
        "node_rank": 1,
        "local_rank": 4,
        "world_size": 14,
        "local_world_size": 6,
    }
    with pytest.raises(ValueError, match="rank or world size"):
        resolver(dict(torchrun, RANK="2"))


def test_public_scheduler_rank_uses_installed_rust_then_legacy_fallback(monkeypatch):
    from multigpubeamsearch import rust_control

    native_resolver = rust_control.resolve_scheduler_rank
    observed = []

    def observe(environment):
        observed.append(dict(environment))
        return native_resolver(environment)

    monkeypatch.setattr(rust_control, "resolve_scheduler_rank", observe)
    python_fallback = cluster._scheduler_rank_python

    def forbidden_fallback(*_args, **_kwargs):
        pytest.fail("the installed Rust scheduler path fell back to Python")

    monkeypatch.setattr(cluster, "_scheduler_rank_python", forbidden_fallback)
    environment = torch_env(1)
    assert cluster.scheduler_rank(environment) == cluster.ScheduledRank(1, 0, 1, 2, 2)
    assert observed == [environment]

    monkeypatch.setattr(rust_control, "resolve_scheduler_rank", lambda _environment: None)
    monkeypatch.setattr(cluster, "_scheduler_rank_python", python_fallback)
    assert cluster.scheduler_rank(environment) == python_fallback(environment)


@pytest.mark.parametrize(
    "local_rank,visible_devices,expected",
    [(7, 1, 0), (0, 4, 0), (3, 4, 3), (-1, 4, -1)],
)
def test_installed_rust_cuda_ordinal_matches_scheduler_mapping(
    local_rank, visible_devices, expected
):
    resolver = getattr(runtime, "resolve_cuda_ordinal", None)
    assert callable(resolver), "the installed Rust runtime lacks CUDA ordinal resolution"
    assert getattr(runtime, "CLUSTER_DEVICE_BINDING_VERSION", None) == 1
    assert resolver(local_rank, visible_devices) == expected


def test_installed_rust_cuda_ordinal_rejects_invalid_visibility_and_rank():
    resolver = getattr(runtime, "resolve_cuda_ordinal", None)
    assert callable(resolver)
    with pytest.raises(ValueError, match="no visible CUDA devices"):
        resolver(0, 0)
    with pytest.raises(ValueError, match="LOCAL_RANK exceeds the visible CUDA devices"):
        resolver(4, 4)


def test_public_cuda_ordinal_uses_rust_and_preserves_legacy_fallback(monkeypatch):
    from multigpubeamsearch import rust_control

    observed = []
    native_resolver = rust_control.resolve_cuda_ordinal

    def observe(local_rank, visible_devices):
        observed.append((local_rank, visible_devices))
        return native_resolver(local_rank, visible_devices)

    def forbidden_fallback(*_args, **_kwargs):
        pytest.fail("the installed Rust CUDA ordinal path fell back to Python")

    monkeypatch.setattr(rust_control, "resolve_cuda_ordinal", observe)
    monkeypatch.setattr(cluster, "_cuda_ordinal_python", forbidden_fallback)
    assert cluster.ScheduledRank(3, 0, 3, 4, 4).cuda_ordinal(4) == 3
    assert cluster.ScheduledRank(0, 0, 9, 9, 1).cuda_ordinal(1) == 0
    assert observed == [(3, 4), (9, 1)]

    fallback = cluster._cuda_ordinal_python
    monkeypatch.setattr(rust_control, "resolve_cuda_ordinal", lambda *_args: None)
    monkeypatch.setattr(cluster, "_cuda_ordinal_python", fallback)
    assert cluster.ScheduledRank(1, 0, 1, 2, 2).cuda_ordinal(2) == 1

    monkeypatch.setattr(
        rust_control, "resolve_cuda_ordinal",
        lambda *_args: (_ for _ in ()).throw(NativeUnavailable("incompatible resolver")),
    )
    with pytest.raises(ValueError, match="incompatible resolver"):
        cluster.ScheduledRank(1, 0, 1, 2, 2).cuda_ordinal(2)
