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
    python_fallback = cluster._cuda_ordinal_python

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

    monkeypatch.setattr(rust_control, "resolve_cuda_ordinal", lambda *_args: None)
    monkeypatch.setattr(cluster, "_cuda_ordinal_python", python_fallback)
    assert cluster.ScheduledRank(1, 0, 1, 2, 2).cuda_ordinal(2) == 1

    monkeypatch.setattr(
        rust_control, "resolve_cuda_ordinal",
        lambda *_args: (_ for _ in ()).throw(NativeUnavailable("incompatible resolver")),
    )
    with pytest.raises(ValueError, match="incompatible resolver"):
        cluster.ScheduledRank(1, 0, 1, 2, 2).cuda_ordinal(2)


def test_installed_rust_cuda_visibility_drives_public_mapping_and_fallback(monkeypatch):
    from multigpubeamsearch import backend, rust_control

    native_resolver = rust_control.resolve_cuda_visibility
    python_fallback = backend._cuda_visibility_python
    observed = []

    def observe(devices, inherited):
        observed.append((tuple(devices), inherited))
        return native_resolver(devices, inherited)

    def forbidden_fallback(*_args, **_kwargs):
        pytest.fail("installed Rust CUDA visibility path fell back to Python")

    monkeypatch.setattr(rust_control, "resolve_cuda_visibility", observe)
    monkeypatch.setattr(backend, "_cuda_visibility_python", forbidden_fallback)
    source = {
        "PATH": "safe",
        "CUDA_VISIBLE_DEVICES": " GPU-a, GPU-b ,GPU-c ",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "RANK": "3",
        "NCCL_DEBUG": "TRACE",
    }
    mapped = backend.cuda_visibility((2, 0), source)
    assert mapped == "GPU-c,GPU-a"
    assert observed == [((2, 0), source["CUDA_VISIBLE_DEVICES"])]

    monkeypatch.setattr(rust_control, "resolve_cuda_visibility", lambda *_args: None)
    monkeypatch.setattr(backend, "_cuda_visibility_python", python_fallback)
    legacy = backend.cuda_visibility((1, 1), source)
    assert legacy == "GPU-b,GPU-b"

    monkeypatch.setattr(rust_control, "resolve_cuda_visibility", native_resolver)
    with pytest.raises(NativeUnavailable, match="cannot map selected devices"):
        backend.cuda_visibility((1,), {
            "CUDA_VISIBLE_DEVICES": "GPU-a,-1",
        })


def test_installed_rust_child_environment_controls_public_child_setup_and_fallback(monkeypatch):
    from multigpubeamsearch import backend, rust_control

    native_resolver = rust_control.resolve_child_environment
    python_fallback = backend._child_environment_python
    observed = []

    def observe(environment, devices, nccl_id_file):
        observed.append((dict(environment), tuple(devices), nccl_id_file))
        return native_resolver(environment, devices, nccl_id_file)

    def forbidden_fallback(*_args, **_kwargs):
        pytest.fail("installed Rust child-environment path fell back to Python")

    source = {
        "PATH": "safe",
        "CUDA_VISIBLE_DEVICES": " GPU-a, GPU-b ,GPU-c ",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "RANK": "3",
        "ROLE_RANK": "2",
        "MASTER_ADDR": "coordinator",
        "NCCL_DEBUG": "TRACE",
        "BEAM_WEIGHT_DIR": "wrong",
        "STREAM4_BATCH_ALIGNMENT": "999",
        "PUBLISH_RESULTS": "1",
        "RESULTS_INGEST_TOKEN": "filtered",
        "CUSTOM_VALUE": "kept",
    }
    monkeypatch.setattr(rust_control, "resolve_child_environment", observe)
    monkeypatch.setattr(backend, "_child_environment_python", forbidden_fallback)
    env = backend.child_environment((2, 0), Path("/rank"), source)
    assert env == {
        "PATH": "safe",
        "CUSTOM_VALUE": "kept",
        "CUDA_VISIBLE_DEVICES": "GPU-c,GPU-a",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "BEAM_NCCL_ID_FILE": "/rank/nccl-id.bin",
    }
    assert observed == [(source, (2, 0), "/rank/nccl-id.bin")]

    monkeypatch.setattr(rust_control, "resolve_child_environment", lambda *_args: None)
    monkeypatch.setattr(rust_control, "resolve_cuda_visibility", lambda *_args: None)
    monkeypatch.setattr(backend, "_child_environment_python", python_fallback)
    legacy = backend.child_environment((1, 1), Path("/legacy"), source)
    assert legacy["CUDA_VISIBLE_DEVICES"] == "GPU-b,GPU-b"
    assert legacy["BEAM_NCCL_ID_FILE"] == "/legacy/nccl-id.bin"
    assert "RANK" not in legacy and "NCCL_DEBUG" not in legacy

    monkeypatch.setattr(rust_control, "resolve_child_environment", native_resolver)
    with pytest.raises(NativeUnavailable, match="cannot map selected devices"):
        backend.child_environment((1,), Path("/invalid"), {
            "CUDA_VISIBLE_DEVICES": "GPU-a,-1",
        })


def test_installed_rust_runtime_device_plan_controls_public_selection_and_fallback(monkeypatch):
    from types import SimpleNamespace
    import torch
    from multigpubeamsearch import backend, rust_control

    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)

    native_resolver = rust_control.resolve_runtime_device_plan
    observed = []

    def observe(cuda_available, visible_count, explicit, num_gpus, graph_devices):
        observed.append((cuda_available, visible_count, explicit, num_gpus, graph_devices))
        return native_resolver(cuda_available, visible_count, explicit, num_gpus, graph_devices)

    def forbidden(*_args, **_kwargs):
        pytest.fail("public runtime device selection bypassed the Rust plan")

    monkeypatch.setattr(rust_control, "resolve_runtime_device_plan", observe)
    monkeypatch.setattr(rust_control, "resolve_runtime_devices", forbidden)
    monkeypatch.setattr(backend, "_resolve_runtime_devices_python", forbidden)
    options = SimpleNamespace(devices=(2, 0), num_gpus=None)
    assert backend.runtime_devices(object(), options) == (2, 0)
    assert observed == [(True, 3, (2, 0), None, ())]

    monkeypatch.setattr(rust_control, "resolve_runtime_device_plan", lambda *_args: None)
    monkeypatch.setattr(rust_control, "resolve_runtime_devices", lambda *_args: (1, 0))
    assert backend.runtime_devices(object(), options) == (1, 0)


def test_installed_rust_runtime_device_plan_rejects_missing_cuda_before_count(monkeypatch):
    from types import SimpleNamespace
    import torch
    from multigpubeamsearch import backend

    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(backend.platform, "system", lambda: "Linux")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden_count():
        pytest.fail("device count should not be queried when CUDA is unavailable")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden_count)
    with pytest.raises(NativeUnavailable, match="native beam requires available CUDA devices"):
        backend.runtime_devices(object(), SimpleNamespace(devices=(0,), num_gpus=None))


def test_installed_rust_search_environment_controls_public_permutation_launch(monkeypatch):
    from multigpubeamsearch import backend, rust_control

    native_resolver = rust_control.resolve_search_environment
    python_fallback = backend._runtime_search_environment_python
    observed = []

    def observe(*args):
        observed.append(args)
        return native_resolver(*args)

    def forbidden(*_args, **_kwargs):
        pytest.fail("installed Rust search environment plan fell back to Python")

    monkeypatch.setattr(rust_control, "resolve_search_environment", observe)
    monkeypatch.setattr(backend, "_runtime_search_environment_python", forbidden)
    values = (
        "/run/puzzle_info.json", "/run/test.csv", "/models/weights",
        "/run", 3, 17,
    )
    expected = {
        "BEAM_GENERATOR_PATH": "/run/puzzle_info.json",
        "BEAM_PUZZLE_INFO_JSON": "/run/puzzle_info.json",
        "BEAM_TEST_CSV": "/run/test.csv",
        "BEAM_WEIGHT_DIR": "/models/weights",
        "BEAM_RUNTIME_CONFIG_MODE": "auto",
        "BEAM_SOLVED_NEIGHBORHOOD_RADIUS": "3",
        "BEAM_SOLVED_NEIGHBORHOOD_MAX_ENTRIES": "17",
        "BEAM_STREAM2_SUFFIX_RADIUS": "0",
        "BEAM_SOLVE_BUCKET_MODE": "0",
        "BEAM_HISTORY_MODE": "disk",
        "BEAM_HISTORY_DIR": "/run/history",
        "BEAM_HISTORY_DISK_PATH": "/run/history",
    }
    assert backend.runtime_search_environment(*values) == expected
    assert observed == [values]

    monkeypatch.setattr(rust_control, "resolve_search_environment", lambda *_args: None)
    monkeypatch.setattr(backend, "_runtime_search_environment_python", python_fallback)
    assert backend.runtime_search_environment(*values) == expected


def test_installed_rust_model_environment_controls_public_permutation_launch_and_fallback(monkeypatch):
    from types import SimpleNamespace
    from pathlib import Path
    from multigpubeamsearch import backend, rust_control

    native_resolver = rust_control.resolve_search_model_environment
    python_fallback = backend._search_model_environment_python
    observed = []

    ensemble = SimpleNamespace(
        backend="ensemble",
        weights_dir=Path("/models/blend"),
        manifest={"ensemble": {"models": [{"weights_dir": "head-a"}]}},
        artifact_hash="a" * 64,
    )
    runtime = SimpleNamespace(build_metadata={"inference_backend": "cutlass"})

    def observe(request):
        observed.append(dict(request))
        return native_resolver(request)

    def forbidden(*_args, **_kwargs):
        pytest.fail("installed Rust model environment path fell back to Python")

    monkeypatch.setattr(rust_control, "resolve_search_model_environment", observe)
    monkeypatch.setattr(backend, "_search_model_environment_python", forbidden)
    assert backend.search_model_environment(
        ensemble, runtime, graph_hash="b" * 64, mlp_host_binding=False
    ) == {
        "BEAM_BLEND_DIR": "/models/blend",
        "BEAM_WEIGHT_DIR": "/models/blend/head-a",
        "BEAM_ENSEMBLE_BACKEND": "cutlass",
        "BEAM_STREAM1_EXECUTOR": "libtorch_eager",
    }
    assert observed == [{
        "model_backend": "ensemble",
        "ensemble_blend_dir": "/models/blend",
        "ensemble_first_weights_dir": "/models/blend/head-a",
        "inference_backend": "cutlass",
        "mlp_host_binding": None,
    }]

    mlp = SimpleNamespace(
        backend="mlp",
        weights_dir=Path("/models/mlp"),
        manifest={},
        artifact_hash="c" * 64,
    )
    mlp_env = backend.search_model_environment(
        mlp, runtime, graph_hash="d" * 64, mlp_host_binding=True
    )
    assert mlp_env == {
        "BEAM_MLP_GRAPH_HASH": "d" * 64,
        "BEAM_MLP_ARTIFACT_HASH": "c" * 64,
    }

    monkeypatch.setattr(rust_control, "resolve_search_model_environment", lambda _request: None)
    monkeypatch.setattr(backend, "_search_model_environment_python", python_fallback)
    assert backend.search_model_environment(
        ensemble, runtime, graph_hash="b" * 64, mlp_host_binding=False
    ) == {
        "BEAM_BLEND_DIR": "/models/blend",
        "BEAM_WEIGHT_DIR": "/models/blend/head-a",
        "BEAM_ENSEMBLE_BACKEND": "cutlass",
        "BEAM_STREAM1_EXECUTOR": "libtorch_eager",
    }

    from multigpubeamsearch.errors import NativeUnavailable

    def rejected(_request):
        raise NativeUnavailable("incompatible Rust model environment")

    monkeypatch.setattr(rust_control, "resolve_search_model_environment", rejected)
    with pytest.raises(NativeUnavailable, match="incompatible Rust model environment"):
        backend.search_model_environment(
            ensemble, runtime, graph_hash="b" * 64, mlp_host_binding=False
        )
