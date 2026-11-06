from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest
from multigpubeamsearch import cluster


def torch_rank(rank, world=16, local_world=8):
    return dict(os.environ, RANK=str(rank), LOCAL_RANK=str(rank % local_world),
                WORLD_SIZE=str(world), LOCAL_WORLD_SIZE=str(local_world),
                GROUP_RANK=str(rank // local_world), TORCHELASTIC_RUN_ID="test-job",
                CUDA_VISIBLE_DEVICES="GPU-a,GPU-b,GPU-c,GPU-d,GPU-e,GPU-f,GPU-g,GPU-h")


def test_slurm_step_uses_its_cohort_not_the_allocation():
    env = dict(SLURM_PROCID="12", SLURM_LOCALID="4", SLURM_NODEID="1",
               SLURM_STEP_NUM_TASKS="14", SLURM_NTASKS="32",
               SLURM_STEP_TASKS_PER_NODE="8,6", SLURM_TASKS_PER_NODE="8(x4)",
               CUDA_VISIBLE_DEVICES="GPU-uuid")
    rank = cluster.scheduler_rank(env)
    assert (rank.global_rank, rank.node_rank, rank.local_rank) == (12, 1, 4)
    assert (rank.world_size, rank.local_world_size) == (14, 6)
    env.update(SLURM_STEP_NUM_TASKS="24", SLURM_STEP_TASKS_PER_NODE="8(x3)")
    assert cluster.scheduler_rank(env).local_world_size == 8


@pytest.mark.parametrize("changes", [
    dict(LOCAL_WORLD_SIZE="0"), dict(LOCAL_RANK="8"), dict(RANK="16"),
    dict(WORLD_SIZE="129"), dict(GROUP_RANK="2"), dict(LOCAL_RANK="1.0"),
])
def test_invalid_scheduler_ranks_fail_before_launch(changes):
    env = torch_rank(15)
    env.update(changes)
    with pytest.raises(ValueError):
        cluster.scheduler_rank(env)


def test_cuda_device_after_scheduler_mask():
    rank = cluster.scheduler_rank(torch_rank(15))
    assert rank.cuda_ordinal(8) == 7
    assert rank.cuda_ordinal(1) == 0
    for count in (0, 4, True):
        with pytest.raises(ValueError):
            rank.cuda_ordinal(count)


def test_slurm_cli_preserves_bound_gpu_and_step_geometry(tmp_path):
    env = dict(os.environ, SLURM_PROCID="12", SLURM_LOCALID="4", SLURM_NODEID="1",
               SLURM_STEP_NUM_TASKS="14", SLURM_STEP_TASKS_PER_NODE="8,6",
               SLURM_JOB_ID="99", SLURM_STEP_ID="2", CUDA_VISIBLE_DEVICES="GPU-four",
               PYTHONPATH=str(Path(cluster.__file__).resolve().parents[1]))
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "TORCHELASTIC_RUN_ID", "TORCHELASTIC_RESTART_COUNT"):
        env.pop(key, None)
    probe = """
import json, os, sys
from multigpubeamsearch import scheduler_rank
rank = scheduler_rank()
print(json.dumps([rank.global_rank, rank.node_rank, rank.local_rank, rank.local_world_size,
                  rank.cuda_ordinal(1), os.environ['GROUP_RANK'], os.environ['CUDA_VISIBLE_DEVICES'],
                  'torch' in sys.modules]))
"""
    worker = subprocess.run([sys.executable, "-m", "multigpubeamsearch.cluster_worker",
                             "--shared-dir", str(tmp_path), "--", sys.executable, "-c", probe],
                            env=env, capture_output=True, text=True, timeout=20)
    log = tmp_path / "slurm-99-2" / "attempt-0" / "rank-12" / "native.log"
    assert worker.returncode == 0, (worker.stderr, log.read_text() if log.exists() else "")
    assert json.loads(log.read_text()) == [12, 1, 4, 6, 0, "1", "GPU-four", False]


def test_two_nodes_pass_one_world_and_isolate_artifacts(tmp_path):
    # Each child records the native ABI it actually receives. A file stands in
    # for NCCL's opaque unique ID; no CUDA collective is simulated here.
    probe = """
import json, os, pathlib, time
env = os.environ
rank = int(env["RANK"])
id_file = pathlib.Path(env["BEAM_NCCL_ID_FILE"])
if rank == 0:
    id_file.write_text("unique-id")
while not id_file.exists():
    time.sleep(.01)
with open("probe.json", "w") as out:
    json.dump([env[k] for k in ("RANK","LOCAL_RANK","LOCAL_WORLD_SIZE","WORLD_SIZE",
              "CUDA_VISIBLE_DEVICES","BEAM_NCCL_RUN_ID")] + [str(id_file)], out)
"""
    def worker(rank):
        return cluster.run_cluster_rank([sys.executable, "-c", probe], shared_dir=tmp_path,
                                        environ=torch_rank(rank), timeout=45)

    with ThreadPoolExecutor(16) as pool:
        outputs = list(pool.map(worker, range(16)))
    rows = [json.loads((directory / "probe.json").read_text()) for directory in outputs]
    assert len(set(outputs)) == 16
    assert len({row[6] for row in rows}) == 1
    assert Path(rows[0][6]).is_relative_to(tmp_path)
    assert rows[8][:6] == ["8", "0", "8", "16", torch_rank(8)["CUDA_VISIBLE_DEVICES"], "test-job"]
    assert rows[15][:4] == ["15", "7", "8", "16"]
    assert all((directory / "native.log").exists() for directory in outputs)
    with pytest.raises((FileExistsError, RuntimeError)):
        worker(15)


def test_failed_rank_cancels_running_peer(tmp_path):
    def worker(rank):
        env = torch_rank(rank, world=2, local_world=1)
        script = "import time; time.sleep(60)" if rank == 0 else "raise SystemExit(7)"
        return cluster.run_cluster_rank([sys.executable, "-c", script], shared_dir=tmp_path,
                                        environ=env, timeout=40)
    start = time.monotonic()
    with ThreadPoolExecutor(2) as pool:
        futures = [pool.submit(worker, rank) for rank in range(2)]
        with pytest.raises(RuntimeError, match="peer|rank 1"):
            futures[0].result(timeout=20)
        with pytest.raises(RuntimeError, match="7"):
            futures[1].result(timeout=20)
    assert time.monotonic() - start < 20


def test_stalled_native_rank_times_out(tmp_path):
    with pytest.raises(RuntimeError, match="timed out"):
        cluster.run_cluster_rank([sys.executable, "-c", "import time; time.sleep(60)"],
                                 shared_dir=tmp_path, environ=torch_rank(0, 1, 1), timeout=.5)
