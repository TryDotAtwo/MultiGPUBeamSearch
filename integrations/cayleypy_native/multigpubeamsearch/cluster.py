"""Global beam geometry and externally scheduled native ranks."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time


MAX_WORLD_SIZE = 128  # Fixed host control arrays in the native final exchange.


@dataclass(frozen=True)
class RankPlacement:
    global_rank: int
    node_rank: int
    local_rank: int


@dataclass(frozen=True)
class ClusterPlan:
    ranks: tuple[RankPlacement, ...]
    requested_beam: int
    effective_beam: int
    alignment: int
    local_frontier_capacity: int
    shard_count: int
    hardware_verified: bool = False


def plan_cluster(gpus_per_node, beam_width, *, shard_count=16, batch_alignment=1024):
    """Plan a block layout and the native exact top-k before memory admission."""
    counts = tuple(gpus_per_node)
    if not counts or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("each node must have a positive GPU count")
    world = sum(counts)
    if world > MAX_WORLD_SIZE:
        raise ValueError(f"native final exchange supports at most {MAX_WORLD_SIZE} ranks")
    for name, value, bits in (("beam_width", beam_width, 64), ("shard_count", shard_count, 32),
                              ("batch_alignment", batch_alignment, 32)):
        if type(value) is not int or not 0 < value < 2**bits:
            raise ValueError(f"{name} must be a positive uint{bits}")
    alignment = world * shard_count * batch_alignment
    capacity = (beam_width + world - 1) // world
    if alignment >= 2**64 or capacity >= 2**32:
        raise ValueError("native alignment or rank-local frontier index overflows")
    ranks = []
    for node, count in enumerate(counts):
        for local in range(count):
            ranks.append(RankPlacement(len(ranks), node, local))
    return ClusterPlan(tuple(ranks), beam_width, beam_width, alignment, capacity, shard_count)


@dataclass(frozen=True)
class ScheduledRank(RankPlacement):
    world_size: int
    local_world_size: int

    def cuda_ordinal(self, visible_devices: int) -> int:
        """Resolve the scheduler rank inside this process's CUDA namespace."""
        if type(visible_devices) is not int or visible_devices <= 0:
            raise ValueError("no visible CUDA devices")
        if visible_devices == 1:
            return 0
        if self.local_rank >= visible_devices:
            raise ValueError("LOCAL_RANK exceeds the visible CUDA devices")
        return self.local_rank


def _number(env, name):
    value = env.get(name, "")
    if not re.fullmatch(r"[0-9]{1,10}", value):
        raise ValueError(f"missing or invalid scheduler variable {name}")
    return int(value)


def _slurm_counts(text):
    # Slurm encodes repeated task counts as e.g. 8(x2),4.
    counts = []
    for part in text.split(","):
        match = re.fullmatch(r"([1-9][0-9]{0,2})(?:\(x([1-9][0-9]{0,2})\))?", part)
        if not match:
            raise ValueError("invalid Slurm tasks-per-node list")
        count, repeat = int(match[1]), int(match[2] or 1)
        if len(counts) + repeat > MAX_WORLD_SIZE:
            raise ValueError("invalid Slurm node count")
        counts.extend([count] * repeat)
    return counts


def scheduler_rank(environ=None):
    """Read the actual torchrun or Slurm step assignment, including task masks."""
    env = os.environ if environ is None else environ
    # A torchrun agent can itself occupy a single Slurm task. Conversely, a
    # direct srun child inherits normalized RANK variables from this launcher.
    if "TORCHELASTIC_RUN_ID" in env or "SLURM_PROCID" not in env:
        world, rank, local_size, local, node = (
            _number(env, key) for key in ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK", "GROUP_RANK"))
        if not local_size or world % local_size or node >= world // local_size:
            raise ValueError("invalid torchrun node or local group size")
    else:
        world = _number(env, "SLURM_STEP_NUM_TASKS" if "SLURM_STEP_NUM_TASKS" in env else "SLURM_NTASKS")
        rank, local, node = (_number(env, key) for key in ("SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID"))
        counts = _slurm_counts(env.get("SLURM_STEP_TASKS_PER_NODE", env.get("SLURM_TASKS_PER_NODE", "")))
        if sum(counts) != world or node >= len(counts):
            raise ValueError("Slurm node counts disagree with the current step")
        local_size = counts[node]
    if not 1 <= world <= MAX_WORLD_SIZE or rank >= world or local >= local_size:
        raise ValueError(f"invalid rank or world size (maximum {MAX_WORLD_SIZE})")
    return ScheduledRank(rank, node, local, world, local_size)


def run_cluster_rank(command, *, shared_dir, run_id=None, timeout=3600, environ=None):
    """Run this scheduler-assigned worker. Return its log and artifact directory.

    Every worker must see shared_dir at the same path. CUDA_VISIBLE_DEVICES and
    NCCL transport settings belong to the scheduler. Run IDs are single-use.
    The native process owns all device work and the communicator.
    """
    env = dict(os.environ if environ is None else environ)
    rank = scheduler_rank(env)
    if not command or isinstance(command, (str, bytes)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("a native argument vector and a positive finite timeout are required")
    if run_id is None:
        run_id = env.get("TORCHELASTIC_RUN_ID")
        if not run_id and env.get("SLURM_JOB_ID") and env.get("SLURM_STEP_ID"):
            run_id = f"slurm-{env['SLURM_JOB_ID']}-{env['SLURM_STEP_ID']}"
    if not isinstance(run_id, str) or run_id == "none" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id):
        raise ValueError("supply a unique --run-id shared by all nodes")
    attempt = _number(env, "TORCHELASTIC_RESTART_COUNT") if "TORCHELASTIC_RESTART_COUNT" in env else 0
    root = Path(shared_dir).resolve() / run_id / f"attempt-{attempt}"
    root.mkdir(parents=True, exist_ok=True)
    directory = root / f"rank-{rank.global_rank}"
    directory.mkdir()  # Refuse collisions or stale rank output.
    failed = root / "FAILED"
    process = None
    try:
        (root / f"node-{rank.node_rank}-local-{rank.local_rank}").mkdir()
        if failed.exists():
            raise RuntimeError(f"a peer failed; see {failed}")
        env.update(RANK=str(rank.global_rank), WORLD_SIZE=str(rank.world_size),
                   GROUP_RANK=str(rank.node_rank),
                   LOCAL_RANK=str(rank.local_rank), LOCAL_WORLD_SIZE=str(rank.local_world_size),
                   BEAM_NCCL_ID_FILE=str(root / "nccl-id.bin"), BEAM_NCCL_RUN_ID=run_id,
                   BEAM_CLUSTER_LAUNCH="1")
        # The child works in a fresh rank-owned directory. Resolve native
        # input paths relative to the caller before changing its cwd.
        for name, default in (("BEAM_PUZZLE_INFO_JSON", "data/puzzle_info.json"),
                              ("BEAM_TEST_CSV", "data/test.csv"), ("BEAM_WEIGHT_DIR", "stream1_weights"),
                              ("BEAM_GENERATOR_PATH", None), ("BEAM_BLEND_DIR", None),
                              ("BEAM_START_STATE_FILE", None), ("BEAM_TARGET_STATE_FILE", None)):
            value = env.get(name, default)
            if value:
                env[name] = str(Path(value).resolve())
        env.setdefault("BEAM_HISTORY_MODE", "disk")
        env.setdefault("BEAM_HISTORY_DIR", str(directory / "history"))
        argv = list(map(os.fspath, command))
        executable = shutil.which(argv[0], path=env.get("PATH"))
        if not executable:
            raise FileNotFoundError(f"cannot find native executable: {argv[0]}")
        argv[0] = str(Path(executable).resolve())
        (directory / "launch.json").write_text(json.dumps({
            "argv": argv, "rank": rank.global_rank, "world_size": rank.world_size, "node": rank.node_rank,
            "local_rank": rank.local_rank, "local_world_size": rank.local_world_size,
            "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES"),
        }, indent=2) + "\n", encoding="utf-8")
        deadline = time.monotonic() + timeout
        log_path = directory / "native.log"
        with log_path.open("wb") as log:
            process = subprocess.Popen(argv, env=env, cwd=directory, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, shell=False)
            while True:
                code = process.poll()
                if code is not None:
                    if code:
                        raise RuntimeError(f"native rank {rank.global_rank} exited {code}; see {log_path}")
                    return directory
                if failed.exists():
                    raise RuntimeError(f"a peer failed; see {failed}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"native rank {rank.global_rank} timed out; see {log_path}")
                time.sleep(min(.2, remaining))
    except BaseException as error:
        try:
            with failed.open("x", encoding="utf-8") as marker:
                marker.write(f"rank {rank.global_rank}: {error}\n")
        except FileExistsError:
            pass
        raise
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
