"""Bounded single-host native ranks; no Python distributed worker startup."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from .errors import NativeBackendError


def run_local_ranks(command, *, world_size, cwd, env, timeout, log_path, worker_logs):
    from .backend import _stop_process_tree
    if type(world_size) is not int or world_size < 2 or timeout <= 0:
        raise ValueError("invalid local rank launch geometry or timeout")
    started = time.monotonic()
    processes = []
    rendezvous = "native-local-" + uuid.uuid4().hex
    root = Path(worker_logs) / "native" / "attempt_0"
    root.mkdir(parents=True, exist_ok=False)
    Path(log_path).write_text("single-host direct native ranks\n", encoding="utf-8")
    try:
        with ExitStack() as stack:
            for rank in range(world_size):
                directory = root / str(rank)
                directory.mkdir()
                rank_env = dict(env, WORLD_SIZE=str(world_size), RANK=str(rank),
                                LOCAL_RANK=str(rank), OMP_NUM_THREADS="1",
                                BEAM_NCCL_RUN_ID=rendezvous)
                args = [*command, str(world_size), str(rank)]
                (directory / "command.json").write_text(json.dumps(args) + "\n", encoding="utf-8")
                stdout = stack.enter_context((directory / "stdout.log").open("wb"))
                stderr = stack.enter_context((directory / "stderr.log").open("wb"))
                kwargs = {"start_new_session": True} if os.name == "posix" else {
                    "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
                processes.append(subprocess.Popen(args, cwd=cwd, env=rank_env,
                    stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL, shell=False, **kwargs))
            while True:
                codes = [p.poll() for p in processes]
                if any(code is not None and code != 0 for code in codes):
                    raise NativeBackendError(f"native local rank failed; exit codes={codes}; log={log_path}")
                if all(code == 0 for code in codes):
                    return time.monotonic() - started
                if time.monotonic() - started >= timeout:
                    raise NativeBackendError(f"native local ranks timed out after {timeout}s; log={log_path}")
                time.sleep(.02)
    finally:
        for process in processes:
            # Also remove orphaned descendants after a rank exits.
            _stop_process_tree(process)
