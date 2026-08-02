"""Bundled single-node static torchrun-compatible process launcher."""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--nnodes", type=int, required=True)
    parser.add_argument("--nproc-per-node", type=int, required=True)
    parser.add_argument("--node-rank", type=int, required=True)
    parser.add_argument("--rdzv-backend", required=True)
    parser.add_argument("--rdzv-endpoint", required=True)
    parser.add_argument("--rdzv-id", required=True)
    parser.add_argument("--no-python", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(list(argv))
    if args.nnodes != 1 or args.node_rank != 0:
        parser.error("bundled launcher supports exactly one node with node rank 0")
    if args.nproc_per_node <= 0:
        parser.error("--nproc-per-node must be positive")
    if not args.no_python:
        parser.error("bundled launcher requires --no-python")
    if not args.command:
        parser.error("missing native command")
    return args


def launch(args: argparse.Namespace) -> int:
    world_size = args.nproc_per_node
    endpoint = args.rdzv_endpoint.rsplit(":", 1)
    if len(endpoint) != 2 or not endpoint[1].isdigit():
        raise ValueError("invalid rendezvous endpoint")
    base_env = dict(os.environ)
    base_env.update({
        "MASTER_ADDR": endpoint[0],
        "MASTER_PORT": endpoint[1],
        "WORLD_SIZE": str(world_size),
        "LOCAL_WORLD_SIZE": str(world_size),
        "GROUP_RANK": "0",
        "ROLE_WORLD_SIZE": str(world_size),
        "TORCHELASTIC_RUN_ID": args.rdzv_id,
        "TORCHELASTIC_RESTART_COUNT": "0",
        "TORCHELASTIC_MAX_RESTARTS": "0",
    })
    processes: list[subprocess.Popen[bytes]] = []
    stopping = False

    def stop_all(_signum: int | None = None, _frame=None) -> None:
        nonlocal stopping
        stopping = True
        for process in processes:
            if process.poll() is None:
                process.terminate()

    previous = {sig: signal.signal(sig, stop_all) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for local_rank in range(world_size):
            env = dict(base_env)
            env.update({
                "RANK": str(local_rank),
                "LOCAL_RANK": str(local_rank),
                "ROLE_RANK": str(local_rank),
            })
            processes.append(subprocess.Popen(args.command, env=env))
        first_failure = 0
        while any(process.poll() is None for process in processes):
            for process in processes:
                code = process.poll()
                if code not in (None, 0) and first_failure == 0:
                    first_failure = code
                    stop_all()
            if stopping and first_failure == 0:
                first_failure = 128 + signal.SIGTERM
            time.sleep(0.02)
        codes = [process.wait() for process in processes]
        return first_failure or next((code for code in codes if code), 0)
    finally:
        stop_all()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return launch(parse_args(sys.argv[1:] if argv is None else argv))
    except (OSError, ValueError) as exc:
        print(f"bundled_torchrun_error={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())