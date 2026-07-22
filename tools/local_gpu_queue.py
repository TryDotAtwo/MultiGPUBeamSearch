#!/usr/bin/env python3
"""Cooperative local GPU benchmark queue.

This is intentionally a small file-system queue. It lets independent Codex
agents serialize local GPU benchmarks without running a daemon.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


DEFAULT_QUEUE_DIR = Path(os.environ.get("CODEX_GPU_QUEUE_DIR", r"D:\100XH100\.gpu_queue"))
DEFAULT_COOLDOWN_SEC = float(os.environ.get("CODEX_GPU_QUEUE_COOLDOWN_SEC", "10"))


def now() -> float:
    return time.time()


def iso(ts: float | None = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now() if ts is None else ts))


def mkdir_lock(path: Path, stale_timeout_sec: float, label: str) -> bool:
    try:
        path.mkdir()
        return True
    except FileExistsError:
        if stale_timeout_sec <= 0:
            return False
        try:
            age = now() - path.stat().st_mtime
        except FileNotFoundError:
            return False
        if age < stale_timeout_sec:
            return False
        stale_name = path.with_name(f"{path.name}.stale-{int(now())}-{os.getpid()}")
        try:
            path.rename(stale_name)
            print(f"gpu_queue stale_lock_renamed={stale_name} original={path} label={label}", flush=True)
            path.mkdir()
            return True
        except OSError:
            return False


def release_lock(path: Path) -> None:
    try:
        path.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        shutil.rmtree(path, ignore_errors=True)


def read_int(path: Path, default: int) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, ValueError):
        return default


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def acquire_ticket(queue_dir: Path, label: str, command: list[str], stale_timeout_sec: float) -> tuple[int, Path]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    tickets_dir = queue_dir / "tickets"
    tickets_dir.mkdir(exist_ok=True)
    ticket_lock = queue_dir / "ticket.lock"
    while not mkdir_lock(ticket_lock, stale_timeout_sec, label):
        time.sleep(0.2)
    try:
        next_path = queue_dir / "next_ticket.txt"
        ticket = read_int(next_path, 0)
        atomic_write(next_path, f"{ticket + 1}\n")
        ticket_path = tickets_dir / f"{ticket:012d}.json"
        metadata = {
            "ticket": ticket,
            "label": label,
            "pid": os.getpid(),
            "created_at": iso(),
            "cwd": str(Path.cwd()),
            "command": command,
        }
        atomic_write(ticket_path, json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        return ticket, ticket_path
    finally:
        release_lock(ticket_lock)


def iter_ticket_numbers(queue_dir: Path) -> Iterable[int]:
    tickets_dir = queue_dir / "tickets"
    if not tickets_dir.exists():
        return []
    nums: list[int] = []
    for path in tickets_dir.glob("*.json"):
        try:
            nums.append(int(path.stem))
        except ValueError:
            continue
    return sorted(nums)


def earlier_tickets_exist(queue_dir: Path, ticket: int) -> bool:
    for current in iter_ticket_numbers(queue_dir):
        if current < ticket:
            return True
    return False


def query_nvidia_smi() -> list[tuple[int, int, int]]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return []
    cmd = [
        exe,
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if proc.returncode != 0:
        print(f"gpu_queue nvidia_smi_failed rc={proc.returncode} stderr={proc.stderr.strip()}", flush=True)
        return []
    rows: list[tuple[int, int, int]] = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            rows.append((int(parts[0]), int(float(parts[1])), int(float(parts[2]))))
        except ValueError:
            continue
    return rows


def print_gpu_snapshot(prefix: str) -> None:
    rows = query_nvidia_smi()
    if not rows:
        print(f"gpu_queue {prefix}_gpu_snapshot=unavailable", flush=True)
        return
    for index, mem_mib, util_pct in rows:
        print(
            f"gpu_queue {prefix}_gpu index={index} memory_used_mib={mem_mib} util_pct={util_pct}",
            flush=True,
        )


def wait_gpu_idle(max_mem_mib: int, max_util_pct: int, stable_polls: int, poll_sec: float) -> None:
    stable = 0
    while stable < stable_polls:
        rows = query_nvidia_smi()
        if not rows:
            print("gpu_queue idle_check=unavailable continuing_without_idle_gate", flush=True)
            return
        busy = [
            (index, mem_mib, util_pct)
            for index, mem_mib, util_pct in rows
            if mem_mib > max_mem_mib or util_pct > max_util_pct
        ]
        if busy:
            stable = 0
            busy_text = ";".join(f"{i}:mem={m}:util={u}" for i, m, u in busy)
            print(f"gpu_queue idle_wait busy={busy_text}", flush=True)
            time.sleep(poll_sec)
            continue
        stable += 1
        print(f"gpu_queue idle_poll_ok stable={stable}/{stable_polls}", flush=True)
        if stable < stable_polls:
            time.sleep(poll_sec)


def wait_for_turn(
    queue_dir: Path,
    ticket: int,
    label: str,
    cooldown_sec: float,
    stale_timeout_sec: float,
) -> Path:
    active_lock = queue_dir / "active.lock"
    while True:
        if earlier_tickets_exist(queue_dir, ticket):
            time.sleep(0.5)
            continue
        if not mkdir_lock(active_lock, stale_timeout_sec, label):
            time.sleep(0.5)
            continue
        last_done = 0.0
        last_done_path = queue_dir / "last_done_epoch.txt"
        try:
            last_done = float(last_done_path.read_text(encoding="utf-8").strip())
        except (FileNotFoundError, ValueError):
            pass
        remaining = cooldown_sec - (now() - last_done)
        if remaining > 0:
            print(f"gpu_queue cooldown_sleep_sec={remaining:.3f} ticket={ticket} label={label}", flush=True)
            time.sleep(remaining)
        metadata = {
            "ticket": ticket,
            "label": label,
            "pid": os.getpid(),
            "started_at": iso(),
        }
        atomic_write(active_lock / "owner.json", json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        return active_lock


def show_status(queue_dir: Path) -> int:
    queue_dir.mkdir(parents=True, exist_ok=True)
    print(f"gpu_queue_dir={queue_dir}")
    active = queue_dir / "active.lock" / "owner.json"
    if active.exists():
        print("active=" + active.read_text(encoding="utf-8").strip().replace("\n", " "))
    else:
        print("active=none")
    tickets = list(iter_ticket_numbers(queue_dir))
    print(f"pending_tickets={len(tickets)}")
    for ticket in tickets[:20]:
        path = queue_dir / "tickets" / f"{ticket:012d}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print(f"ticket={ticket} unreadable={path}")
            continue
        print(f"ticket={ticket} label={data.get('label', '')} pid={data.get('pid', '')} created_at={data.get('created_at', '')}")
    print_gpu_snapshot("status")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--label", default="gpu-task")
    parser.add_argument("--cooldown-sec", type=float, default=DEFAULT_COOLDOWN_SEC)
    parser.add_argument("--stale-timeout-sec", type=float, default=24 * 60 * 60)
    parser.add_argument("--wait-gpu-idle", action="store_true")
    parser.add_argument("--max-gpu-memory-mib", type=int, default=1024)
    parser.add_argument("--max-gpu-util-pct", type=int, default=10)
    parser.add_argument("--idle-stable-polls", type=int, default=2)
    parser.add_argument("--idle-poll-sec", type=float, default=5)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    queue_dir = args.queue_dir
    if args.status:
        return show_status(queue_dir)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error("command is required unless --status is used")

    ticket, ticket_path = acquire_ticket(queue_dir, args.label, command, args.stale_timeout_sec)
    print(f"gpu_queue ticket={ticket} label={args.label} ticket_path={ticket_path}", flush=True)
    active_lock: Path | None = None
    rc = 125
    try:
        active_lock = wait_for_turn(queue_dir, ticket, args.label, args.cooldown_sec, args.stale_timeout_sec)
        print(f"gpu_queue start ticket={ticket} label={args.label} at={iso()}", flush=True)
        print_gpu_snapshot("before")
        if args.wait_gpu_idle:
            wait_gpu_idle(args.max_gpu_memory_mib, args.max_gpu_util_pct, args.idle_stable_polls, args.idle_poll_sec)
        rc = subprocess.run(command, check=False).returncode
        print_gpu_snapshot("after")
        print(f"gpu_queue done ticket={ticket} label={args.label} rc={rc} at={iso()}", flush=True)
        return rc
    finally:
        try:
            atomic_write(queue_dir / "last_done_epoch.txt", f"{now():.6f}\n")
        except OSError:
            pass
        if active_lock is not None:
            release_lock(active_lock)
        try:
            ticket_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    sys.exit(main())
