"""Shared Python side of the strict CNOT acceptance event protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys


EVENT_SCHEMA = "org.trydotatwo.multigpubeamsearch.cnot.acceptance"
EVENT_VERSION = 1
EVENT_PREFIX = "CNOT_ACCEPTANCE_EVENT "
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA40_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
SHA64_RE = re.compile(r"[0-9a-fA-F]{64}\Z")


def acceptance_identity():
    run_id = os.environ.get("MGBS_CNOT_ACCEPTANCE_RUN_ID")
    case_id = os.environ.get("MGBS_CNOT_ACCEPTANCE_CASE")
    if run_id is None and case_id is None:
        return None
    if not run_id or not case_id or not TOKEN_RE.fullmatch(run_id) or not TOKEN_RE.fullmatch(case_id):
        raise RuntimeError("CNOT acceptance run and case IDs must be path-safe tokens")
    source_sha = os.environ.get("MGBS_CNOT_ACCEPTANCE_SOURCE_SHA", "").lower()
    wheel_sha256 = os.environ.get("MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256", "").lower()
    if not SHA40_RE.fullmatch(source_sha) or not SHA64_RE.fullmatch(wheel_sha256):
        raise RuntimeError("CNOT acceptance requires the source and installed wheel SHA identities")
    raw_uuids = os.environ.get("MGBS_CNOT_ACCEPTANCE_GPU_UUIDS", "")
    gpu_uuids = [item.strip() for item in raw_uuids.split(",") if item.strip()]
    if not gpu_uuids or any(any(char.isspace() for char in item) for item in gpu_uuids):
        raise RuntimeError("CNOT acceptance requires physical GPU UUIDs")
    evidence_dir = os.environ.get("CNOT_RUST_EVIDENCE")
    if not evidence_dir:
        raise RuntimeError("CNOT_RUST_EVIDENCE must identify the per-rank receipt root")
    return {
        "run_id": run_id,
        "case_id": case_id,
        "source_sha": source_sha,
        "wheel_sha256": wheel_sha256,
        "gpu_uuids": gpu_uuids,
        "evidence_dir": Path(evidence_dir),
    }


def gpu_uuid_for_rank(rank, world_size):
    identity = acceptance_identity()
    if identity is None:
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if local_rank < 0 or local_rank >= len(identity["gpu_uuids"]) or len(identity["gpu_uuids"]) < world_size:
        raise RuntimeError(
            f"CNOT acceptance has {len(identity['gpu_uuids'])} GPU UUIDs for rank {rank}/{world_size}"
        )
    return identity["gpu_uuids"][local_rank]


def report_sha256(report):
    payload = dict(report)
    payload.pop("acceptance", None)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _event(identity, kind, world, fields):
    rank = int(world.rank)
    size = int(world.size)
    pid = os.getpid()
    device = str(world.device)
    uuid = gpu_uuid_for_rank(rank, size)
    if uuid is None:
        raise RuntimeError("acceptance event has no physical GPU UUID")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in fields.items()):
        raise TypeError("CNOT acceptance event fields must be strings")
    return {
        "schema": EVENT_SCHEMA,
        "version": EVENT_VERSION,
        "run_id": identity["run_id"],
        "case_id": identity["case_id"],
        "source_sha": identity["source_sha"],
        "wheel_sha256": identity["wheel_sha256"],
        "kind": kind,
        "rank": rank,
        "world": size,
        "pid": pid,
        "device": device,
        "gpu_uuid": uuid,
        "fields": {key: fields[key] for key in sorted(fields)},
    }


def emit_event(kind, world, fields):
    identity = acceptance_identity()
    if identity is None:
        return None
    event = _event(identity, kind, world, fields)
    encoded = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
    receipt_dir = identity["evidence_dir"] / identity["run_id"] / identity["case_id"]
    receipt_dir.mkdir(parents=True, exist_ok=True)
    path = receipt_dir / f"rank-{event['rank']}-pid-{event['pid']}.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
    print(EVENT_PREFIX + encoded, file=sys.stderr, flush=True)
    return event
