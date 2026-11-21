"""Shared Python side of the strict CNOT acceptance event protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sys

try:
    from .gpu_identity import CudaDeviceObservation, observe_current_cuda_context
except ImportError:  # standalone hosted protocol fixture
    from gpu_identity import CudaDeviceObservation, observe_current_cuda_context


EVENT_SCHEMA = "org.trydotatwo.multigpubeamsearch.cnot.acceptance"
EVENT_VERSION = 1
EVENT_PREFIX = "CNOT_ACCEPTANCE_EVENT "
TOKEN_RE = re.compile(r"[A-Za-z0-9_-]+\Z")
SHA40_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
SHA64_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
_PINNED_GPU_OBSERVATIONS = {}


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
    if len(set(gpu_uuids)) != len(gpu_uuids):
        raise RuntimeError("CNOT acceptance requires unique physical GPU UUIDs")
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


def gpu_observation_for_rank(
    rank, world_size, *, device=None, torch_module=None, driver=None,
    observation_provider=None,
):
    identity = acceptance_identity()
    if identity is None:
        return None
    rank, world_size = int(rank), int(world_size)
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if rank < 0 or rank >= world_size or local_rank < 0 \
            or local_rank >= len(identity["gpu_uuids"]) \
            or len(identity["gpu_uuids"]) < world_size:
        raise RuntimeError(
            f"CNOT acceptance has {len(identity['gpu_uuids'])} GPU UUIDs for rank {rank}/{world_size}"
        )
    expected_uuid = identity["gpu_uuids"][local_rank]
    if device is None:
        if torch_module is None:
            import torch as torch_module
        device = torch_module.device("cuda")
    provider = observe_current_cuda_context if observation_provider is None else observation_provider
    observation = provider(device, torch_module=torch_module, driver=driver)
    required = ("device_ordinal", "gpu_uuid", "context_handle", "thread_id")
    if not isinstance(observation, CudaDeviceObservation) and any(
        not hasattr(observation, name) for name in required
    ):
        raise RuntimeError("CUDA identity provider returned an invalid typed observation")
    if observation.context_handle <= 0 or observation.device_ordinal < 0:
        raise RuntimeError("CUDA identity provider returned an empty active-context observation")
    if observation.gpu_uuid != expected_uuid:
        raise RuntimeError(
            f"CNOT acceptance rank {rank} observed GPU {observation.gpu_uuid}, expected {expected_uuid}"
        )
    cache_key = (rank, local_rank)
    previous = _PINNED_GPU_OBSERVATIONS.get(cache_key)
    if previous is not None and previous != observation:
        raise RuntimeError("CUDA context changed after CNOT admission")
    _PINNED_GPU_OBSERVATIONS[cache_key] = observation
    return observation


def gpu_uuid_for_rank(rank, world_size, *, device=None, torch_module=None, driver=None):
    observation = gpu_observation_for_rank(
        rank, world_size, device=device, torch_module=torch_module, driver=driver
    )
    return None if observation is None else observation.gpu_uuid


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
    uuid = gpu_uuid_for_rank(rank, size, device=world.device)
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
