"""Fail-closed hardware-specific beam profile selection."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import math
from typing import Any, Mapping


RUNNABLE_STATUSES = frozenset({"measured", "bounded_from_measured"})
RUNTIME_KEYS = frozenset(
    {
        "b_micro",
        "stream1_concurrency",
        "stream3_ring_slots",
        "shard_count",
        "shard_capacity_scale_ppm",
        "stream4_batch_candidates",
        "stream4_trigger_candidates",
        "stream4_active_sort_slots",
        "final_materialize_chunk_candidates",
    }
)


@dataclass(frozen=True)
class HardwareKey:
    gpu_family: str
    vram_mib: int
    sm: int
    world_size: int


@dataclass(frozen=True)
class SelectedProfile:
    hardware: HardwareKey
    requested_beam: int
    profile_power: int
    backend: str
    model_class: str
    status: str
    evidence_id: str
    runtime: Mapping[str, int]


@dataclass(frozen=True)
class RuntimePlan:
    requested_beam: int
    effective_beam: int
    alignment_delta: int
    local_beam: int
    parent_batch: int
    stream3_batch_candidates: int
    shard_capacity_candidates: int
    profile_power: int
    evidence_id: str
    runtime: Mapping[str, int]


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def round_half_up_log2(beam_width: int) -> int:
    beam = _positive_int(beam_width, "beam_width")
    return int(math.floor(math.log2(beam) + 0.5))


def _hardware_from_record(record: Mapping[str, Any]) -> HardwareKey:
    return HardwareKey(
        str(record.get("gpu_family", "")),
        _positive_int(record.get("vram_mib"), "hardware.vram_mib"),
        _positive_int(record.get("sm"), "hardware.sm"),
        _positive_int(record.get("world_size"), "hardware.world_size"),
    )


def _runtime(record: Mapping[str, Any]) -> Mapping[str, int]:
    missing = RUNTIME_KEYS.difference(record)
    if missing:
        raise ValueError(f"profile runtime missing keys: {sorted(missing)}")
    return MappingProxyType({key: _positive_int(record[key], f"runtime.{key}") for key in RUNTIME_KEYS})


def select_profile(
    registry: Mapping[str, Any],
    hardware: HardwareKey,
    requested_beam: int,
    backend: str,
    model_class: str,
) -> SelectedProfile:
    if registry.get("schema_version") != 1 or not isinstance(registry.get("profiles"), list):
        raise ValueError("profile registry schema_version must be 1 with a profiles array")
    beam = _positive_int(requested_beam, "requested_beam")
    matches = [
        item
        for item in registry["profiles"]
        if isinstance(item, Mapping)
        and _hardware_from_record(item.get("hardware", {})) == hardware
        and item.get("backend") == backend
        and item.get("model_class") == model_class
    ]
    if len(matches) != 1:
        raise ValueError(
            "no exact profile registry for "
            f"hardware={hardware} backend={backend!r} model_class={model_class!r}"
        )
    family = matches[0]
    minimum = _positive_int(family.get("min_beam_power"), "min_beam_power")
    maximum = _positive_int(family.get("max_beam_power"), "max_beam_power")
    power = round_half_up_log2(beam)
    if power < minimum or power > maximum:
        raise ValueError(f"supported beam powers are {minimum}..{maximum}; requested maps to {power}")
    anchors = family.get("anchors")
    if not isinstance(anchors, Mapping) or str(power) not in anchors:
        raise ValueError(f"missing measured profile anchor for beam power {power}")
    anchor = anchors[str(power)]
    if not isinstance(anchor, Mapping):
        raise ValueError(f"profile anchor {power} must be an object")
    status = str(anchor.get("status", ""))
    if status not in RUNNABLE_STATUSES:
        raise ValueError(f"profile status {status or 'missing'} is not runnable")
    evidence_id = anchor.get("evidence_id")
    if not isinstance(evidence_id, str) or not evidence_id:
        raise ValueError("profile evidence_id must be a nonempty string")
    runtime = anchor.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("profile runtime must be an object")
    return SelectedProfile(
        hardware, beam, power, backend, model_class, status, evidence_id, _runtime(runtime)
    )


def _align(value: int, quantum: int) -> int:
    return ((value + quantum - 1) // quantum) * quantum


def derive_capacities(
    selected: SelectedProfile,
    requested_beam: int,
    move_count: int,
    output_dim: int,
) -> RuntimePlan:
    beam = _positive_int(requested_beam, "requested_beam")
    if beam != selected.requested_beam:
        raise ValueError("requested_beam differs from selected profile request")
    moves = _positive_int(move_count, "move_count")
    expected_class = "output1" if output_dim == 1 else "output_move_count" if output_dim == moves else ""
    if not expected_class or expected_class != selected.model_class:
        raise ValueError(
            f"output_dim={output_dim} and move_count={moves} do not match model_class={selected.model_class}"
        )
    runtime = selected.runtime
    ranks = selected.hardware.world_size
    shard_count = runtime["shard_count"]
    effective = _align(beam, ranks * shard_count * 1024)
    local_beam = effective // ranks
    parent_batch = runtime["b_micro"] // moves if expected_class == "output1" else runtime["b_micro"]
    if parent_batch <= 0:
        raise ValueError("b_micro is smaller than move_count")
    stream3 = parent_batch * moves * runtime["stream3_ring_slots"]
    logical_shard = (local_beam + shard_count - 1) // shard_count
    scaled = (logical_shard * runtime["shard_capacity_scale_ppm"] + 999_999) // 1_000_000
    capacity = _align(
        max(scaled, stream3, runtime["stream4_batch_candidates"], runtime["stream4_trigger_candidates"]),
        1024,
    )
    return RuntimePlan(
        beam,
        effective,
        effective - beam,
        local_beam,
        parent_batch,
        stream3,
        capacity,
        selected.profile_power,
        selected.evidence_id,
        runtime,
    )
