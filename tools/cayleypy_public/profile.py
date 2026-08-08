"""Measured 2xT4 profile derivation and safe public-run preflight payloads."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

from tools.kaggle_t4_mlp_profiles import align_beam


_REQUIRED_RUNTIME_KEYS = frozenset(
    {
        "b_micro",
        "stream1_concurrency",
        "stream3_ring_slots",
        "shard_count",
        "shard_capacity_scale_ppm",
        "stream4_batch_candidates",
        "stream4_trigger_candidates",
        "stream4_active_sort_slots",
    }
)


@dataclass(frozen=True)
class RuntimePlan:
    requested_beam: int
    effective_beam: int
    alignment_delta: int
    profile_power: int
    model_class: str
    local_beam: int
    parent_batch: int
    stream3_batch_candidates: int
    shard_capacity_candidates: int
    runtime: Mapping[str, int]
    cross_puzzle_profile_note: str


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _aligned(value: int, alignment: int = 1024) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _runtime_values(profile: Mapping[str, Any]) -> dict[str, int]:
    runtime = profile.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError("profile runtime must be an object")
    missing = _REQUIRED_RUNTIME_KEYS.difference(runtime)
    if missing:
        raise ValueError(f"profile runtime missing keys: {sorted(missing)}")
    return {name: _positive_int(runtime[name], f"runtime.{name}") for name in _REQUIRED_RUNTIME_KEYS}


def derive_runtime(
    profile: Mapping[str, Any],
    beam_width: int,
    output_dim: int,
    move_count: int,
    world_size: int = 2,
) -> RuntimePlan:
    """Derive move-count-aware batches without replacing the requested beam."""
    requested_beam = _positive_int(beam_width, "beam_width")
    moves = _positive_int(move_count, "move_count")
    ranks = _positive_int(world_size, "world_size")
    if ranks != 2:
        raise ValueError("world_size must be exactly 2 for kaggle_2xt4")
    if profile.get("validation_status") != "measured":
        raise ValueError("profile validation_status must be measured")

    model_class = profile.get("model_class")
    if output_dim == 1:
        expected_class = "output1"
    elif output_dim == moves:
        expected_class = "output_move_count"
    else:
        raise ValueError("output_dim must be 1 or equal move_count")
    if model_class != expected_class:
        raise ValueError(f"profile model_class must be {expected_class!r}; got {model_class!r}")

    runtime = _runtime_values(profile)
    effective_beam = align_beam(requested_beam, ranks, runtime["shard_count"])
    selected_effective = profile.get("effective_beam")
    if selected_effective is not None and selected_effective != effective_beam:
        raise ValueError("profile effective_beam does not match requested beam alignment")
    parent_batch = runtime["b_micro"] // moves if expected_class == "output1" else runtime["b_micro"]
    if parent_batch <= 0:
        raise ValueError("runtime.b_micro is smaller than move_count for output1")
    stream3_batch = parent_batch * moves * runtime["stream3_ring_slots"]
    local_beam = effective_beam // ranks
    logical_shard = (local_beam + runtime["shard_count"] - 1) // runtime["shard_count"]
    scaled_shard = (logical_shard * runtime["shard_capacity_scale_ppm"] + 999_999) // 1_000_000
    capacity = _aligned(
        max(
            scaled_shard,
            stream3_batch,
            runtime["stream4_batch_candidates"],
            runtime["stream4_trigger_candidates"],
        )
    )
    profile_power = _positive_int(profile.get("profile_power"), "profile_power")
    return RuntimePlan(
        requested_beam=requested_beam,
        effective_beam=effective_beam,
        alignment_delta=effective_beam - requested_beam,
        profile_power=profile_power,
        model_class=expected_class,
        local_beam=local_beam,
        parent_batch=parent_batch,
        stream3_batch_candidates=stream3_batch,
        shard_capacity_candidates=capacity,
        runtime=MappingProxyType(dict(runtime)),
        cross_puzzle_profile_note="measured_24_move_seed" if moves != 24 else "",
    )


def serialize_preflight(
    plan: RuntimePlan,
    profile: Mapping[str, Any],
    move_count: int,
    history_ram_bytes: int,
    history_disk_bytes: int,
    tmp_free_bytes: int,
) -> dict[str, Any]:
    """Return a JSON-ready, fail-closed preflight record for one public run."""
    if profile.get("validation_status") != "measured":
        raise ValueError("profile validation_status must be measured")
    moves = _positive_int(move_count, "move_count")
    ram_budget = _positive_int(history_ram_bytes, "history_ram_bytes")
    disk_budget = _positive_int(history_disk_bytes, "history_disk_bytes")
    tmp_free = _positive_int(tmp_free_bytes, "tmp_free_bytes")
    if tmp_free < disk_budget:
        raise ValueError("/tmp free disk is smaller than the history disk budget")
    evidence = profile.get("evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("measured profile evidence must be an object")
    hardware = profile.get("hardware")
    if hardware != "kaggle_2xt4":
        raise ValueError(f"profile hardware must be 'kaggle_2xt4'; got {hardware!r}")
    evidence_version = _positive_int(profile.get("profile_registry_schema_version"), "profile_registry_schema_version")
    return {
        "profile_evidence_version": evidence_version,
        "profile_evidence": dict(evidence),
        "hardware": hardware,
        "move_count": moves,
        "requested_beam": plan.requested_beam,
        "effective_beam": plan.effective_beam,
        "alignment_delta": plan.alignment_delta,
        "profile_power": plan.profile_power,
        "model_class": plan.model_class,
        "local_beam": plan.local_beam,
        "parent_batch": plan.parent_batch,
        "stream3_batch_candidates": plan.stream3_batch_candidates,
        "shard_capacity_candidates": plan.shard_capacity_candidates,
        "runtime": dict(plan.runtime),
        "cross_puzzle_profile_note": plan.cross_puzzle_profile_note,
        "capacity_derivation": {
            "stream3_batch_candidates": plan.stream3_batch_candidates,
            "stream4_batch_candidates": plan.runtime["stream4_batch_candidates"],
            "stream4_trigger_candidates": plan.runtime["stream4_trigger_candidates"],
            "shard_capacity_candidates": plan.shard_capacity_candidates,
            "alignment": 1024,
        },
        "history_budgets": {"ram_bytes": ram_budget, "disk_bytes": disk_budget},
        "tmp_free_bytes": tmp_free,
    }