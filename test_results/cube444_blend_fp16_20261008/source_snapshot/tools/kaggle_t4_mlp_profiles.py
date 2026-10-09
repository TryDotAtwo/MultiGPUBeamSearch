#!/usr/bin/env python3
"""Select measured Kaggle 2xT4 runtime profiles without changing user beam."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any


MIN_PROFILE_POWER = 16
MAX_PROFILE_POWER = 25
SUPPORTED_HARDWARE = "kaggle_2xt4"
SUPPORTED_STATUSES = frozenset({"seed", "measured", "bounded_from_measured"})


def round_half_up_log2(beam_width: int, minimum: int = MIN_PROFILE_POWER, maximum: int = MAX_PROFILE_POWER) -> int:
    """Return the nearest supported power-of-two profile anchor."""
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError(f"beam_width must be a positive integer; got {beam_width!r}")
    power = int(math.floor(math.log2(beam_width) + 0.5))
    if minimum > maximum:
        raise ValueError("minimum profile power must not exceed maximum")
    return min(maximum, max(minimum, power))


def align_beam(
    beam_width: int,
    world_size: int,
    shard_count: int,
    alignment: int = 1024,
) -> int:
    """Align beam to the existing distributed shard layout."""
    values = {
        "beam_width": beam_width,
        "world_size": world_size,
        "shard_count": shard_count,
        "alignment": alignment,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer; got {value!r}")
    quantum = world_size * shard_count * alignment
    return ((beam_width + quantum - 1) // quantum) * quantum


def _model_class(output_dim: int, move_count: int) -> str:
    if output_dim == 1:
        return "output1"
    if output_dim == move_count:
        return "output_move_count"
    raise ValueError(
        "unsupported output_dim: "
        f"got output_dim={output_dim}, move_count={move_count}; "
        "only output_dim=1 or output_dim=move_count is supported"
    )


def select_profile(
    profiles: dict[str, Any],
    beam_width: int,
    output_dim: int,
    move_count: int,
) -> dict[str, Any]:
    """Select a runtime profile while preserving the requested beam."""
    if profiles.get("hardware") != SUPPORTED_HARDWARE:
        raise ValueError(
            f"profile registry hardware must be {SUPPORTED_HARDWARE!r}; "
            f"got {profiles.get('hardware')!r}"
        )
    model_class = _model_class(output_dim, move_count)
    model_profiles = profiles.get("profiles", {}).get(model_class, {})
    try:
        available_powers = sorted(int(power) for power in model_profiles)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid profile anchors for model_class={model_class}") from exc
    if not available_powers:
        raise ValueError(f"missing profiles for model_class={model_class}")
    profile_power = round_half_up_log2(beam_width, available_powers[0], available_powers[-1])
    try:
        profile = deepcopy(model_profiles[str(profile_power)])
    except KeyError as exc:
        raise ValueError(
            f"missing profile for model_class={model_class} power={profile_power}"
        ) from exc
    status = profile.get("validation_status")
    if status not in SUPPORTED_STATUSES:
        raise ValueError(
            f"profile validation_status must be one of {sorted(SUPPORTED_STATUSES)}; "
            f"got {status!r}"
        )
    runtime = profile.get("runtime")
    if not isinstance(runtime, dict):
        raise ValueError(
            f"profile runtime must be an object for model_class={model_class} "
            f"power={profile_power}"
        )
    shard_count = runtime.get("shard_count")
    effective_beam = align_beam(beam_width, 2, shard_count)
    return {
        "requested_beam": beam_width,
        "effective_beam": effective_beam,
        "alignment_delta": effective_beam - beam_width,
        "profile_power": profile_power,
        "profile_anchor_beam": 2**profile_power,
        "model_class": model_class,
        "hardware": profiles["hardware"],
        "backend": profiles.get("backend", "mlp"),
        "profile_registry_schema_version": profiles.get("schema_version"),
        **profile,
    }


def validate_manifest(
    manifest: dict[str, Any],
    state_len: int,
    num_classes: int,
    move_count: int,
) -> str:
    """Validate a Stream1 MLP manifest against the selected generator."""
    for field, expected in (("state_len", state_len), ("num_classes", num_classes)):
        actual = manifest.get(field)
        if actual != expected:
            raise ValueError(
                f"manifest {field} mismatch: observed={actual!r}, expected={expected!r}"
            )
    normalization = manifest.get("normalization")
    if normalization not in {"batchnorm_folded", "layernorm"}:
        raise ValueError(
            "unsupported MLP manifest normalization: "
            f"observed={normalization!r}, expected batchnorm_folded or layernorm"
        )
    output_dim = manifest.get("output_dim")
    if not isinstance(output_dim, int):
        raise ValueError(f"manifest output_dim must be an integer; got {output_dim!r}")
    if not isinstance(move_count, int) or move_count <= 1:
        raise ValueError(f"move_count must be an integer greater than one; got {move_count!r}")
    return _model_class(output_dim, move_count)


def supported_model_header() -> str:
    """Return the exact reader-facing model support contract."""
    return """# SUPPORTED MODELS

This notebook does not support an arbitrary PyTorch architecture. It accepts
only these checkpoint layouts already implemented by `export_stream1_mlp.py`:

1. PilgrimAttnRes-style BatchNorm MLP (`batchnorm-folded`).
2. ResMLPDistance-style LayerNorm MLP (`resmlp-layernorm`).

The exported head must have `output_dim=1` or `output_dim=move_count` (24 for
Megaminx). Every other checkpoint layout or output width fails before launch.
"""
