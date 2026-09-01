"""Explicit process-local configuration, independent of competition launchers."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class NativeOptions:
    source_dir: Path | None = None
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "cayleypy-native")
    cutlass_dir: Path | None = None
    runner_path: Path | None = None
    devices: tuple[int, ...] | None = None
    timeout_seconds: float = 3600.0
    build_timeout_seconds: float = 1800.0
    touch_bfs_radius: int = 0
    touch_bfs_max_entries: int = 1_048_576
    build_jobs: int = 2
    warn_on_fallback: bool = True

    def __post_init__(self):
        for name in ("source_dir", "cache_dir", "cutlass_dir", "runner_path"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, Path(value).expanduser().resolve())
        if self.devices is not None:
            values = tuple(self.devices)
            if not values or any(type(v) is not int or v < 0 for v in values) or len(set(values)) != len(values):
                raise ValueError("devices must be a nonempty sequence of distinct nonnegative CUDA indices")
            object.__setattr__(self, "devices", values)
        import math
        for name in ("timeout_seconds", "build_timeout_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.touch_bfs_radius) is not int or not 0 <= self.touch_bfs_radius <= 12:
            raise ValueError("touch_bfs_radius must be an integer in [0, 12]")
        if type(self.touch_bfs_max_entries) is not int or self.touch_bfs_max_entries <= 0:
            raise ValueError("touch_bfs_max_entries must be a positive integer")
        if type(self.build_jobs) is not int or self.build_jobs <= 0:
            raise ValueError("build_jobs must be a positive integer")


@dataclass(frozen=True)
class NativeOutcome:
    path: tuple[int, ...] | None
    elapsed_seconds: float
    effective_beam_width: int | None
    run_dir: Path
    metadata: dict[str, Any] = field(default_factory=dict)
