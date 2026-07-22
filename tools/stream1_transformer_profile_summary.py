#!/usr/bin/env python3
"""Summarize Stream1 transformer GPU kernels by stable performance family."""

from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from typing import Iterable


@dataclass(frozen=True)
class KernelRow:
    name: str
    duration_ns: int


@dataclass(frozen=True)
class KernelFamilySummary:
    family: str
    launches: int
    total_ns: int
    names: tuple[str, ...]


def read_nsys_kernel_rows(connection: sqlite3.Connection) -> list[KernelRow]:
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    required = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"}
    missing = sorted(required - tables)
    if missing:
        raise ValueError(f"missing Nsight SQLite table: {', '.join(missing)}")
    rows = connection.execute(
        "SELECT strings.value, kernels.end - kernels.start "
        "FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernels "
        "JOIN StringIds AS strings ON strings.id = kernels.demangledName"
    )
    return [KernelRow(str(name), int(duration_ns)) for name, duration_ns in rows]


def _kernel_family(name: str) -> str:
    lowered = name.lower()
    if "attention_kernel_batched_impl" in lowered or "fmha" in lowered:
        return "attention"
    if "kernel2" in lowered:
        return "gemm_fused"
    if "cutlass" in lowered and "kernel" in lowered:
        return "gemm_plain"
    if "layernorm" in lowered:
        return "layernorm"
    if "build_input" in lowered:
        return "input_build"
    if "score" in lowered and "quant" in lowered:
        return "score_quantize"
    return "other"


def summarize_kernels(rows: Iterable[KernelRow]) -> list[KernelFamilySummary]:
    grouped: dict[str, tuple[int, int, set[str]]] = {}
    row_count = 0
    for row in rows:
        row_count += 1
        if row.duration_ns < 0:
            raise ValueError("duration_ns must be non-negative")
        family = _kernel_family(row.name)
        launches, total_ns, names = grouped.get(family, (0, 0, set()))
        names.add(row.name)
        grouped[family] = (launches + 1, total_ns + row.duration_ns, names)
    if row_count == 0:
        raise ValueError("no kernel rows")
    summaries = [
        KernelFamilySummary(family, launches, total_ns, tuple(sorted(names)))
        for family, (launches, total_ns, names) in grouped.items()
    ]
    return sorted(summaries, key=lambda item: (-item.total_ns, item.family))
