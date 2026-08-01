"""Compute-node hardware and immutable payload preflight."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import re
from typing import Iterable, Mapping


@dataclass(frozen=True)
class GpuInfo:
    index: int
    name: str
    family: str
    vram_mib: int
    sm: int
    driver_major: int


def _family(name: str) -> str:
    upper = name.upper()
    for family in ("A100", "H100", "L4", "T4"):
        if family in upper:
            return family
    if "RTX PRO 6000" in upper and "BLACKWELL" in upper:
        return "RTX_PRO_6000_BLACKWELL"
    match = re.search(r"RTX\s+([0-9]{4})", upper)
    if match:
        return f"RTX_{match.group(1)}"
    raise ValueError(f"unsupported GPU family: {name}")


def inspect_gpus(csv_text: str) -> tuple[GpuInfo, ...]:
    result = []
    for line_number, raw in enumerate(csv_text.splitlines(), 1):
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split(",")]
        if len(fields) != 5:
            raise ValueError(f"invalid nvidia-smi row {line_number}: expected 5 fields")
        index, name, memory, capability, driver = fields
        memory_match = re.fullmatch(r"([0-9]+)\s+MiB", memory)
        capability_match = re.fullmatch(r"([0-9]+)\.([0-9]+)", capability)
        driver_match = re.match(r"([0-9]+)\.", driver)
        if not memory_match or not capability_match or not driver_match:
            raise ValueError(f"invalid nvidia-smi row {line_number}")
        result.append(GpuInfo(int(index), name, _family(name), int(memory_match.group(1)), int(capability_match.group(1)) * 10 + int(capability_match.group(2)), int(driver_match.group(1))))
    if not result:
        raise ValueError("nvidia-smi returned no GPUs")
    return tuple(result)


def verify_payload_hashes(root: Path, payloads: Mapping[str, str]) -> None:
    for relative, expected in payloads.items():
        path = root / relative
        if not path.is_file():
            raise ValueError(f"missing payload: {relative}")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"sha256 mismatch for {relative}: expected={expected} actual={actual}")


def validate_allocation(manifest: Mapping[str, object], plan: Mapping[str, int], gpus: Iterable[GpuInfo], disk_free_bytes: int) -> dict[str, object]:
    rows = tuple(gpus)
    expected_count = int(plan["world_size"])
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} GPUs; observed {len(rows)}")
    if len({gpu.sm for gpu in rows}) != 1 or len({gpu.family for gpu in rows}) != 1 or len({gpu.vram_mib for gpu in rows}) != 1:
        raise ValueError("mixed GPU architecture or memory class is unsupported")
    archive_sm = int(manifest["archive_sm"])
    observed_sm = rows[0].sm
    if archive_sm != observed_sm:
        raise ValueError(f"archive sm{archive_sm} does not match allocated sm{observed_sm}")
    configured = manifest.get("gpu_families")
    allowed_families = {str(value) for value in configured} if isinstance(configured, list) else {str(manifest["gpu_family"])}
    if rows[0].family not in allowed_families:
        raise ValueError(f"archive GPU families {sorted(allowed_families)} do not include {rows[0].family}")
    required_vram = int(plan["required_vram_mib"])
    if any(gpu.vram_mib < required_vram for gpu in rows):
        raise ValueError(f"insufficient VRAM: required {required_vram} MiB per GPU")
    minimum_driver = int(manifest["minimum_driver_major"])
    if any(gpu.driver_major < minimum_driver for gpu in rows):
        raise ValueError(f"driver major must be at least {minimum_driver}")
    required_disk = int(plan["history_disk_bytes"])
    if disk_free_bytes < required_disk:
        raise ValueError(f"scratch free bytes {disk_free_bytes} is below required {required_disk}")
    return {"status": "ok", "archive_sm": archive_sm, "world_size": expected_count, "required_vram_mib": required_vram, "scratch_free_bytes": disk_free_bytes, "gpus": [asdict(gpu) for gpu in rows]}


def write_record(path: Path, record: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
