import hashlib
from pathlib import Path

import pytest

from portable.megaminx_cluster.scripts.preflight import (
    inspect_gpus,
    validate_allocation,
    verify_payload_hashes,
)


GPU_CSV = """0, NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0, 570.124.06
1, NVIDIA A100-SXM4-40GB, 40960 MiB, 8.0, 570.124.06
"""


def manifest(tmp_path):
    payload = tmp_path / "bin" / "production_runner"
    payload.parent.mkdir()
    payload.write_bytes(b"runner")
    return {
        "archive_sm": 80,
        "gpu_family": "A100",
        "vram_mib": 40960,
        "minimum_driver_major": 570,
        "payloads": {"bin/production_runner": hashlib.sha256(b"runner").hexdigest()},
    }


def plan():
    return {"world_size": 2, "required_vram_mib": 38000, "history_disk_bytes": 1000}


def test_inspects_nvidia_smi_csv():
    gpus = inspect_gpus(GPU_CSV)
    assert [(g.index, g.family, g.vram_mib, g.sm, g.driver_major) for g in gpus] == [
        (0, "A100", 40960, 80, 570),
        (1, "A100", 40960, 80, 570),
    ]


@pytest.mark.parametrize(
    ("csv", "message"),
    [
        (GPU_CSV.splitlines()[0] + "\n", "expected 2 GPUs"),
        (GPU_CSV.replace("8.0, 570", "9.0, 570", 1), "mixed GPU architecture"),
        (GPU_CSV.replace("40960 MiB", "24576 MiB"), "insufficient VRAM"),
        (GPU_CSV.replace("570.124.06", "550.54.15"), "driver major"),
    ],
)
def test_allocation_fails_closed(tmp_path, csv, message):
    with pytest.raises(ValueError, match=message):
        validate_allocation(manifest(tmp_path), plan(), inspect_gpus(csv), disk_free_bytes=2000)


def test_rejects_archive_sm_mismatch(tmp_path):
    data = manifest(tmp_path)
    data["archive_sm"] = 90
    with pytest.raises(ValueError, match="archive sm90 does not match"):
        validate_allocation(data, plan(), inspect_gpus(GPU_CSV), disk_free_bytes=2000)


def test_rejects_insufficient_scratch(tmp_path):
    with pytest.raises(ValueError, match="scratch free bytes"):
        validate_allocation(manifest(tmp_path), plan(), inspect_gpus(GPU_CSV), disk_free_bytes=999)


def test_hash_verification_detects_corruption(tmp_path):
    data = manifest(tmp_path)
    verify_payload_hashes(tmp_path, data["payloads"])
    (tmp_path / "bin" / "production_runner").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_payload_hashes(tmp_path, data["payloads"])
