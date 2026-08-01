import pytest

from portable.megaminx_cluster.scripts.preflight import inspect_gpus, validate_allocation


def test_sm89_archive_allows_l4_and_rtx40_families_but_not_other_sm89_names():
    manifest = {"archive_sm": 89, "gpu_families": ["L4", "RTX_4090"], "minimum_driver_major": 550}
    plan = {"world_size": 1, "required_vram_mib": 22000, "history_disk_bytes": 1}
    l4 = inspect_gpus("0, NVIDIA L4, 23034 MiB, 8.9, 570.1\n")
    rtx = inspect_gpus("0, NVIDIA RTX 4090, 24564 MiB, 8.9, 570.1\n")
    assert validate_allocation(manifest, plan, l4, 2)["status"] == "ok"
    assert validate_allocation(manifest, plan, rtx, 2)["status"] == "ok"
    bad = dict(manifest, gpu_families=["L4"])
    with pytest.raises(ValueError, match="GPU families"):
        validate_allocation(bad, plan, rtx, 2)
