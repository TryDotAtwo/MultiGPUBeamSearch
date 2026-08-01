import json
from pathlib import Path
import tarfile

import pytest
import zstandard

from tools.build_megaminx_native_release import ALLOWED_SMS, build_release
from tools.check_megaminx_native_archive import check_archive, inspect_cuda_image_text


def stage(tmp_path: Path) -> Path:
    root = tmp_path / "stage"
    for directory in ("bin", "lib", "data", "weights", "profiles", "scripts", "portable/megaminx_cluster", "portable/megaminx_cluster/autotune"):
        (root / directory).mkdir(parents=True)
    (root / "bin/production_runner").write_bytes(b"ELF fake sm image")
    (root / "lib/libtorch.so").write_bytes(b"runtime")
    (root / "data/test.csv").write_text("initial_state_id,initial_state\n900,1,0\n")
    (root / "data/puzzle_info.json").write_text("{}")
    (root / "weights/megaminx.pt").write_bytes(b"weights")
    (root / "profiles/registry.json").write_text('{"schema_version":1,"profiles":[]}')
    (root / "scripts/job.sh").write_text("#!/usr/bin/env bash\n")
    (root / "scripts/preflight.sh").write_text("#!/usr/bin/env bash\n")
    (root / "scripts/autotune_job.sh").write_text("#!/usr/bin/env bash\n")
    (root / "README.md").write_text("Megaminx release\n")
    (root / "run.sh").write_text("#!/usr/bin/env bash\n")
    (root / "autotune.sh").write_text("#!/usr/bin/env bash\n")
    (root / "portable/megaminx_cluster/autotune/calibration.json").write_text("{}")
    return root


def cuobjdump_for(sm: int):
    return lambda _: f"Fatbin elf code:\narch = sm_{sm}\ncode version = [1,7]\n"


def test_exact_native_sm_allowlist():
    assert ALLOWED_SMS == (75, 80, 86, 89, 90, 120)


@pytest.mark.parametrize("sm", [74, 76, 87, 121])
def test_builder_rejects_unknown_sm(tmp_path, sm):
    with pytest.raises(ValueError, match="unsupported SM"):
        build_release(stage(tmp_path), tmp_path / "out", sm, cuobjdump_for(sm))


def test_cuda_gate_accepts_exactly_one_sm_without_ptx():
    assert inspect_cuda_image_text("arch = sm_90\n", 90) == (90,)


def test_cuda_gate_accepts_empty_ptx_listing_header():
    listing = "Fatbin elf code:\narch = sm_80\nFatbin ptx code:\n"
    assert inspect_cuda_image_text(listing, 80) == (80,)


@pytest.mark.parametrize("text", ["arch = sm_80\narch = sm_90", "arch = sm_90\nptxas", "arch = compute_90", "arch = sm_90\nPTX file 1: kernel.ptx", "arch = sm_80"])
def test_cuda_gate_rejects_multiple_ptx_or_wrong_sm(text):
    with pytest.raises(ValueError):
        inspect_cuda_image_text(text, 90)


def test_builds_named_deterministic_archive_and_checks_it(tmp_path):
    root = stage(tmp_path)
    first = build_release(root, tmp_path / "a", 90, cuobjdump_for(90))
    second = build_release(root, tmp_path / "b", 90, cuobjdump_for(90))
    assert first.name == "megaminx-sm90-linux-x86_64.tar.zst"
    assert first.read_bytes() == second.read_bytes()
    report = check_archive(first, 90, cuobjdump_for(90))
    assert report["native_sm"] == 90 and report["ptx"] is False


@pytest.mark.parametrize("name", ["token.txt", ".env", "compile.sh", "source.cu", "Dockerfile"])
def test_builder_rejects_forbidden_payload_names(tmp_path, name):
    root = stage(tmp_path)
    (root / name).write_text("secret")
    with pytest.raises(ValueError, match="forbidden"):
        build_release(root, tmp_path / "out", 90, cuobjdump_for(90))


def test_builder_rejects_symlink_escape(tmp_path):
    root = stage(tmp_path)
    outside = tmp_path / "outside"
    outside.write_text("private")
    try:
        (root / "data/escape").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        build_release(root, tmp_path / "out", 90, cuobjdump_for(90))


def test_manifest_and_checksums_are_inside_archive(tmp_path):
    archive = build_release(stage(tmp_path), tmp_path / "out", 120, cuobjdump_for(120))
    raw = zstandard.ZstdDecompressor().decompress(archive.read_bytes())
    tar_path = tmp_path / "payload.tar"
    tar_path.write_bytes(raw)
    with tarfile.open(tar_path, "r:") as tar:
        names = tar.getnames()
        manifest = json.load(tar.extractfile("megaminx-sm120-linux-x86_64/MANIFEST.json"))
    assert "megaminx-sm120-linux-x86_64/SHA256SUMS" in names
    assert manifest["native_sm"] == 120 and manifest["contains_ptx"] is False
