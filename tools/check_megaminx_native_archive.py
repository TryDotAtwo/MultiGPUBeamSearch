"""Content and CUDA-image gates for native Megaminx archives."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile
from typing import Callable

import zstandard


def inspect_cuda_image_text(text: str, expected_sm: int) -> tuple[int, ...]:
    lowered = text.lower()
    if "ptx" in lowered or re.search(r"\bcompute_[0-9]+\b", lowered):
        raise ValueError("PTX/JIT image is forbidden")
    images = tuple(sorted({int(value) for value in re.findall(r"\bsm_([0-9]+)\b", lowered)}))
    if images != (expected_sm,):
        raise ValueError(f"expected only sm_{expected_sm}; found {images}")
    return images


def _safe_members(tar: tarfile.TarFile, prefix: str) -> list[tarfile.TarInfo]:
    members = tar.getmembers()
    for member in members:
        path = PurePosixPath(member.name)
        if member.issym() or member.islnk() or path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != prefix:
            raise ValueError(f"unsafe archive member: {member.name}")
        if not member.isfile():
            raise ValueError(f"non-file archive member: {member.name}")
    return members


def check_archive(archive: Path, expected_sm: int, cuobjdump: Callable[[Path], str]) -> dict[str, object]:
    expected_name = f"megaminx-sm{expected_sm}-linux-x86_64.tar.zst"
    if archive.name != expected_name:
        raise ValueError(f"archive name must be {expected_name}")
    raw = zstandard.ZstdDecompressor().decompress(archive.read_bytes(), max_output_size=8 * 1024 * 1024 * 1024)
    prefix = expected_name.removesuffix(".tar.zst")
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as tar:
        members = _safe_members(tar, prefix)
        by_name = {member.name: member for member in members}
        manifest_name = f"{prefix}/MANIFEST.json"
        checksum_name = f"{prefix}/SHA256SUMS"
        runner_name = f"{prefix}/bin/production_runner"
        if not all(name in by_name for name in (manifest_name, checksum_name, runner_name)):
            raise ValueError("archive lacks manifest, checksums, or runner")
        manifest = json.load(tar.extractfile(by_name[manifest_name]))
        if manifest.get("native_sm") != expected_sm or manifest.get("contains_ptx") is not False:
            raise ValueError("manifest native image declaration mismatch")
        runner = tar.extractfile(by_name[runner_name]).read()
    with tempfile.TemporaryDirectory() as temp:
        runner_path = Path(temp) / "production_runner"
        runner_path.write_bytes(runner)
        inspect_cuda_image_text(cuobjdump(runner_path), expected_sm)
    return {"archive": archive.name, "native_sm": expected_sm, "ptx": False, "members": len(members)}


def _cuobjdump(path: Path) -> str:
    result = subprocess.run(["cuobjdump", "--list-elf", "--list-ptx", str(path)], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError("cuobjdump failed")
    return result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--sm", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(check_archive(args.archive, args.sm, _cuobjdump), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
