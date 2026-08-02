"""Build deterministic compiler-free Megaminx native release archives."""

from __future__ import annotations

import argparse
from hashlib import sha256
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
from typing import Callable, Mapping

import zstandard

from tools.megaminx_archive_contract import require_allowed_path

ALLOWED_SMS = (75, 80, 86, 89, 90, 120)
REQUIRED_PATHS = ("bin/production_runner", "lib", "data/test.csv", "data/puzzle_info.json", "weights", "profiles/registry.json", "scripts/job.sh", "scripts/preflight.sh", "scripts/autotune_job.sh", "run.sh", "autotune.sh", "portable/megaminx_cluster/autotune/calibration.json", "portable/megaminx_cluster/torchrun.py", "README.md")
FORBIDDEN_NAMES = frozenset({".env", "Dockerfile", "token.txt", "compile.sh"})
FORBIDDEN_SUFFIXES = frozenset({".cu", ".cuh", ".cpp", ".cc", ".o", ".a", ".ptx"})
SECRET_PATTERN = re.compile(rb"(?:ghp_[A-Za-z0-9]{20,}|BEGIN (?:RSA|OPENSSH) PRIVATE KEY|CLOUDFLARE_API_TOKEN)")
PRIVATE_PATH_PATTERN = re.compile(rb"(?:[A-Za-z]:\\\\(?:Users|Documents|Downloads)\\\\|/(?:home|Users|mnt)/)[^\x00\r\n\"']+", re.IGNORECASE)


def _files(root: Path) -> list[Path]:
    result = []
    resolved_root = root.resolve()
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"symlink payload is forbidden: {path.relative_to(root)}")
        if not path.is_file():
            continue
        if resolved_root not in path.resolve().parents:
            raise ValueError(f"payload escapes staging root: {path}")
        relative = path.relative_to(root)
        if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"forbidden payload file: {relative.as_posix()}")
        require_allowed_path(relative.as_posix())
        data = path.read_bytes()
        if SECRET_PATTERN.search(data):
            raise ValueError(f"forbidden secret-like content: {relative.as_posix()}")
        if PRIVATE_PATH_PATTERN.search(data):
            raise ValueError(f"private absolute path in payload: {relative.as_posix()}")
        result.append(path)
    return result


def _verify_required(root: Path) -> None:
    for relative in REQUIRED_PATHS:
        if not (root / relative).exists():
            raise ValueError(f"missing release asset: {relative}")


def _tarinfo(name: str, size: int, executable: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size, info.mtime, info.uid, info.gid = size, 0, 0, 0
    info.uname = info.gname = ""
    info.mode = 0o755 if executable else 0o644
    return info


def _metadata(root: Path, supplied: Mapping[str, object] | None) -> dict[str, object]:
    if supplied is not None:
        return dict(supplied)
    path = root / "RELEASE.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8-sig"))
    return {"backend": "mlp", "model_class": "output_move_count", "move_count": 12, "output_dim": 12, "gpu_family": "UNVERIFIED", "minimum_vram_mib": 1, "minimum_driver_major": 1, "history_disk_bytes": 1}


def build_release(stage_root: Path, output_dir: Path, sm: int, cuobjdump: Callable[[Path], str], metadata: Mapping[str, object] | None = None) -> Path:
    if sm not in ALLOWED_SMS:
        raise ValueError(f"unsupported SM: {sm}")
    root = stage_root.resolve()
    _verify_required(root)
    from tools.check_megaminx_native_archive import inspect_cuda_image_text
    inspect_cuda_image_text(cuobjdump(root / "bin/production_runner"), sm)
    files = _files(root)
    prefix = f"megaminx-sm{sm}-linux-x86_64"
    payloads = {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in files}
    manifest = {"schema_version": 1, "archive_sm": sm, "native_sm": sm, "contains_ptx": False, "payloads": payloads, **_metadata(root, metadata)}
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    payloads["MANIFEST.json"] = sha256(manifest_bytes).hexdigest()
    checksum_bytes = "".join(f"{digest}  {name}\n" for name, digest in sorted(payloads.items())).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in files:
            relative, data = path.relative_to(root).as_posix(), path.read_bytes()
            tar.addfile(_tarinfo(f"{prefix}/{relative}", len(data), relative in {"run.sh", "autotune.sh", "scripts/job.sh", "scripts/autotune_job.sh", "bin/production_runner"}), io.BytesIO(data))
        tar.addfile(_tarinfo(f"{prefix}/MANIFEST.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
        tar.addfile(_tarinfo(f"{prefix}/SHA256SUMS", len(checksum_bytes)), io.BytesIO(checksum_bytes))
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{prefix}.tar.zst"
    target.write_bytes(zstandard.ZstdCompressor(level=19, threads=0, write_checksum=True).compress(stream.getvalue()))
    return target


def _cuobjdump(path: Path) -> str:
    result = subprocess.run(["cuobjdump", "--list-elf", "--list-ptx", str(path)], text=True, capture_output=True, check=False)
    if result.returncode != 0:
        raise ValueError(f"cuobjdump failed for {path.name}")
    return result.stdout + result.stderr


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sm", type=int, choices=ALLOWED_SMS, required=True); parser.add_argument("--metadata", type=Path)
    args = parser.parse_args()
    metadata = None if args.metadata is None else json.loads(args.metadata.read_text(encoding="utf-8-sig"))
    print(build_release(args.stage, args.output, args.sm, _cuobjdump, metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
