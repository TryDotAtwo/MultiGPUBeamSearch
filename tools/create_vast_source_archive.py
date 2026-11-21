#!/usr/bin/env python3
"""Create the source-only archive used by the Vast acceptance workflow."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import tarfile


MANIFEST_PATH = (
    Path(__file__).resolve().parents[1]
    / "integrations/cayleypy_native/multigpubeamsearch/native_source_manifest.json"
)


def load_source_manifest(path: Path = MANIFEST_PATH) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read native source manifest: {path}") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported native source manifest schema")
    for key in ("archive_paths", "native_fingerprint_roots", "required_files"):
        values = manifest.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(item, str) for item in values):
            raise ValueError(f"native source manifest field {key} must be a non-empty string list")
        for item in values:
            relative = PurePosixPath(item)
            if relative.is_absolute() or ".." in relative.parts or "\\" in item:
                raise ValueError(f"unsafe path in native source manifest: {item}")
    manifest_path = manifest.get("manifest_path")
    if not isinstance(manifest_path, str):
        raise ValueError("native source manifest field manifest_path must be a string")
    relative_manifest = PurePosixPath(manifest_path)
    if (relative_manifest.is_absolute() or ".." in relative_manifest.parts
            or "\\" in manifest_path):
        raise ValueError(f"unsafe manifest path in native source manifest: {manifest_path}")
    if not any(manifest_path == root or manifest_path.startswith(root.rstrip("/") + "/")
               for root in manifest["archive_paths"]):
        raise ValueError("native source manifest path is outside the archive roots")
    return manifest


MANIFEST = load_source_manifest()
ARCHIVE_PATHS = tuple(Path(item) for item in MANIFEST["archive_paths"])
REQUIRED_SOURCE_FILES = tuple(Path(item) for item in MANIFEST["required_files"])
REQUIRED_VENDOR_HEADERS = tuple(
    path for path in REQUIRED_SOURCE_FILES if path.as_posix().startswith("third_party/")
)


def validate_source_tree(source_root: Path, manifest: dict = MANIFEST) -> None:
    source_root = source_root.resolve(strict=True)
    manifest = manifest or MANIFEST
    source_manifest_path = source_root / manifest["manifest_path"]
    if source_manifest_path.is_symlink() or not source_manifest_path.is_file():
        raise ValueError(f"source archive is missing {manifest['manifest_path']}")
    try:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"source archive has an invalid {manifest['manifest_path']}") from error
    if source_manifest != manifest:
        raise ValueError("source archive native source manifest does not match the packager")
    for relative in (Path(item) for item in manifest["archive_paths"]):
        path = source_root / relative
        if not path.exists() or path.is_symlink():
            raise ValueError(f"source archive input is missing or unsafe: {relative.as_posix()}")
    for relative in (Path(item) for item in manifest["required_files"]):
        path = source_root / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"required source file is missing: {relative.as_posix()}")


def _reject_links(member: tarfile.TarInfo) -> tarfile.TarInfo:
    if member.issym() or member.islnk():
        raise ValueError(f"source archive cannot contain links: {member.name}")
    return member


def create_source_archive(source_root: Path, archive_path: Path) -> Path:
    source_root = source_root.resolve(strict=True)
    validate_source_tree(source_root)
    archive_path = archive_path.resolve()
    if archive_path == source_root or source_root in archive_path.parents:
        raise ValueError("source archive output must be outside the source tree")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="w:gz") as archive:
        for relative in ARCHIVE_PATHS:
            archive.add(source_root / relative, arcname=relative.as_posix(), filter=_reject_links)
    return archive_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        output = create_source_archive(args.source_dir, args.output)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
