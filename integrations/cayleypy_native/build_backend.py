"""PEP 517 wrapper that stages the matching native sources for maturin.

The checkout is never modified. Source distributions already contain the
verified bundle, so they can be rebuilt without the original checkout.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import maturin


_PROJECT = Path(__file__).resolve().parent
_BUNDLE = Path("multigpubeamsearch/_native_source")
_EXTENSIONS = {".py", ".cu", ".cuh", ".cpp", ".hpp", ".h", ".cmake",
               ".sh", ".json", ".txt", ".rs", ".toml"}
_FOLDERS = ("cuda", "src", "tools", "tests", "cmake", "configs", "schemas",
            "third_party", "profiles", "native", "rust")
_IGNORE = shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", "*.pyc", "target", "_native_source")


def _bundle_checkout(root: Path, destination: Path) -> None:
    if not (root / "tools/stream1_mlp_libtorch_backend.hpp").is_file():
        raise RuntimeError("build from the complete matching native checkout or its source distribution")
    files = [root / "CMakeLists.txt", root / "Cargo.toml", root / "Cargo.lock"]
    for folder in _FOLDERS:
        files.extend(path for path in (root / folder).rglob("*") if path.is_file()
                     and (path.suffix in _EXTENSIONS or path.name.startswith("LICENSE"))
                     and not any(part in (".git", "__pycache__", ".pytest_cache", "target") for part in path.parts))
    entries = [(path.relative_to(root), path) for path in sorted(set(files))]
    entries.append((Path("LICENSE"), _PROJECT / "LICENSE"))
    manifest = {}
    for relative, path in entries:
        raw = path.read_bytes().replace(b"\r\n", b"\n")
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
        manifest[relative.as_posix()] = hashlib.sha256(raw).hexdigest()
    (destination / "source_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")


def _verify_bundle(bundle: Path) -> None:
    manifest = json.loads((bundle / "source_manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not {"CMakeLists.txt", "cuda/cnot_domain.cu",
                                              "tools/production_runner.cu"} <= set(manifest):
        raise RuntimeError("source distribution has an incomplete native source manifest")
    for name, expected in manifest.items():
        path = bundle / name
        if (not path.resolve().is_relative_to(bundle.resolve()) or path.is_symlink()
                or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
            raise RuntimeError(f"source distribution contains modified native source: {name}")


@contextmanager
def _staged_project():
    # A source distribution has a workspace at its root and a bundled source
    # tree inside the Python package. A checkout has neither at this location.
    if (_PROJECT / _BUNDLE / "source_manifest.json").is_file() and (_PROJECT / "Cargo.toml").is_file():
        _verify_bundle(_PROJECT / _BUNDLE)
        yield _PROJECT
        return
    root = _PROJECT.parents[1]
    with tempfile.TemporaryDirectory(prefix="multigpubeamsearch-build-") as temporary:
        staged_root = Path(temporary) / "repository"
        staged_project = staged_root / "integrations/cayleypy_native"
        staged_root.mkdir()
        for name in ("Cargo.toml", "Cargo.lock"):
            shutil.copy2(root / name, staged_root / name)
        shutil.copytree(root / "rust", staged_root / "rust", ignore=_IGNORE)
        shutil.copytree(_PROJECT, staged_project, ignore=_IGNORE)
        _bundle_checkout(root, staged_project / _BUNDLE)
        yield staged_project


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


get_requires_for_build_wheel = maturin.get_requires_for_build_wheel
get_requires_for_build_sdist = maturin.get_requires_for_build_sdist
get_requires_for_build_editable = maturin.get_requires_for_build_editable


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    with _staged_project() as project, _working_directory(project):
        return maturin.prepare_metadata_for_build_wheel(metadata_directory, config_settings)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    with _staged_project() as project, _working_directory(project):
        return maturin.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    with _staged_project() as project, _working_directory(project):
        return maturin.build_sdist(sdist_directory, config_settings)


def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
    return maturin.prepare_metadata_for_build_editable(metadata_directory, config_settings)


def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
    # Editable installs must point to the original checkout, not a temporary copy.
    return maturin.build_editable(wheel_directory, config_settings, metadata_directory)
