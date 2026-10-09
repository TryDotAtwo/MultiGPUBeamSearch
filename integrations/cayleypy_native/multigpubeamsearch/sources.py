"""Explicit, checksummed public-source setup. Never called implicitly by search."""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request

from .errors import NativeBackendError, NativeUnavailable
from .options import NativeOptions


@dataclass(frozen=True)
class SourceArchive:
    name: str
    repository: str
    revision: str
    sha256: str
    root: str

    @property
    def url(self):
        return f"https://codeload.github.com/{self.repository}/tar.gz/{self.revision}"


# Immutable native/CUTLASS snapshots used in real two-T4 acceptance. Updating a
# pin requires a new archive checksum and native integration validation.
NATIVE_SOURCE = SourceArchive(
    "native", "TryDotAtwo/MultiGPUBeamSearch", "a1db0e6d9bb5458c8a842b37dfa99572d3025667",
    "53a68e2261e799aa5421f925c2a70bd23b40dae262092a3a34fea6026bafd6ad", "MultiGPUBeamSearch-a1db0e6d9bb5458c8a842b37dfa99572d3025667",
)
CUTLASS_SOURCE = SourceArchive(
    "cutlass", "NVIDIA/cutlass", "afa1772203677c5118fcd82537a9c8fefbcc7008",
    "e62fb320c2b61e7e0c7a1163c9c5a58f6dd86025adf7df4d124933152482997f", "cutlass-afa1772203677c5118fcd82537a9c8fefbcc7008",
)
_MAX_DOWNLOAD = 256 * 1024**2
_MAX_EXTRACTED = 1024**3
_MAX_MEMBERS = 100000


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(spec, destination):
    request = urllib.request.Request(spec.url, headers={"User-Agent": "cayleypy-native-source-setup"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, destination.open("xb") as output:
            total = 0
            while block := response.read(1024**2):
                total += len(block)
                if total > _MAX_DOWNLOAD:
                    raise NativeBackendError("source archive exceeds download size limit")
                output.write(block)
    except (OSError, urllib.error.URLError) as exc:
        raise NativeUnavailable(f"cannot download {spec.name} sources; check HTTPS/proxy access or use explicit "
                                f"source_dir/cutlass_dir for offline setup: {exc}") from exc
    if _sha256(destination) != spec.sha256:
        raise NativeBackendError(f"{spec.name} archive SHA256 mismatch; refusing to extract")


def _members(archive, spec):
    members, total = [], 0
    names = set()
    for member in archive:
        parts = PurePosixPath(member.name).parts
        if (not parts or parts[0] != spec.root or ".." in parts or "\\" in member.name
                or ":" in member.name or not (member.isdir() or member.isfile())):
            raise NativeBackendError(f"unsafe source archive member: {member.name}")
        if len(parts) == 1:
            if not member.isdir():
                raise NativeBackendError("source archive root must be a directory")
            continue
        relative = PurePosixPath(*parts[1:]).as_posix()
        if relative in names:
            raise NativeBackendError(f"duplicate source archive member: {relative}")
        names.add(relative)
        total += member.size
        if total > _MAX_EXTRACTED or len(names) > _MAX_MEMBERS:
            raise NativeBackendError("source archive exceeds extraction limits")
        members.append((member, relative))
    if not members:
        raise NativeBackendError("empty source archive")
    return members


def _extract(archive_path, tree, spec):
    # No extractall, links, traversal or archive-supplied ownership/modes.
    with tarfile.open(archive_path, "r:gz") as archive:
        members = _members(archive, spec)
        tree.mkdir()
        for member, relative in members:
            target = tree / relative
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)


def _verify_cached(directory, spec):
    archive_path, tree = directory / "source.tar.gz", directory / "tree"
    if (directory.is_symlink() or archive_path.is_symlink() or tree.is_symlink()
            or not archive_path.is_file() or not tree.is_dir() or _sha256(archive_path) != spec.sha256):
        raise NativeBackendError(f"incomplete or modified {spec.name} source cache: {directory}")
    expected = set()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member, relative in _members(archive, spec):
            target = tree / relative
            if any(parent.is_symlink() for parent in [target, *target.parents] if parent != directory.parent):
                raise NativeBackendError(f"source cache contains a symlink: {target}")
            if member.isdir():
                if not target.is_dir():
                    raise NativeBackendError(f"source cache directory missing: {target}")
            else:
                expected.add(relative)
                with archive.extractfile(member) as original:
                    digest = hashlib.sha256()
                    for block in iter(lambda: original.read(1024**2), b""):
                        digest.update(block)
                if not target.is_file() or _sha256(target) != digest.hexdigest():
                    raise NativeBackendError(f"source cache modified: {target}")
    actual = set()
    for path in tree.rglob("*"):
        if path.is_symlink():
            raise NativeBackendError(f"source cache contains a symlink: {path}")
        if path.is_file():
            actual.add(path.relative_to(tree).as_posix())
    if actual != expected:
        raise NativeBackendError(f"source cache contains unexpected files: {tree}")
    return tree


def _setup_one(cache, spec, *, offline):
    directory = cache / "sources" / f"{spec.name}-{spec.revision}"
    if directory.exists() or directory.is_symlink():
        return _verify_cached(directory, spec)
    if offline:
        raise NativeUnavailable(f"{spec.name} sources are not cached; run setup_sources() once online "
                                "or provide source_dir/cutlass_dir")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{spec.name}-", dir=directory.parent))
    try:
        _download(spec, temporary / "source.tar.gz")
        _extract(temporary / "source.tar.gz", temporary / "tree", spec)
        try:
            os.rename(temporary, directory)
        except OSError:
            # Another explicit setup may have completed first. Never overwrite
            # an existing cache; validate it before accepting it.
            if not directory.exists():
                raise
        return _verify_cached(directory, spec)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def setup_sources(*, options=None, cache_dir=None, source_dir=None, cutlass_dir=None, offline=False):
    """Return NativeOptions using explicit directories or verified pinned source caches.

    This explicit call may download public native and CUTLASS source archives.
    It does not install packages, compile, probe CUDA, fetch weights or search.
    ``offline=True`` never performs network access. Explicit directories are
    caller-managed and checked for required files, not against the default pin.
    """
    if options is not None and not isinstance(options, NativeOptions):
        raise TypeError("options must be NativeOptions")
    selected = options or NativeOptions()
    updates = {}
    for name, value in (("cache_dir", cache_dir), ("source_dir", source_dir), ("cutlass_dir", cutlass_dir)):
        if value is not None:
            updates[name] = value
    selected = replace(selected, **updates)
    if selected.runner_path is not None:
        raise ValueError("setup_sources prepares source builds; runner_path is already a separate runtime choice")
    # Validate both explicit paths before any possible download.
    required = (("source_dir", ("CMakeLists.txt", "tools/production_runner.cu", "tools/export_stream1_mlp.py")),
                ("cutlass_dir", ("include/cutlass/cutlass.h",)))
    for name, files in required:
        directory = getattr(selected, name)
        if directory is not None and not all((directory / filename).is_file() for filename in files):
            raise NativeUnavailable(f"{name} is missing required source files: {directory}")
    native = selected.source_dir or _setup_one(selected.cache_dir, NATIVE_SOURCE, offline=offline)
    cutlass = selected.cutlass_dir or _setup_one(selected.cache_dir, CUTLASS_SOURCE, offline=offline)
    return replace(selected, source_dir=native, cutlass_dir=cutlass)
