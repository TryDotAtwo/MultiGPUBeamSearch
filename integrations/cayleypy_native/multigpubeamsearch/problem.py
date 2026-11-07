"""Small, puzzle-agnostic helpers for Cayley graph inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
import tempfile
import urllib.request
import zipfile


_MAX_BUNDLE_BYTES = 2 * 1024**3
_MAX_BUNDLE_FILES = 100_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_entries(root: Path) -> dict[str, dict]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise ValueError("input bundle is missing manifest.json")
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries, dict) or not entries:
        raise ValueError("input bundle manifest must be a nonempty object")
    checked = {}
    for name, item in entries.items():
        relative = PurePosixPath(name)
        if (not isinstance(name, str) or not name or relative.is_absolute()
                or ".." in relative.parts or "\\" in name or ":" in name
                or name == "manifest.json"):
            raise ValueError(f"unsafe input bundle manifest path: {name!r}")
        if (not isinstance(item, dict) or type(item.get("bytes")) is not int
                or item["bytes"] < 0 or not isinstance(item.get("sha256"), str)
                or len(item["sha256"]) != 64):
            raise ValueError(f"invalid input bundle manifest entry: {name}")
        path = root.joinpath(*relative.parts)
        if not path.resolve().is_relative_to(root.resolve()) or path.is_symlink():
            raise ValueError(f"unsafe input bundle path: {name}")
        if not path.is_file() or path.stat().st_size != item["bytes"] or _sha256(path) != item["sha256"]:
            raise ValueError(f"input bundle verification failed: {name}")
        checked[name] = item
    discovered = list(root.rglob("*"))
    if any(path.is_symlink() for path in discovered):
        raise ValueError("input bundle contains a symbolic link")
    files = {path.relative_to(root).as_posix() for path in discovered if path.is_file()}
    if files != set(checked) | {"manifest.json"}:
        raise ValueError("input bundle contains missing or unexpected files")
    return checked


def _extract_bundle(archive_path: Path, destination: Path) -> None:
    staging = Path(tempfile.mkdtemp(prefix=".inputs-", dir=destination.parent))
    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if len(members) > _MAX_BUNDLE_FILES:
                raise ValueError("input bundle contains too many entries")
            total = 0
            names = set()
            for member in members:
                relative = PurePosixPath(member.filename)
                mode = member.external_attr >> 16
                file_type = stat.S_IFMT(mode)
                if (not relative.parts or relative.is_absolute() or ".." in relative.parts
                        or "\\" in member.filename or ":" in member.filename
                        or (file_type not in (0, stat.S_IFREG, stat.S_IFDIR))):
                    raise ValueError(f"unsafe input bundle entry: {member.filename}")
                if member.filename in names:
                    raise ValueError(f"duplicate input bundle entry: {member.filename}")
                names.add(member.filename)
                total += member.file_size
                if total > _MAX_BUNDLE_BYTES:
                    raise ValueError("input bundle exceeds the extraction limit")
            for member in members:
                target = staging.joinpath(*PurePosixPath(member.filename).parts)
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
        _manifest_entries(staging)
        os.rename(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def download_bundle(url: str, cache_dir=None, *, archive_sha256: str | None = None) -> Path:
    """Download and verify a ZIP containing manifest.json plus SHA256 entries."""
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a nonempty string")
    cache = Path(cache_dir or Path.home() / ".cache" / "multigpubeamsearch").expanduser().resolve()
    bundle_dir = cache / "bundles" / hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]
    bundle_dir.mkdir(parents=True, exist_ok=True)
    archive_path = bundle_dir / "inputs.zip"
    destination = bundle_dir / "inputs"
    if destination.exists():
        _manifest_entries(destination)
        if archive_sha256 and archive_path.is_file() and _sha256(archive_path) != archive_sha256:
            raise ValueError("cached input bundle archive SHA256 mismatch")
        return destination
    if not archive_path.is_file():
        with tempfile.NamedTemporaryFile(prefix=".inputs-", suffix=".zip", dir=bundle_dir,
                                         delete=False) as output:
            temporary = Path(output.name)
        try:
            total = 0
            with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while block := response.read(1024 * 1024):
                    total += len(block)
                    if total > _MAX_BUNDLE_BYTES:
                        raise ValueError("input bundle exceeds the download limit")
                    output.write(block)
            if archive_sha256 and _sha256(temporary) != archive_sha256:
                raise ValueError("input bundle archive SHA256 mismatch")
            os.replace(temporary, archive_path)
        finally:
            temporary.unlink(missing_ok=True)
    elif archive_sha256 and _sha256(archive_path) != archive_sha256:
        raise ValueError("cached input bundle archive SHA256 mismatch")
    _extract_bundle(archive_path, destination)
    return destination


def load_graph(graph_info, *, device="cpu", random_seed=None):
    """Create a CayleyPy graph from its ordered ``generators`` and center."""
    from cayleypy import CayleyGraph, CayleyGraphDef

    if isinstance(graph_info, (str, Path)):
        graph_info = json.loads(Path(graph_info).read_text(encoding="utf-8"))
    if not isinstance(graph_info, dict):
        raise ValueError("graph_info must be a mapping or a JSON file path")
    generators = graph_info.get("generators")
    center = graph_info.get("central_state")
    if isinstance(generators, dict):
        names, permutations = list(generators), list(generators.values())
    elif isinstance(generators, list):
        permutations = generators
        names = graph_info.get("generator_names", [f"m{i}" for i in range(len(generators))])
    else:
        raise ValueError("graph_info.generators must be a mapping or a list")
    if not isinstance(center, (list, tuple)) or not permutations:
        raise ValueError("graph_info must contain generators and central_state")
    if not isinstance(names, (list, tuple)) or len(names) != len(permutations):
        raise ValueError("generator_names must match the ordered generators")
    definition = CayleyGraphDef.create(
        permutations, generator_names=names, central_state=center
    )
    return CayleyGraph(definition, device=device, random_seed=random_seed)


def read_states(path, *, id_column="id", state_column="state", separator=",") -> dict[str, list[int]]:
    """Read ID→state rows from a CSV; state values may be delimited or JSON."""
    output = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if id_column not in (reader.fieldnames or ()) or state_column not in (reader.fieldnames or ()):
            raise ValueError(f"CSV must contain {id_column!r} and {state_column!r} columns")
        for row in reader:
            key = row[id_column].strip()
            if not key:
                raise ValueError("state CSV contains an empty ID")
            if key in output:
                raise ValueError(f"duplicate state ID: {key}")
            raw = row[state_column].strip()
            values = json.loads(raw) if raw.startswith("[") else raw.split(separator)
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(f"state for {key} must be a nonempty integer vector")
            try:
                state = [int(value) for value in values]
            except (TypeError, ValueError):
                raise ValueError(f"state for {key} contains a non-integer") from None
            output[key] = state
    if not output:
        raise ValueError("state CSV contains no rows")
    return output


def read_paths(path, graph, *, id_column="id", path_column="solution", separator=".") -> dict[str, tuple[int, ...]]:
    """Read named move sequences from CSV and resolve them in this graph."""
    from .results import resolve_move_path

    output = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if id_column not in (reader.fieldnames or ()) or path_column not in (reader.fieldnames or ()):
            raise ValueError(f"CSV must contain {id_column!r} and {path_column!r} columns")
        for row in reader:
            key = row[id_column].strip()
            if not key:
                raise ValueError("solution CSV contains an empty ID")
            if key in output:
                raise ValueError(f"duplicate solution ID: {key}")
            raw = row[path_column].strip()
            moves = json.loads(raw) if raw.startswith("[") else ([x for x in raw.split(separator) if x] if raw else [])
            output[key] = resolve_move_path(graph, moves)
    return output


def select_ids(start=1020, end=None, *, indices_url=None, random_choice=False) -> list[str]:
    """Select generic CSV IDs from an inclusive numeric range or an ID list."""
    if indices_url:
        with urllib.request.urlopen(indices_url, timeout=30) as response:
            text = response.read(1_000_001)
        if len(text) > 1_000_000:
            raise ValueError("ID list is too large")
        ids = [value.strip() for row in csv.reader(text.decode("utf-8-sig").splitlines())
               for value in row if value.strip()]
    else:
        if type(start) is not int or (end is not None and type(end) is not int) or start < 0:
            raise ValueError("start and end must be nonnegative integer IDs")
        last = start if end is None else end
        if last < start:
            raise ValueError("end must be greater than or equal to start")
        ids = [str(value) for value in range(start, last + 1)]
    if not ids:
        raise ValueError("expected at least one ID")
    if random_choice:
        return [random.choice(ids)]
    return ids
