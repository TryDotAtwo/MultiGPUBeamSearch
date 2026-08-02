"""Layered, user-writable cache for hardware-measured profiles."""
from __future__ import annotations
from pathlib import Path
import json, os, tempfile
from typing import Mapping

def load_registry(bundled: Path, cached: Path | None = None) -> dict[str, object]:
    base = json.loads(bundled.read_text(encoding="utf-8-sig"))
    if base.get("schema_version") != 1 or not isinstance(base.get("profiles"), list): raise ValueError("invalid bundled profile registry")
    profiles = list(base["profiles"])
    if cached is not None and cached.exists():
        local = json.loads(cached.read_text(encoding="utf-8-sig"))
        if local.get("schema_version") != 1 or not isinstance(local.get("profiles"), list): raise ValueError("invalid cached profile registry")
        def key(item):
            return (item.get("hardware"), item.get("backend"), item.get("model_class")) if isinstance(item, Mapping) else None
        local_keys = {json.dumps(key(item), sort_keys=True) for item in local["profiles"]}
        profiles = [item for item in profiles if json.dumps(key(item), sort_keys=True) not in local_keys]
        profiles.extend(local["profiles"])
    return {"schema_version": 1, "profiles": profiles}

def install_fragment(cache_path: Path, fragment: Mapping[str, object]) -> None:
    profiles = fragment.get("profiles")
    if fragment.get("schema_version") != 1 or not isinstance(profiles, list) or len(profiles) != 1: raise ValueError("autotune fragment must contain exactly one profile")
    candidate = profiles[0]
    if not isinstance(candidate, Mapping): raise ValueError("autotune profile must be an object")
    anchors = candidate.get("anchors")
    if not isinstance(anchors, Mapping) or not anchors: raise ValueError("autotune profile has no anchors")
    if any(not isinstance(a, Mapping) or a.get("status") != "measured" for a in anchors.values()): raise ValueError("only fully measured profiles may be installed")
    current = {"schema_version": 1, "profiles": []}
    if cache_path.exists(): current = json.loads(cache_path.read_text(encoding="utf-8-sig"))
    existing = current.get("profiles")
    if current.get("schema_version") != 1 or not isinstance(existing, list): raise ValueError("invalid cached profile registry")
    key = (candidate.get("hardware"), candidate.get("backend"), candidate.get("model_class"))
    kept = [p for p in existing if not isinstance(p, Mapping) or (p.get("hardware"), p.get("backend"), p.get("model_class")) != key]
    value = {"schema_version": 1, "profiles": [*kept, dict(candidate)]}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".registry.", suffix=".tmp", dir=cache_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, indent=2, sort_keys=True); handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, cache_path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)
