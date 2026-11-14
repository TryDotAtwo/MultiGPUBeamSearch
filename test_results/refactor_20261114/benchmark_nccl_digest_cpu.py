"""Small Windows CPU diagnostic, not a GPU or end-to-end benchmark."""
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import tempfile
import time

from multigpubeamsearch.build import file_sha256
from multigpubeamsearch.nccl_digest import ChunkDigestCache


def timed(fn, path, expected):
    start = time.perf_counter()
    actual = fn(path)
    seconds = time.perf_counter() - start
    if actual != expected:
        raise AssertionError("digest mismatch")
    return seconds


def main():
    directory = Path(__file__).resolve().parents[2] / ".local" / "tmp"
    directory.mkdir(parents=True, exist_ok=True)
    directory = directory.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="nccl-digest-diagnostic-", dir=directory) as temporary:
        temporary_path = Path(temporary).resolve(strict=True)
        if temporary_path.parent != directory:
            raise RuntimeError("temporary path is outside the expected directory")
        path = temporary_path / "synthetic-library"
        block = bytes(range(256)) * 4096
        with path.open("wb") as handle:
            for _ in range(16):
                handle.write(block)
        expected = hashlib.sha256(block * 16).hexdigest()
        cache = ChunkDigestCache()
        cold_baseline = timed(file_sha256, path, expected)
        cold_cached = timed(cache.sha256, path, expected)
        pairs = []
        for index in range(5):
            sample = {}
            order = [("baseline", file_sha256), ("cached", cache.sha256)]
            if index % 2:
                order.reverse()
            for name, fn in order:
                sample[name] = timed(fn, path, expected)
            pairs.append(sample)
        baseline = statistics.median(item["baseline"] for item in pairs)
        cached = statistics.median(item["cached"] for item in pairs)
        print(json.dumps({
            "purpose": "small warmed CPU diagnostic; not an end-to-end or GPU result",
            "os": platform.platform(), "logical_cpus": os.cpu_count(),
            "bytes": path.stat().st_size, "sha256": expected,
            "cold_baseline_seconds": cold_baseline, "cold_cached_seconds": cold_cached,
            "pairs": pairs, "median_baseline_seconds": baseline,
            "median_cached_seconds": cached, "median_change_percent": 100 * (cached / baseline - 1),
        }, indent=2))


if __name__ == "__main__":
    main()
