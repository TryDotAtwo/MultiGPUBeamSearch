"""Experimental per-session verification of a large NCCL library.

The conventional full-file SHA-256 is cached only after recording independent
SHA-256 digests of every fixed-size block. Reuse requires reading and hashing
every block again; file metadata is used to detect concurrent replacement or
modification during a read, never as proof that the bytes are unchanged.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import threading


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _require_unchanged(handle, path, expected):
    if _signature(os.fstat(handle.fileno())) != expected or _signature(path.stat()) != expected:
        raise OSError(f"NCCL library changed while reading: {path}")


def _digest_partition(path, expected, chunk_bytes, partition, partitions):
    size = expected[2]
    count = (size + chunk_bytes - 1) // chunk_bytes
    result = []
    with path.open("rb") as handle:
        _require_unchanged(handle, path, expected)
        for index in range(partition, count, partitions):
            offset = index * chunk_bytes
            length = min(chunk_bytes, size - offset)
            handle.seek(offset)
            data = handle.read(length)
            if len(data) != length:
                raise OSError(f"NCCL library changed while reading: {path}")
            result.append((index, hashlib.sha256(data).digest()))
        _require_unchanged(handle, path, expected)
    return result


@dataclass(frozen=True)
class _Entry:
    path: Path
    size: int
    whole_sha256: str
    chunks: tuple[bytes, ...]


class ChunkDigestCache:
    """Bounded, session-owned cache; it retains digests, never library bytes."""

    def __init__(self, *, chunk_bytes=1024 * 1024, workers=4):
        if type(chunk_bytes) is not int or chunk_bytes <= 0:
            raise ValueError("chunk_bytes must be a positive integer")
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("workers must be an integer between 1 and 64")
        self._chunk_bytes = chunk_bytes
        self._workers = workers
        self._entry = None
        self._lock = threading.Lock()

    def clear(self):
        with self._lock:
            self._entry = None

    def sha256(self, path):
        path = Path(path).resolve(strict=True)
        with self._lock, path.open("rb") as handle:
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise OSError(f"NCCL library is not a regular file: {path}")
            expected = _signature(info)
            _require_unchanged(handle, path, expected)
            entry = self._entry
            if entry is not None and entry.path == path and entry.size == info.st_size:
                count = (info.st_size + self._chunk_bytes - 1) // self._chunk_bytes
                chunks = [None] * count
                partitions = min(self._workers, count)
                if partitions:
                    try:
                        with ThreadPoolExecutor(max_workers=partitions) as executor:
                            results = executor.map(
                                _digest_partition,
                                [path] * partitions, [expected] * partitions,
                                [self._chunk_bytes] * partitions, range(partitions),
                                [partitions] * partitions,
                            )
                            for partition in results:
                                for index, digest in partition:
                                    chunks[index] = digest
                    except RuntimeError:
                        # If threads cannot be created, still verify every byte.
                        # A worker RuntimeError is retried here and propagates if
                        # it also occurs without the executor.
                        chunks = [digest for _, digest in
                                  _digest_partition(path, expected, self._chunk_bytes, 0, 1)]
                _require_unchanged(handle, path, expected)
                if tuple(chunks) == entry.chunks:
                    return entry.whole_sha256

            handle.seek(0)
            whole = hashlib.sha256()
            chunks = []
            remaining = info.st_size
            while remaining:
                length = min(self._chunk_bytes, remaining)
                data = handle.read(length)
                if len(data) != length:
                    raise OSError(f"NCCL library changed while reading: {path}")
                whole.update(data)
                chunks.append(hashlib.sha256(data).digest())
                remaining -= length
            _require_unchanged(handle, path, expected)
            self._entry = _Entry(path, info.st_size, whole.hexdigest(), tuple(chunks))
            return self._entry.whole_sha256
