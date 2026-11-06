"""NCCL variable-record transport shared with the native permutation runner."""
from __future__ import annotations

import ctypes as C
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist


def _build_stream5():
    from torch.utils.cpp_extension import load

    package = Path(__file__).resolve().parent
    root = package / "_native_source"
    if not (root / "cuda/stream5.cu").is_file():
        root = package.parents[2]
    sources = [root / "cuda/stream5.cu", root / "cuda/cnot_stream5_bridge.cu"]
    deps = ["cuda/stream5.hpp", "cuda/nvtx_ranges.hpp", "src/config.hpp", "src/types.hpp"]
    digest = hashlib.sha256()
    for path in [*sources, *(root / name for name in deps)]:
        digest.update(path.read_bytes())
    includes, libraries = [str(root / "src")], ["-lnccl"]
    for entry in sys.path:
        vendor = Path(entry) / "nvidia/nccl"
        if (vendor / "include/nccl.h").is_file():
            includes.append(str(vendor / "include"))
        files = sorted((vendor / "lib").glob("libnccl.so*"))
        if files:
            libraries = [str(files[0])]
            break
    return load(name="beam_cnot_stream5_" + digest.hexdigest()[:16],
                sources=[str(s) for s in sources], extra_include_paths=includes,
                extra_cuda_cflags=["-O2", "-DBEAM_STATE_LOGICAL_BYTES=120",
                                   "-DBEAM_STATE_PHYSICAL_BYTES=128", "-DBEAM_STATE_ALIGNMENT=16",
                                   "-DBEAM_MOVE_COUNT=1"], extra_ldflags=libraries, is_python_module=False)


@lru_cache(maxsize=1)
def load_stream5():
    # Prebuilt libraries let compute nodes share a tested binary without nvcc.
    path = os.environ.get("MGBS_CNOT_STREAM5_LIBRARY") or _build_stream5()
    lib = C.CDLL(str(Path(path).resolve(strict=True)))
    if lib.cnot_s5_abi() != 1:
        raise RuntimeError("incompatible native Stream 5 CNOT ABI")
    ptr, integer = C.c_void_p, C.c_int
    for name, args in {"id": [ptr], "init": [ptr, ptr, integer, integer], "close": [ptr, integer],
                       "counts": [ptr, ptr, integer, integer, ptr, ptr],
                       "rows": [ptr, ptr, integer, integer, ptr, ptr, ptr, ptr, C.c_uint32]}.items():
        getattr(lib, "cnot_s5_" + name).argtypes = args
    lib.cnot_s5_error.restype = C.c_char_p
    return lib


class Stream5:
    def __init__(self, lib, world):
        self.lib, self.world, self.comm = lib, world, C.c_void_p()
        length = lib.cnot_s5_id_bytes()
        buffer = C.create_string_buffer(length)
        def unique_id():
            if world.rank == 0:
                self.check(lib.cnot_s5_id(buffer))
        world.checked("create NCCL unique ID", unique_id)
        ident = torch.tensor(list(buffer.raw), dtype=torch.uint8, device=world.device)
        dist.broadcast(ident, 0)
        buffer.raw = bytes(ident.tolist())
        self.check(lib.cnot_s5_init(C.byref(self.comm), buffer, world.rank, world.size))

    def check(self, status):
        if status:
            raise RuntimeError(self.lib.cnot_s5_error().decode())

    def close(self, *, abort):
        self.check(self.lib.cnot_s5_close(self.comm, int(abort)))

    def exchange(self, send, counts):
        world = self.world
        limit = 0xffffffff
        # Offsets, including the final offset, are uint32 in the existing wire protocol.
        if world.sum(world.longs([int(len(send) > limit)])).item():
            raise OverflowError("per-rank CNOT send exceeds Stream 5 uint32 offsets")
        send_counts = counts.to(torch.int32)
        recv_counts = torch.empty_like(send_counts)
        stream = torch.cuda.current_stream(world.device).cuda_stream
        args = (self.comm, stream, world.rank, world.size)
        self.check(self.lib.cnot_s5_counts(*args, send_counts.data_ptr(), recv_counts.data_ptr()))
        send_split = counts.tolist()
        recv_split = [count & limit for count in recv_counts.tolist()]
        size = sum(recv_split)
        if world.sum(world.longs([int(size > limit)])).item():
            raise OverflowError("per-rank CNOT receive exceeds Stream 5 uint32 offsets")
        # A valid address also permits zero-count NCCL operations on an empty rank.
        backing = send.new_empty((max(1, size), send.size(1)))
        source = send if len(send) else send.new_empty((1, send.size(1)))
        host_counts = C.c_uint32 * world.size
        self.check(self.lib.cnot_s5_rows(*args, source.data_ptr(), backing.data_ptr(),
                                         host_counts(*send_split), host_counts(*recv_split), send.size(1)))
        return backing[:size]
