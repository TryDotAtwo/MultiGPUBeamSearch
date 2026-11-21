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
        vendor_root = Path(entry) / "nvidia"
        nvtx = vendor_root / "nvtx/include"
        if (nvtx / "nvtx3/nvToolsExt.h").is_file():
            includes.append(str(nvtx))
        vendor = vendor_root / "nccl"
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


class RustStream5:
    """Thin adapter for a Rust-owned communicator and whole-record exchange."""

    @staticmethod
    def compatible_library(lib):
        try:
            from .rust_control import extension
            cls = getattr(extension(), "CnotStream5Library", None)
            if cls is None:
                return None
            library = cls(str(Path(lib._name).resolve(strict=True)))
            return library if library.supports_exchange and library.supports_collectives else None
        except Exception:
            # Older prebuilt Stream 5 libraries and older PyO3 modules retain
            # the existing transport on every rank, selected collectively.
            return None

    def __init__(self, library, world):
        self.world = world
        source = getattr(world._stream5_library, "_name", None)
        self.library_sha256 = (
            hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else "unknown"
        )
        identity = [bytes(library.id_size)]

        def create_id():
            if world.rank == 0:
                identity[0] = library.unique_id()

        world.checked("create NCCL unique ID", create_id)
        payload = world.checked(
            "prepare NCCL unique ID broadcast payload",
            lambda: torch.tensor(list(identity[0]), dtype=torch.uint8, device=world.device),
        )
        world.checked(
            "broadcast NCCL unique ID",
            lambda: dist.broadcast(payload, 0),
        )
        identity_bytes = world.checked(
            "materialize NCCL unique ID",
            lambda: bytes(payload.tolist()),
        )
        self.comm = world.checked(
            "initialize Rust Stream 5 communicator",
            lambda: library.init(identity_bytes, world.rank, world.size),
        )

    def _trace_dispatch(self, operation, rows=None):
        if os.environ.get("MGBS_CNOT_STRICT_BACKEND") != "1":
            return
        suffix = "" if rows is None else f" rows={rows}"
        print(
            "CNOT_RUST_STREAM5_DISPATCH "
            f"rank={self.world.rank} world={self.world.size} pid={os.getpid()} device={self.world.device} "
            f"implementation=RustStream5/NCCL operation={operation} "
            f"abi=1 library_sha256={self.library_sha256}{suffix}",
            file=sys.stderr,
            flush=True,
        )

    def close(self, *, abort):
        self.comm.close(abort)

    def sum(self, tensor):
        self.comm.all_reduce_i64(tensor)
        self._trace_dispatch("all_reduce_i64", rows=tensor.numel())
        return tensor

    def gather_fixed(self, tensor):
        if not all(hasattr(self.comm, name) for name in
                   ("validate_all_gather_i64_into", "all_gather_i64_into")):
            result = self.comm.all_gather_i64(tensor)
            self._trace_dispatch("all_gather_i64", rows=tensor.numel())
            return result

        def prepare():
            work = tensor.contiguous()
            output_shape = (self.world.size, *tuple(tensor.shape))
            return work, tensor.new_empty(output_shape)

        work, output = self.world.checked(
            "prepare Rust Stream 5 fixed gather",
            prepare,
        )
        self.world.checked(
            "validate Rust Stream 5 fixed gather",
            lambda: self.comm.validate_all_gather_i64_into(tensor, work, output),
        )
        result = self.world.checked(
            "gather Rust Stream 5 fixed values",
            lambda: self.comm.all_gather_i64_into(tensor, work, output),
        )
        self._trace_dispatch("all_gather_i64_into", rows=tensor.numel())
        return result

    def broadcast(self, tensor, source):
        self.comm.broadcast_i64(tensor, source)
        self._trace_dispatch(f"broadcast_i64:source={source}", rows=tensor.numel())
        return tensor

    def exchange(self, send, counts):
        self.world.checked("validate Rust Stream 5 exchange",
                           lambda: self.comm.validate(send, counts, self.world))
        result = self.comm.exchange(send, counts, self.world)
        self._trace_dispatch("exchange", rows=len(send))
        return result


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
        if world.any_rank_failed(len(send) > limit):
            raise OverflowError("per-rank CNOT send exceeds Stream 5 uint32 offsets")
        send_counts = counts.to(torch.int32)
        recv_counts = torch.empty_like(send_counts)
        stream = torch.cuda.current_stream(world.device).cuda_stream
        args = (self.comm, stream, world.rank, world.size)
        self.check(self.lib.cnot_s5_counts(*args, send_counts.data_ptr(), recv_counts.data_ptr()))
        send_split = counts.tolist()
        recv_split = [count & limit for count in recv_counts.tolist()]
        size = sum(recv_split)
        if world.any_rank_failed(size > limit):
            raise OverflowError("per-rank CNOT receive exceeds Stream 5 uint32 offsets")
        # Reserve both device buffers before any rank enters the payload collective.
        backing = None
        source = None
        allocation_error = None
        try:
            backing = send.new_empty((max(1, size), send.size(1)))
            source = send if len(send) else send.new_empty((1, send.size(1)))
        except Exception as error:
            allocation_error = error
        if world.any_rank_failed(allocation_error is not None):
            if allocation_error is not None:
                raise allocation_error
            raise RuntimeError("CNOT Stream 5 receive buffer allocation failed on another rank")
        host_counts = C.c_uint32 * world.size
        self.check(self.lib.cnot_s5_rows(*args, source.data_ptr(), backing.data_ptr(),
                                         host_counts(*send_split), host_counts(*recv_split), send.size(1)))
        return backing[:size]
