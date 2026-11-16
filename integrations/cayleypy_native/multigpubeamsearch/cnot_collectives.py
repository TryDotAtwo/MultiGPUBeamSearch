"""World-rank collectives for the packed CNOT frontier."""
from __future__ import annotations

import torch
import torch.distributed as dist


class World:
    """Use the caller's default process group and the matrix's local device."""

    def __init__(self, device):
        self.device = torch.device(device)
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.size = dist.get_world_size() if dist.is_initialized() else 1
        self.transport = None
        self._stream5_library = None
        if self.size > 65536:
            raise ValueError("CNOT physical routes support at most 65536 world ranks")

    def __enter__(self):
        if self.device.type == "cuda" and self.size > 1:
            from .cnot_stream5 import load_stream5

            if dist.get_backend() != "nccl" or torch.cuda.current_device() != self.device.index:
                raise ValueError("set the process's local CUDA device and initialize the NCCL world")
            self._stream5_library = self.checked("load native Stream 5", load_stream5)
        return self

    def start_transport(self, *, prefer_rust):
        """Choose one compatible transport for all ranks after engine admission."""
        if self._stream5_library is None:
            return
        from .cnot_stream5 import RustStream5, Stream5

        library = RustStream5.compatible_library(self._stream5_library) if prefer_rust else None
        if prefer_rust and self.sum(self.longs([int(library is not None)])).item() == self.size:
            self.transport = RustStream5(library, self)
        else:
            self.transport = Stream5(self._stream5_library, self)

    def __exit__(self, kind, value, traceback):
        if self.transport is not None:
            self.transport.close(abort=kind is not None)

    def longs(self, items):
        return torch.as_tensor(items, dtype=torch.int64, device=self.device)

    def sum(self, tensor):
        if self.size > 1:
            dist.all_reduce(tensor)
        return tensor

    def gather_fixed(self, tensor):
        if self.size == 1:
            return tensor.unsqueeze(0)
        parts = [torch.empty_like(tensor) for _ in range(self.size)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.stack(parts)

    def checked(self, stage, fn):
        """Coordinate local model/validation failures before peers enter the next exchange."""
        if self.size == 1:
            return fn()
        result, error = None, None
        try:
            result = fn()
        except Exception as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        if self.sum(self.longs([int(error is not None)])).item():
            errors = [None] * self.size
            dist.all_gather_object(errors, error)
            raise RuntimeError(f"{stage} failed: " + " | ".join(e for e in errors if e))
        return result

    def gather_rows(self, rows):
        """Replicate variable-size metadata with O(total rows) storage, including empty ranks."""
        if self.size == 1:
            return rows
        counts = self.gather_fixed(self.longs([len(rows)])).flatten().tolist()
        result = rows.new_empty((sum(counts), rows.size(1)))
        begin = 0
        for src, count in enumerate(counts):
            if count:
                part = result.narrow(0, begin, count)
                if src == self.rank:
                    part.copy_(rows)
                dist.broadcast(part, src)
            begin += count
        return result

    def exchange(self, rows, targets, *, native_packet=False):
        """Send variable-width int64 records to world ranks, including self and empty splits."""
        if self.size == 1:
            return rows
        # Older callers that enter World directly still get the original
        # transport even when they do not perform explicit engine admission.
        if self._stream5_library is not None and self.transport is None:
            self.start_transport(prefer_rust=False)
        ops = None
        if native_packet and rows.is_cuda:
            from . import cnot_native
            ops = cnot_native._cuda_ops()
        if ops is not None and hasattr(ops, "exchange_packet"):
            send, counts = ops.exchange_packet(rows, targets, self.size)
        else:
            counts = torch.bincount(targets, minlength=self.size)
            send = rows.index_select(0, torch.argsort(targets, stable=True)).contiguous()
        if self.transport is not None:
            return self.transport.exchange(send, counts)
        recv_counts = torch.empty_like(counts)
        dist.all_to_all_single(recv_counts, counts)
        send_split, recv_split = counts.tolist(), recv_counts.tolist()
        recv = rows.new_empty((sum(recv_split), rows.shape[1]))
        dist.all_to_all_single(recv, send, recv_split, send_split)
        return recv
