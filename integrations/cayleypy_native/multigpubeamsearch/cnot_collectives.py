"""World-rank collectives for the packed CNOT frontier."""
from __future__ import annotations

import torch
import torch.distributed as dist


_UNSET = object()


class World:
    """Use the caller's world ranks and local device for CNOT collectives."""

    def __init__(self, device):
        self._device_arg = device
        self.device = None
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.size = dist.get_world_size() if dist.is_initialized() else 1
        self.transport = None
        self._stream5_library = None
        self._control_group = None
        self._owns_control_group = False
        self._native_extension = _UNSET
        self._native_support_cache = {}
        self._checked_error_gather_v2 = None
        self._checked_error_gather_v1 = None
        if self.size > 65536:
            raise ValueError("CNOT physical routes support at most 65536 world ranks")

    def __enter__(self):
        if self.size == 1:
            self.device = torch.device(self._device_arg)
            return self

        # Every rank creates the same CPU control group before checking local
        # backend/device state or inspecting optional PyO3 capabilities.
        self._control_group = dist.new_group(
            ranks=list(range(self.size)), backend="gloo"
        )
        self._owns_control_group = True
        local_error = None
        try:
            self.device = torch.device(self._device_arg)
            backend = str(dist.get_backend()).lower()
            if self.device.type == "cuda":
                if "nccl" not in backend or torch.cuda.current_device() != self.device.index:
                    raise ValueError(
                        "set the process's local CUDA device and initialize the NCCL world"
                    )
            elif self.device.type == "cpu" and "nccl" in backend and "gloo" not in backend:
                raise ValueError("CPU CNOT tensors require a CPU-capable distributed backend")

            from .errors import NativeUnavailable
            from .rust_control import extension

            try:
                self._native_extension = extension()
            except (ImportError, NativeUnavailable):
                self._native_extension = None
        except BaseException as exc:
            local_error = f"rank {self.rank}: {type(exc).__name__}: {exc}"

        errors = [None] * self.size
        dist.all_gather_object(errors, local_error, group=self._control_group)
        if any(errors):
            details = " | ".join(error for error in errors if error)
            self._destroy_control_group()
            raise RuntimeError(f"CNOT distributed preflight failed: {details}")

        if self.device.type == "cuda":
            def load_stream5():
                from .cnot_stream5 import load_stream5

                return load_stream5()

            try:
                self._stream5_library = self.checked("load native Stream 5", load_stream5)
            except BaseException:
                self._destroy_control_group()
                raise
        return self

    def start_transport(self, *, prefer_rust):
        """Choose one exchange/control transport for all ranks after admission."""
        used, result = self._rust_collective(
            "run_cnot_native_start_transport", prefer_rust
        )
        if used:
            return result
        if self._stream5_library is None:
            return
        from .cnot_stream5 import RustStream5, Stream5

        library = RustStream5.compatible_library(self._stream5_library) if prefer_rust else None
        if prefer_rust and self.sum(self.longs([int(library is not None)])).item() == self.size:
            self.transport = RustStream5(library, self)
        else:
            self.transport = Stream5(self._stream5_library, self)

    def all_ranks_support(self, local_support):
        """Vote over the stable CPU control group, independent of optional Rust ABIs."""
        vote = torch.tensor([int(bool(local_support))], dtype=torch.int64)
        gathered = self._control_gather(vote)
        return gathered.sum().item() == self.size

    def require_consensus(self, digest):
        """Reject CNOT setup when the ranks received different search inputs."""
        used, result = self._rust_collective(
            "run_cnot_native_require_consensus", digest
        )
        if used:
            return result
        identities = self.gather_fixed(self.longs(list(digest)))
        if not (identities == identities[0]).all().item():
            raise ValueError("CNOT root, search settings or scorer fingerprints differ between ranks")

    def require_engine_consensus(self, digest):
        """Compare requested CNOT engine settings on the CPU Gloo control group."""
        if self.size == 1:
            return
        identities = self._control_gather(
            torch.tensor(list(digest), dtype=torch.int64)
        )
        if not (identities == identities[0]).all().item():
            raise ValueError("CNOT search engine settings differ between ranks")

    def __exit__(self, kind, value, traceback):
        try:
            if self.transport is not None:
                self.transport.close(abort=kind is not None)
        finally:
            self._destroy_control_group()

    def _destroy_control_group(self):
        group = self._control_group
        if not self._owns_control_group or group is None:
            return
        self._control_group = None
        self._owns_control_group = False
        if dist.is_initialized():
            dist.destroy_process_group(group)

    def longs(self, items):
        return torch.as_tensor(items, dtype=torch.int64, device=self.device)

    def _control_gather(self, tensor):
        if self.size == 1:
            return tensor.unsqueeze(0)
        if self._control_group is None:
            raise RuntimeError("multi-rank CNOT collectives require an entered World")
        local = tensor.detach().to(device="cpu").contiguous()
        parts = [torch.empty_like(local) for _ in range(self.size)]
        dist.all_gather(parts, local, group=self._control_group)
        return torch.stack(parts)

    def sum(self, tensor):
        if self.size > 1:
            used, result = self._rust_collective("run_cnot_native_sum", tensor)
            if used:
                return result
            if self.transport is not None and hasattr(self.transport, "sum"):
                self.transport.sum(tensor)
            else:
                dist.all_reduce(tensor)
        return tensor

    def gather_fixed(self, tensor):
        if self.size == 1:
            return tensor.unsqueeze(0)
        used, result = self._rust_collective("run_cnot_native_gather_fixed", tensor)
        if used:
            return result
        if self.transport is not None and hasattr(self.transport, "gather_fixed"):
            return self.transport.gather_fixed(tensor)
        parts = [torch.empty_like(tensor) for _ in range(self.size)]
        dist.all_gather(parts, tensor.contiguous())
        return torch.stack(parts)

    def broadcast(self, tensor, source):
        if self.size > 1:
            used, result = self._rust_collective("run_cnot_native_broadcast", tensor, source)
            if used:
                return result
            if self.transport is not None and hasattr(self.transport, "broadcast"):
                self.transport.broadcast(tensor, source)
            else:
                dist.broadcast(tensor, source)
        return tensor

    def checked(self, stage, fn):
        """Coordinate local failures and negotiate the same checked-stage ABI."""
        if self.size == 1:
            return fn()

        native = self._extension()
        checker_v2 = getattr(native, "run_cnot_native_checked_v2", None)
        legacy_checker = getattr(native, "run_cnot_native_checked", None)

        if self._checked_error_gather_v2 is None:
            supported = self.supports_native("run_cnot_native_checked_v2")
            self._checked_error_gather_v2 = checker_v2 if supported else False
        if callable(self._checked_error_gather_v2):
            return self._checked_error_gather_v2(self, stage, fn)

        if self._checked_error_gather_v1 is None:
            supported = self.supports_native("run_cnot_native_checked")
            self._checked_error_gather_v1 = legacy_checker if supported else False
        if callable(self._checked_error_gather_v1):
            return self._checked_error_gather_v1(self, stage, fn)

        # Compatibility path for older PyO3 modules without either checked ABI.
        result, error = None, None
        try:
            result = fn()
        except BaseException as exc:
            error = f"rank {self.rank}: {type(exc).__name__}: {exc}"
        if self.sum(self.longs([int(error is not None)])).item():
            errors = [None] * self.size
            dist.all_gather_object(errors, error, group=self._control_group)
            raise RuntimeError(f"{stage} failed: " + " | ".join(e for e in errors if e))
        return result

    def _extension(self):
        if self._native_extension is _UNSET:
            from .errors import NativeUnavailable
            from .rust_control import extension

            try:
                self._native_extension = extension()
            except (ImportError, NativeUnavailable):
                self._native_extension = None
        return self._native_extension

    def native_method(self, name):
        native = self._extension()
        return getattr(native, name, None) if native is not None else None

    def supports_native(self, name):
        if name not in self._native_support_cache:
            operation = self.native_method(name)
            self._native_support_cache[name] = (
                self.all_ranks_support(callable(operation))
                if self.size > 1
                else callable(operation)
            )
        return self._native_support_cache[name]

    def _rust_collective(self, name, *args):
        operation = self.native_method(name)
        if self.supports_native(name) and callable(operation):
            return True, operation(self, *args)
        return False, None

    def gather_rows(self, rows):
        """Replicate variable-size metadata with O(total rows) storage, including empty ranks."""
        if self.size == 1:
            return rows
        used, result = self._rust_collective(
            "run_cnot_native_gather_rows", rows
        )
        if used:
            return result

        # Compatibility path for older PyO3 modules without the gather ABI.
        counts = self.gather_fixed(self.longs([len(rows)])).flatten().tolist()
        result = rows.new_empty((sum(counts), rows.size(1)))
        begin = 0
        for src, count in enumerate(counts):
            if count:
                part = result.narrow(0, begin, count)
                if src == self.rank:
                    part.copy_(rows)
                self.broadcast(part, src)
            begin += count
        return result

    def exchange(self, rows, targets, *, native_packet=False):
        """Send variable-width int64 records to world ranks, including self and empty splits."""
        if self.size == 1:
            return rows
        used, result = self._rust_collective(
            "run_cnot_native_world_exchange", rows, targets, native_packet
        )
        if used:
            return result
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
