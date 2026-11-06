"""CUDA scorer backed by the restored 102-input TheQuantumProject field."""

import ctypes as C
import json
from pathlib import Path
import threading
import weakref


def _load_api(path):
    lib = C.CDLL(str(Path(path).resolve(strict=True)))
    signatures = {
        "mgbs_tqp_abi_version": ([], C.c_uint32),
        "mgbs_tqp_error": ([], C.c_char_p),
        "mgbs_tqp_open": ([C.c_char_p, C.c_uint32, C.c_uint32], C.c_void_p),
        "mgbs_tqp_metadata": ([C.c_void_p], C.c_char_p),
        "mgbs_tqp_score": ([C.c_void_p, C.c_uint32, C.c_int64, C.c_void_p, C.c_void_p, C.c_void_p], C.c_int),
        "mgbs_tqp_close": ([C.c_void_p], None),
    }
    for name, (args, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes = args
        function.restype = result
    if lib.mgbs_tqp_abi_version() != 1:
        raise RuntimeError("unsupported CNOT scorer ABI")
    return lib


class TQPScorer:
    """One restored field on one rank. Inputs are valid packed GL(n,2) states."""

    def __init__(self, library, manifest, *, n, device, inference_batch=4):
        import torch

        if not isinstance(n, int) or not 2 <= n <= 0xFFFFFFFF:
            raise ValueError("n must be a valid CNOT dimension")
        if not isinstance(inference_batch, int) or inference_batch < 1:
            raise ValueError("inference_batch must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda" or self.device.index is None:
            raise ValueError("select an explicit rank-local CUDA device such as cuda:0")
        if torch.__version__.split("+")[0] != "2.9.0" or not torch.version.cuda:
            raise RuntimeError("the pinned tch 0.22.0 binding requires CUDA PyTorch 2.9.0")
        self.n = n
        self._lock = threading.Lock()
        self._api = _load_api(library)
        manifest = str(Path(manifest).resolve(strict=True)).encode("utf-8")
        with torch.cuda.device(self.device):
            self._handle = self._api.mgbs_tqp_open(manifest, n, self.device.index)
        if not self._handle:
            raise self._native_error()
        self._release = weakref.finalize(self, self._api.mgbs_tqp_close, self._handle)
        try:
            data = self._api.mgbs_tqp_metadata(self._handle)
            if not data:
                raise self._native_error()
            self.metadata = json.loads(data)
            if self.metadata["feature_model_inputs"] != 102 or len(self.metadata["feature_ids"]) != 102:
                raise RuntimeError("expected all 102 registered field inputs")
            self._batch = min(inference_batch, self.metadata["max_packet_rows"])
            if self._batch < 1:
                raise RuntimeError("the feature policy cannot admit a single row")
        except BaseException:
            self.close()
            raise

    def _native_error(self):
        error = self._api.mgbs_tqp_error()
        return RuntimeError(error.decode("utf-8", "replace") if error else "CNOT scorer failed")

    def close(self):
        with self._lock:
            self._release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def parent_scores(self, rows, mode):
        """Return raw Qhat costs or raw proposal logits in control-major move order."""
        if mode not in ("qhat", "proposal"):
            raise ValueError("parent scorer mode must be qhat or proposal")
        return self._score(rows, 0 if mode == "qhat" else 1)

    def child_values(self, rows):
        """Return raw scalar V per child. Lower values rank first."""
        return self._score(rows, 2)

    def _score(self, rows, head):
        import torch

        if (not isinstance(rows, torch.Tensor) or rows.dtype != torch.int64
                or rows.device != self.device or rows.ndim != 3
                or tuple(rows.shape[1:]) != (self.n, (self.n + 63) // 64)):
            raise ValueError("rows must be rank-local CUDA int64 [batch, n, ceil(n/64)]")
        with self._lock, torch.cuda.device(self.device), torch.inference_mode():
            if not self._release.alive:
                raise RuntimeError("CNOT scorer is closed")
            count = rows.shape[0]
            shape = (count,) if head == 2 else (count, self.n * (self.n - 1))
            output = torch.empty(shape, dtype=torch.float32, device=self.device)
            if not count:
                return output
            rows = rows.contiguous()
            status = torch.empty(count, dtype=torch.int64, device=self.device)
            for start in range(0, count, self._batch):
                size = min(self._batch, count - start)
                code = self._api.mgbs_tqp_score(
                    self._handle, head, size, rows[start:].data_ptr(),
                    output[start:].data_ptr(), status[start:].data_ptr(),
                )
                if code:
                    raise self._native_error()
            finite = output.isfinite().reshape(count, -1).all(dim=1)
            failed = status.ne(0) | ~finite
            # Only admission flags cross to the host. The engine receives the
            # original CUDA scores and cannot select a row with failed coverage.
            if failed.any().item():
                row = int(failed.int().argmax().item())
                coverage = int(status[row].item())
                if coverage:
                    slot, value = divmod(coverage, 256)
                    feature = self.metadata["feature_ids"][slot - 1]
                    raise RuntimeError(f"CNOT row {row}: feature {feature} rejected (status {value})")
                raise RuntimeError(f"CNOT row {row}: nonfinite model score")
            return output
