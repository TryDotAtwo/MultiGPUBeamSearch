"""Read the physical CUDA UUID selected by the current process context."""
from __future__ import annotations

import ctypes
import re


GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\Z")


def _cuda_driver():
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError(f"cannot load the CUDA driver library: {error}") from error


def observe_current_cuda_uuid(device, *, torch_module=None, driver=None):
    """Return the UUID of the CUDA device bound to this thread's current context.

    ``driver`` is injectable so hosted CPU tests can exercise device selection
    and UUID mapping without a CUDA installation.
    """
    if torch_module is None:
        import torch as torch_module

    selected = torch_module.device(device)
    if selected.type != "cuda":
        raise RuntimeError(f"physical GPU identity requires a CUDA device, got {selected}")
    torch_module.cuda.set_device(selected)
    torch_module.cuda.init()
    torch_module.empty(0, device=selected)
    current_ordinal = int(torch_module.cuda.current_device())
    if selected.index is not None and current_ordinal != selected.index:
        raise RuntimeError(
            f"CUDA selected device {selected.index}, but the current process uses {current_ordinal}"
        )

    driver = _cuda_driver() if driver is None else driver
    current_device = ctypes.c_int()
    status = driver.cuCtxGetDevice(ctypes.byref(current_device))
    if status != 0:
        raise RuntimeError(f"cuCtxGetDevice failed with CUDA driver status {status}")
    uuid_bytes = (ctypes.c_ubyte * 16)()
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or getattr(driver, "cuDeviceGetUuid", None)
    if get_uuid is None:
        raise RuntimeError("CUDA driver does not expose cuDeviceGetUuid")
    status = get_uuid(ctypes.byref(uuid_bytes), current_device.value)
    if status != 0:
        raise RuntimeError(f"cuDeviceGetUuid failed with CUDA driver status {status}")
    compact = bytes(uuid_bytes).hex()
    result = f"GPU-{compact[:8]}-{compact[8:12]}-{compact[12:16]}-{compact[16:20]}-{compact[20:]}"
    if not GPU_UUID_RE.fullmatch(result):
        raise RuntimeError("CUDA driver returned a malformed physical GPU UUID")
    return result
