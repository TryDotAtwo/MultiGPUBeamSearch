"""Read the physical CUDA UUID from an already active current-thread context."""
from __future__ import annotations
import ctypes
from dataclasses import dataclass
import re
import threading

GPU_UUID_RE = re.compile(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\\Z")

@dataclass(frozen=True)
class CudaDeviceObservation:
    device_ordinal: int
    gpu_uuid: str
    context_handle: int
    thread_id: int

def _cuda_driver():
    try:
        return ctypes.CDLL("libcuda.so.1")
    except OSError as error:
        raise RuntimeError(f"cannot load the CUDA driver library: {error}") from error

def observe_current_cuda_context(device=None, *, torch_module=None, driver=None):
    """Observe the current-thread CUDA context without binding or creating one."""
    if torch_module is None:
        import torch as torch_module
    selected = torch_module.device("cuda") if device is None else torch_module.device(device)
    if selected.type != "cuda":
        raise RuntimeError(f"physical GPU identity requires a CUDA device, got {selected}")
    driver = _cuda_driver() if driver is None else driver
    context = ctypes.c_void_p()
    status = driver.cuCtxGetCurrent(ctypes.byref(context))
    if status != 0 or not context.value:
        raise RuntimeError(f"no active CUDA context on the current worker thread (driver status {status})")
    current_device = ctypes.c_int()
    status = driver.cuCtxGetDevice(ctypes.byref(current_device))
    if status != 0:
        raise RuntimeError(f"cuCtxGetDevice failed with CUDA driver status {status}")
    ordinal = int(current_device.value)
    if selected.index is not None and ordinal != selected.index:
        raise RuntimeError(f"CUDA active context uses device {ordinal}, but the worker expects {selected.index}")
    uuid_bytes = (ctypes.c_ubyte * 16)()
    get_uuid = getattr(driver, "cuDeviceGetUuid_v2", None) or getattr(driver, "cuDeviceGetUuid", None)
    if get_uuid is None:
        raise RuntimeError("CUDA driver does not expose cuDeviceGetUuid")
    status = get_uuid(ctypes.byref(uuid_bytes), ordinal)
    if status != 0:
        raise RuntimeError(f"cuDeviceGetUuid failed with CUDA driver status {status}")
    compact = bytes(uuid_bytes).hex()
    result = f"GPU-{compact[:8]}-{compact[8:12]}-{compact[12:16]}-{compact[16:20]}-{compact[20:]}"
    if not GPU_UUID_RE.fullmatch(result):
        raise RuntimeError("CUDA driver returned a malformed physical GPU UUID")
    return CudaDeviceObservation(ordinal, result, int(context.value), threading.get_ident())

def observe_current_cuda_uuid(device=None, *, torch_module=None, driver=None):
    """Compatibility wrapper returning only the observed physical UUID."""
    return observe_current_cuda_context(device, torch_module=torch_module, driver=driver).gpu_uuid
