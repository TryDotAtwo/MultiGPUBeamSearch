"""CPU-only tests for querying the active CUDA context's physical UUID."""
from __future__ import annotations

import ctypes
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent
IDENTITY_PATH = ROOT.parent / "integrations" / "cayleypy_native" / "multigpubeamsearch" / "gpu_identity.py"
SPEC = importlib.util.spec_from_file_location("mgbs_gpu_identity_fixture", IDENTITY_PATH)
GPU_IDENTITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GPU_IDENTITY)


class FakeDevice:
    def __init__(self, index):
        self.type = "cuda"
        self.index = index

    def __str__(self):
        return "cuda" if self.index is None else f"cuda:{self.index}"


class FakeDriver:
    def __init__(self, uuids):
        self.uuids = uuids
        self.current = None

    def cuCtxGetDevice(self, output):
        ctypes.cast(output, ctypes.POINTER(ctypes.c_int))[0] = self.current
        return 0

    def cuDeviceGetUuid(self, output, device):
        raw = bytes.fromhex(self.uuids[device])
        ctypes.memmove(ctypes.cast(output, ctypes.c_void_p), raw, len(raw))
        return 0


class FakeCuda:
    def __init__(self, driver):
        self.driver = driver
        self.selected = None
        self.initialized = False

    def set_device(self, device):
        self.selected = device.index
        self.driver.current = device.index

    def init(self):
        self.initialized = True

    def current_device(self):
        return self.selected


class FakeTorch:
    def __init__(self, driver):
        self.cuda = FakeCuda(driver)
        self.empty_calls = []

    @staticmethod
    def device(value):
        if isinstance(value, FakeDevice):
            return value
        if value == "cuda":
            return FakeDevice(None)
        if isinstance(value, str) and value.startswith("cuda:"):
            return FakeDevice(int(value.split(":", 1)[1]))
        return type("CpuDevice", (), {"type": "cpu", "index": None, "__str__": lambda self: str(value)})()

    def empty(self, size, *, device):
        self.empty_calls.append((size, str(device)))
        return object()


class CnotGpuIdentityTests(unittest.TestCase):
    def test_reads_physical_uuids_for_reordered_visible_device_ordinals(self):
        # CUDA ordinal order can differ from nvidia-smi's physical inventory.
        uuids = {
            0: "fedcba98765432100123456789abcdef",
            1: "0123456789abcdef0123456789abcdef",
        }
        for ordinal, expected in (
            (0, "GPU-fedcba98-7654-3210-0123-456789abcdef"),
            (1, "GPU-01234567-89ab-cdef-0123-456789abcdef"),
        ):
            with self.subTest(ordinal=ordinal):
                driver = FakeDriver(uuids)
                torch = FakeTorch(driver)
                actual = GPU_IDENTITY.observe_current_cuda_uuid(
                    f"cuda:{ordinal}", torch_module=torch, driver=driver
                )
                self.assertEqual(actual, expected)
                self.assertEqual(driver.current, ordinal)
                self.assertTrue(torch.cuda.initialized)
                self.assertEqual(torch.empty_calls, [(0, f"cuda:{ordinal}")])

    def test_rejects_non_cuda_and_disagreement_with_selected_ordinal(self):
        driver = FakeDriver({0: "0123456789abcdef0123456789abcdef"})
        torch = FakeTorch(driver)
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            GPU_IDENTITY.observe_current_cuda_uuid("cpu", torch_module=torch, driver=driver)

        class WrongCurrentDevice(FakeCuda):
            def current_device(self):
                return self.selected + 1

        torch.cuda = WrongCurrentDevice(driver)
        with self.assertRaisesRegex(RuntimeError, "current process uses"):
            GPU_IDENTITY.observe_current_cuda_uuid("cuda:0", torch_module=torch, driver=driver)


if __name__ == "__main__":
    unittest.main(verbosity=2)
