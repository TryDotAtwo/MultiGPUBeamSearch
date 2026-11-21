"""CPU-only tests for querying the active CUDA context's physical UUID."""
from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock


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
        self.current_context = 0xCAFE

    def cuCtxGetCurrent(self, output):
        if self.current_context is None:
            return 201
        ctypes.cast(output, ctypes.POINTER(ctypes.c_void_p))[0] = ctypes.c_void_p(self.current_context)
        return 0

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
        self.set_device_calls = 0
        self.init_calls = 0
        self.current_device_calls = 0

    def set_device(self, device):
        self.set_device_calls += 1
        self.selected = device.index
        self.driver.current = device.index
        self.driver.current_context = 0xCAFE + device.index

    def init(self):
        self.init_calls += 1
        self.initialized = True

    def current_device(self):
        self.current_device_calls += 1
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
                driver.current = ordinal
                driver.current_context = 0xB000 + ordinal
                torch = FakeTorch(driver)
                torch.cuda.selected = ordinal
                actual = GPU_IDENTITY.observe_current_cuda_uuid(
                    f"cuda:{ordinal}", torch_module=torch, driver=driver
                )
                self.assertEqual(actual, expected)
                self.assertEqual(driver.current, ordinal)
                self.assertEqual(torch.cuda.set_device_calls, 0)
                self.assertEqual(torch.cuda.init_calls, 0)
                self.assertEqual(torch.empty_calls, [])

    def test_rejects_non_cuda_and_disagreement_with_selected_ordinal(self):
        driver = FakeDriver({0: "0123456789abcdef0123456789abcdef"})
        torch = FakeTorch(driver)
        with self.assertRaisesRegex(RuntimeError, "requires a CUDA device"):
            GPU_IDENTITY.observe_current_cuda_uuid("cpu", torch_module=torch, driver=driver)

        class WrongCurrentDevice(FakeCuda):
            def current_device(self):
                return self.selected + 1

        torch.cuda = WrongCurrentDevice(driver)
        driver.current = 1
        with self.assertRaisesRegex(RuntimeError, "active context uses"):
            GPU_IDENTITY.observe_current_cuda_uuid("cuda:0", torch_module=torch, driver=driver)


    def test_observation_is_typed_and_reads_only_an_existing_context(self):
        expected_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        driver = FakeDriver({1: "0123456789abcdef0123456789abcdef"})
        driver.current = 1
        driver.current_context = 0xABCDEF
        torch = FakeTorch(driver)
        torch.cuda.selected = 1
        observation = GPU_IDENTITY.observe_current_cuda_context(
            "cuda:1", torch_module=torch, driver=driver
        )
        self.assertEqual(observation.device_ordinal, 1)
        self.assertEqual(observation.gpu_uuid, expected_uuid)
        self.assertEqual(observation.context_handle, 0xABCDEF)
        self.assertEqual(torch.cuda.set_device_calls, 0)
        self.assertEqual(torch.cuda.init_calls, 0)
        self.assertEqual(torch.cuda.current_device_calls, 0)
        self.assertEqual(torch.empty_calls, [])

    def test_rejects_missing_current_context_without_creating_one(self):
        driver = FakeDriver({0: "0123456789abcdef0123456789abcdef"})
        driver.current = 0
        driver.current_context = None
        torch = FakeTorch(driver)
        with self.assertRaisesRegex(RuntimeError, "no active CUDA context"):
            GPU_IDENTITY.observe_current_cuda_context(
                "cuda:0", torch_module=torch, driver=driver
            )
        self.assertEqual(torch.cuda.set_device_calls, 0)
        self.assertEqual(torch.cuda.init_calls, 0)
        self.assertEqual(torch.empty_calls, [])

    def test_rank_observation_uses_physical_uuid_order_and_rejects_duplicate_devices(self):
        sys.modules["gpu_identity"] = GPU_IDENTITY
        acceptance_path = IDENTITY_PATH.with_name("cnot_acceptance.py")
        spec = importlib.util.spec_from_file_location("mgbs_acceptance_fixture", acceptance_path)
        acceptance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(acceptance)
        uuids = [
            "GPU-fedcba98-7654-3210-0123-456789abcdef",
            "GPU-01234567-89ab-cdef-0123-456789abcdef",
        ]
        variables = {
            "MGBS_CNOT_ACCEPTANCE_RUN_ID": "fake-run",
            "MGBS_CNOT_ACCEPTANCE_CASE": "fake-case",
            "MGBS_CNOT_ACCEPTANCE_SOURCE_SHA": "a" * 40,
            "MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256": "b" * 64,
            "MGBS_CNOT_ACCEPTANCE_GPU_UUIDS": ",".join(uuids),
            "CNOT_RUST_EVIDENCE": "/tmp/mgbs-test",
            "LOCAL_RANK": "0",
        }
        observation = SimpleNamespace(device_ordinal=1, gpu_uuid=uuids[0], context_handle=0x111)
        with mock.patch.dict(os.environ, variables, clear=False), mock.patch.object(
            acceptance, "observe_current_cuda_context", return_value=observation
        ):
            pinned = acceptance.gpu_observation_for_rank(0, 2, device="cuda:1")
            self.assertEqual(pinned.gpu_uuid, uuids[0])
            self.assertEqual(pinned.device_ordinal, 1)
        variables["MGBS_CNOT_ACCEPTANCE_GPU_UUIDS"] = f"{uuids[0]},{uuids[0]}"
        with mock.patch.dict(os.environ, variables, clear=False), mock.patch.object(
            acceptance, "observe_current_cuda_context", return_value=observation
        ):
            with self.assertRaisesRegex(RuntimeError, "unique physical GPU UUIDs"):
                acceptance.gpu_observation_for_rank(0, 2, device="cuda:1")

    def test_rejects_context_replacement_after_acceptance_admission(self):
        sys.modules["gpu_identity"] = GPU_IDENTITY
        acceptance_path = IDENTITY_PATH.with_name("cnot_acceptance.py")
        spec = importlib.util.spec_from_file_location("mgbs_acceptance_fixture_changed", acceptance_path)
        acceptance = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(acceptance)
        expected_uuid = "GPU-01234567-89ab-cdef-0123-456789abcdef"
        variables = {
            "MGBS_CNOT_ACCEPTANCE_RUN_ID": "fake-run",
            "MGBS_CNOT_ACCEPTANCE_CASE": "fake-case",
            "MGBS_CNOT_ACCEPTANCE_SOURCE_SHA": "a" * 40,
            "MGBS_CNOT_ACCEPTANCE_WHEEL_SHA256": "b" * 64,
            "MGBS_CNOT_ACCEPTANCE_GPU_UUIDS": expected_uuid,
            "CNOT_RUST_EVIDENCE": "/tmp/mgbs-test",
            "LOCAL_RANK": "0",
        }
        observations = [
            SimpleNamespace(device_ordinal=0, gpu_uuid=expected_uuid, context_handle=0x111),
            SimpleNamespace(device_ordinal=0, gpu_uuid=expected_uuid, context_handle=0x222),
        ]
        with mock.patch.dict(os.environ, variables, clear=False), mock.patch.object(
            acceptance, "observe_current_cuda_context", side_effect=observations
        ):
            acceptance.gpu_observation_for_rank(0, 1, device="cuda:0")
            with self.assertRaisesRegex(RuntimeError, "CUDA context changed after CNOT admission"):
                acceptance.gpu_observation_for_rank(0, 1, device="cuda:0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
