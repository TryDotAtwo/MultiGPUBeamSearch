"""Torch-owned CUDA buffers passed to the optional CNOT device-view ABI."""
import ctypes as c
import gc
import os
import unittest

import torch

import cnot_owned_domain_test as reference


class View(c.Structure):
    _fields_ = [("data", c.c_void_p), ("elements", c.c_uint64),
                ("device", c.c_int32), ("reserved", c.c_uint32)]


class BorrowedDeviceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("BEAM_CNOT_OWNED_LIBRARY")
        if not path:
            raise unittest.SkipTest("standalone CUDA domain library was not supplied")
        cls.lib = c.CDLL(path)
        cls.lib.beam_cnot_domain_device_abi_version.restype = c.c_uint32
        cls.lib.beam_cnot_domain_error.restype = c.c_char_p
        cls.lib.beam_cnot_domain_create.argtypes = [c.POINTER(reference.Config), c.POINTER(c.c_void_p)]
        cls.lib.beam_cnot_domain_destroy.argtypes = [c.c_void_p]
        cls.lib.beam_cnot_domain_owners_device.argtypes = [c.c_void_p, c.POINTER(View), c.c_uint64,
                                                           c.POINTER(View), c.c_void_p]

    @classmethod
    def tearDownClass(cls):
        # Sanitizer stack collection can retain completed test frames. Tests
        # explicitly release tensors before returning, then flush Torch's
        # allocator so leak-check covers native and caller-owned allocations.
        gc.collect()
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def ok(self, code):
        self.assertEqual(code, 0, self.lib.beam_cnot_domain_error().decode())

    def fails(self, code, *parts):
        self.assertNotEqual(code, 0)
        message = self.lib.beam_cnot_domain_error().decode()
        for part in parts:
            self.assertIn(part, message)

    def context(self, n, device=0, world=7, capacity=4):
        config = reference.Config(1, c.sizeof(reference.Config), device, n, world, 0, 0, capacity, 0)
        ctx = c.c_void_p()
        self.ok(self.lib.beam_cnot_domain_create(c.byref(config), c.byref(ctx)))
        self.addCleanup(lambda: self.ok(self.lib.beam_cnot_domain_destroy(ctx)))
        return ctx

    def call(self, ctx, states, count, output, stream):
        return self.lib.beam_cnot_domain_owners_device(ctx, c.byref(states), count,
                                                        c.byref(output), c.c_void_p(stream))

    def run_device(self, n, device):
        stride = n * ((n + 63) // 64)
        count = 3
        ctx = self.context(n, device)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            torch.cuda._sleep(2_000_000)
            states = torch.arange(count * stride, dtype=torch.int64, device=device).reshape(count, n, -1)
            states = states * 65537 + 9
            owners = torch.empty(count, dtype=torch.int64, device=device)
            a = View(states.data_ptr(), states.numel(), device, 0)
            b = View(owners.data_ptr(), owners.numel(), device, 0)
            self.ok(self.call(ctx, a, count, b, stream.cuda_stream))
            self.assertTrue(stream.query(), "borrowed stream must be complete on return")
            # A later whole-batch Torch consumer sees the same unchanged input
            # and completed output on this non-default stream.
            consumed = owners * 3 + states[:, 0, 0]
        # The Torch consumer is asynchronous; the subsequent CPU reads occur
        # outside this stream's scope and must wait for the consumer as well.
        stream.synchronize()
        expected_states = [[(i * stride + j) * 65537 + 9 for j in range(stride)] for i in range(count)]
        expected_owners = [reference.owner(state, n, 7) for state in expected_states]
        self.assertEqual(owners.cpu().tolist(), expected_owners)
        self.assertEqual(consumed.cpu().tolist(), [expected_owners[i] * 3 + row[0]
                                                   for i, row in enumerate(expected_states)])
        del states, owners, consumed

    def test_nondefault_stream_and_torch_buffers(self):
        self.assertEqual(self.lib.beam_cnot_domain_device_abi_version(), 1)
        for n in (4, 64, 144):
            with self.subTest(n=n):
                self.run_device(n, 0)
        if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
            self.run_device(4, 1)

    def test_default_stream(self):
        ctx = self.context(4, capacity=2)
        with torch.cuda.device(0), torch.cuda.stream(torch.cuda.default_stream(0)):
            states = torch.arange(8, dtype=torch.int64, device='cuda:0')
            owners = torch.empty(2, dtype=torch.int64, device='cuda:0')
            self.ok(self.call(ctx, View(states.data_ptr(), 8, 0, 0), 2,
                              View(owners.data_ptr(), 2, 0, 0), 0))
            expected = [reference.owner(list(range(4)), 4, 7),
                        reference.owner(list(range(4, 8)), 4, 7)]
            self.assertEqual(owners.cpu().tolist(), expected)
            del states, owners

    def test_rejects_invalid_views_without_poisoning_context(self):
        ctx = self.context(4, capacity=2)
        stream = torch.cuda.Stream(device=0)
        states = torch.arange(8, dtype=torch.int64, device='cuda:0')
        output = torch.empty(2, dtype=torch.int64, device='cuda:0')
        a = View(states.data_ptr(), states.numel(), 0, 0)
        b = View(output.data_ptr(), output.numel(), 0, 0)
        # The default-stream producer must complete before use on another
        # stream; production callers must establish this dependency themselves.
        stream.wait_stream(torch.cuda.current_stream(0))
        self.fails(self.call(ctx, View(a.data, 7, 0, 0), 2, b, stream.cuda_stream), "required=8", "available=7")
        self.fails(self.call(ctx, a, 2, View(b.data, 1, 0, 0), stream.cuda_stream), "required=2", "available=1")
        self.fails(self.call(ctx, a, 3, b, stream.cuda_stream), "required=3", "available=2")
        self.fails(self.call(ctx, a, 2, View(a.data, 2, 0, 0), stream.cuda_stream), "overlap")
        self.fails(self.call(ctx, View(a.data, 8, 0, 1), 2, b, stream.cuda_stream), "reserved")
        self.fails(self.call(ctx, View(a.data + 1, 8, 0, 0), 2, b, stream.cuda_stream), "align")
        host = (c.c_int64 * 8)()
        self.fails(self.call(ctx, View(c.addressof(host), 8, 0, 0), 2, b, stream.cuda_stream), "device")
        empty = View(None, 0, 0, 0)
        self.ok(self.call(ctx, empty, 0, empty, stream.cuda_stream))
        if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
            other = torch.empty(8, dtype=torch.int64, device='cuda:1')
            self.fails(self.call(ctx, View(other.data_ptr(), 8, 1, 0), 2, b, stream.cuda_stream), "device")
            self.fails(self.call(ctx, View(other.data_ptr(), 8, 0, 0), 2, b, stream.cuda_stream), "device")
            other_stream = torch.cuda.Stream(device=1)
            self.fails(self.call(ctx, a, 2, b, other_stream.cuda_stream), "stream")
        self.ok(self.call(ctx, a, 2, b, stream.cuda_stream))
        expected = [reference.owner(list(range(4)), 4, 7), reference.owner(list(range(4, 8)), 4, 7)]
        self.assertEqual(output.cpu().tolist(), expected)
        if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
            del other
        del states, output

    def test_previous_device_restored(self):
        if os.environ.get("BEAM_TEST_SECOND_GPU") != "1":
            self.skipTest("second GPU was not requested")
        ctx = self.context(4, device=0)
        stream = torch.cuda.Stream(device=0)
        states = torch.arange(4, dtype=torch.int64, device='cuda:0')
        owners = torch.empty(1, dtype=torch.int64, device='cuda:0')
        stream.wait_stream(torch.cuda.current_stream(0))
        with torch.cuda.device(1):
            self.ok(self.call(ctx, View(states.data_ptr(), 4, 0, 0), 1,
                              View(owners.data_ptr(), 1, 0, 0), stream.cuda_stream))
            self.assertEqual(torch.cuda.current_device(), 1)
        del states, owners


if __name__ == '__main__':
    unittest.main(verbosity=2)
