"""Borrowed-device CNOT expansion with caller-owned Torch allocations."""
import ctypes as c
import gc
import os
import threading
import unittest

import torch

import cnot_owned_domain_test as reference
from cnot_borrowed_device_test import View


class BorrowedExpansionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("BEAM_CNOT_OWNED_LIBRARY")
        if not path:
            raise unittest.SkipTest("standalone CUDA domain library was not supplied")
        cls.lib = c.CDLL(path)
        cls.lib.beam_cnot_domain_error.restype = c.c_char_p
        cls.lib.beam_cnot_domain_create_device.argtypes = [c.POINTER(reference.Config), c.POINTER(c.c_void_p)]
        cls.lib.beam_cnot_domain_create.argtypes = [c.POINTER(reference.Config), c.POINTER(c.c_void_p)]
        cls.lib.beam_cnot_domain_destroy.argtypes = [c.c_void_p]
        cls.lib.beam_cnot_domain_set_parents.argtypes = [c.c_void_p, reference.I64P, c.c_uint64, c.c_uint64]
        cls.lib.beam_cnot_domain_set_descriptors.argtypes = [
            c.c_void_p, reference.I64P, c.c_uint64, reference.I64P, c.c_uint64,
        ]
        cls.lib.beam_cnot_domain_expand.argtypes = [c.c_void_p]
        cls.lib.beam_cnot_domain_download_children.argtypes = [c.c_void_p, reference.I64P, c.c_uint64]
        cls.lib.beam_cnot_domain_children_device.argtypes = [
            c.c_void_p, c.POINTER(View), c.c_uint64, c.POINTER(View), c.POINTER(View), c.c_uint32,
            c.c_uint64, c.POINTER(View), c.c_void_p,
        ]
        cls.lib.beam_cnot_domain_owners_device.argtypes = [
            c.c_void_p, c.POINTER(View), c.c_uint64, c.POINTER(View), c.c_void_p,
        ]

    @classmethod
    def tearDownClass(cls):
        # Sanitizer may retain completed Python frames and their local tensors.
        gc.collect()
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def ok(self, code):
        self.assertEqual(code, 0, self.lib.beam_cnot_domain_error().decode())

    def fails(self, code, *parts):
        self.assertNotEqual(code, 0)
        error = self.lib.beam_cnot_domain_error().decode()
        for part in parts:
            self.assertIn(part, error)

    def context(self, n, device=0, parents=3, children=4, device_only=True):
        cfg = reference.Config(1, c.sizeof(reference.Config), device, n, 7, 0,
                               parents, children, 0)
        ctx = c.c_void_p()
        create = self.lib.beam_cnot_domain_create_device if device_only else self.lib.beam_cnot_domain_create
        self.ok(create(c.byref(cfg), c.byref(ctx)))
        self.addCleanup(lambda: self.ok(self.lib.beam_cnot_domain_destroy(ctx)))
        return ctx

    def call(self, ctx, parents, parent_count, indices, moves, move_type, count, children, stream):
        return self.lib.beam_cnot_domain_children_device(
            ctx, c.byref(parents), parent_count, c.byref(indices), c.byref(moves), move_type,
            count, c.byref(children), c.c_void_p(stream))

    def exercise(self, n, device, dtype, use_default=False):
        width = (n + 63) // 64
        stride = n * width
        ctx = self.context(n, device=device)
        stream = torch.cuda.default_stream(device) if use_default else torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            # Nonzero high padding tests the same canonicalization as the old kernel.
            parents = (torch.arange(3 * stride, dtype=torch.int64, device=device) * 65537 - 7).reshape(3, n, width)
            indices = torch.tensor([2, 0, 1, 2], dtype=torch.int64, device=device)
            move_values = [0, n - 1, n * (n - 1) - 1, 1]
            moves = torch.tensor(move_values, dtype=dtype, device=device)
            children = torch.empty((4, n, width), dtype=torch.int64, device=device)
            owners = torch.empty(4, dtype=torch.int64, device=device)
            a = View(parents.data_ptr(), parents.numel(), device, 0)
            b = View(indices.data_ptr(), indices.numel(), device, 0)
            m = View(moves.data_ptr(), moves.numel(), device, 0)
            out = View(children.data_ptr(), children.numel(), device, 0)
            self.ok(self.call(ctx, a, 3, b, m, int(dtype == torch.uint16), 4, out, stream.cuda_stream))
            owner_view = View(owners.data_ptr(), owners.numel(), device, 0)
            self.ok(self.lib.beam_cnot_domain_owners_device(ctx, c.byref(out), 4,
                                                             c.byref(owner_view), c.c_void_p(stream.cuda_stream)))
            self.assertTrue(stream.query())
            consumed = children + 1
        stream.synchronize()
        source = [list(row) for row in parents.cpu().reshape(3, stride).tolist()]
        original = [[(parent * stride + word) * 65537 - 7 for word in range(stride)]
                    for parent in range(3)]
        self.assertEqual(source, original)
        expected = [reference.child(original[parent], n, move)
                    for parent, move in zip([2, 0, 1, 2], move_values)]
        actual = [[x & reference.MASK64 for x in row]
                  for row in children.cpu().reshape(4, stride).tolist()]
        self.assertEqual(actual, [list(row) for row in expected])
        self.assertEqual(owners.cpu().tolist(), [reference.owner(row, n, 7) for row in expected])
        self.assertEqual(consumed.cpu().reshape(4, stride).tolist(),
                         [[reference.signed(x + 1) for x in row] for row in expected])
        del parents, indices, moves, children, owners, consumed

    def test_expansion_and_move_formats(self):
        for n in (2, 4, 64, 144, 256):
            for dtype in (torch.int64, torch.uint16):
                with self.subTest(n=n, dtype=dtype):
                    self.exercise(n, 0, dtype, use_default=(n == 2))
        if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
            self.exercise(4, 1, torch.uint16)

    def test_invalid_views_and_device_only_context(self):
        ctx = self.context(4, parents=2, children=2)
        stream = torch.cuda.Stream(device=0)
        parents = torch.arange(8, dtype=torch.int64, device='cuda:0')
        index_storage = torch.zeros(8, dtype=torch.int64, device='cuda:0')
        moves = torch.tensor([0, 1], dtype=torch.int64, device='cuda:0')
        children = torch.empty(8, dtype=torch.int64, device='cuda:0')
        stream.wait_stream(torch.cuda.current_stream(0))
        p, i = View(parents.data_ptr(), 8, 0, 0), View(index_storage.data_ptr(), 2, 0, 0)
        m, out = View(moves.data_ptr(), 2, 0, 0), View(children.data_ptr(), 8, 0, 0)
        s = stream.cuda_stream
        self.fails(self.lib.beam_cnot_domain_set_parents(ctx, None, 0, 0), "device-only")
        self.fails(self.call(ctx, p, 3, i, m, 0, 2, out, s), "required=3", "available=2")
        self.fails(self.call(ctx, p, 2, i, m, 0, 3, out, s), "required=3", "available=2")
        self.fails(self.call(ctx, View(p.data, 7, 0, 0), 2, i, m, 0, 2, out, s), "required=8", "available=7")
        self.fails(self.call(ctx, p, 2, View(i.data, 1, 0, 0), m, 0, 2, out, s), "required=2", "available=1")
        self.fails(self.call(ctx, p, 2, i, View(m.data, 1, 0, 0), 1, 2, out, s), "required=2", "available=1")
        self.fails(self.call(ctx, p, 2, i, m, 2, 2, out, s), "move")
        self.fails(self.call(ctx, p, 2, i, m, 0, 2, View(p.data, 8, 0, 0), s), "overlap")
        self.fails(self.call(ctx, p, 2, i, m, 0, 2, View(i.data, 8, 0, 0), s), "overlap")
        self.fails(self.call(ctx, p, 2, i, View(out.data, 2, 0, 0), 1, 2, out, s), "overlap")
        self.fails(self.call(ctx, p, 2, i, View(m.data + 1, 2, 0, 0), 1, 2, out, s), "align")
        self.fails(self.call(ctx, View(p.data, 8, 0, 1), 2, i, m, 0, 2, out, s), "reserved")
        host = (c.c_int64 * 8)()
        self.fails(self.call(ctx, View(c.addressof(host), 8, 0, 0), 2, i, m, 0, 2, out, s), "device")
        empty = View(None, 0, 0, 0)
        self.ok(self.call(ctx, empty, 0, empty, empty, 0, 0, empty, s))
        self.fails(self.call(ctx, empty, 0, i, m, 0, 2, out, s), "empty parent")
        if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
            other = torch.empty(8, dtype=torch.int64, device='cuda:1')
            other_stream = torch.cuda.Stream(device=1)
            self.fails(self.call(ctx, View(other.data_ptr(), 8, 0, 0), 2, i, m, 0, 2, out, s), "device")
            self.fails(self.call(ctx, p, 2, i, m, 0, 2, out, other_stream.cuda_stream), "stream")
            del other
        self.ok(self.call(ctx, p, 2, i, m, 0, 2, out, s))
        expected = [reference.child(list(range(4)), 4, move) for move in [0, 1]]
        self.assertEqual(children.cpu().reshape(2, 4).tolist(), [list(row) for row in expected])
        del parents, index_storage, moves, children

    def test_device_only_context_does_not_allocate_host_stage_buffers(self):
        # The equivalent parent buffer would exceed ordinary GPU memory by
        # orders of magnitude; a device-only context retains only its capacity.
        ctx = self.context(256, parents=1 << 34, children=0)
        self.fails(self.lib.beam_cnot_domain_set_parents(ctx, None, 0, 0), "device-only")

    def test_device_only_context_rejects_another_thread(self):
        ctx = self.context(4, parents=0, children=0)
        empty = View(None, 0, 0, 0)
        result = []

        def worker():
            code = self.call(ctx, empty, 0, empty, empty, 0, 0, empty, 0)
            result.append((code, self.lib.beam_cnot_domain_error().decode()))

        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(result[0][0], 4)
        self.assertIn("different thread", result[0][1])
        self.ok(self.call(ctx, empty, 0, empty, empty, 0, 0, empty, 0))

    def test_full_context_keeps_its_host_stage_results(self):
        ctx = self.context(4, parents=1, children=1, device_only=False)
        host_parent, host_index, host_move = reference.array([1, 2, 4, 8]), reference.array([0]), reference.array([0])
        self.ok(self.lib.beam_cnot_domain_set_parents(ctx, host_parent, 4, 1))
        self.ok(self.lib.beam_cnot_domain_set_descriptors(ctx, host_index, 1, host_move, 1))
        self.ok(self.lib.beam_cnot_domain_expand(ctx))
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.stream(stream):
            parents = torch.tensor([1, 2, 4, 8], dtype=torch.int64, device='cuda:0')
            indices = torch.zeros(1, dtype=torch.int64, device='cuda:0')
            moves = torch.tensor([2], dtype=torch.int64, device='cuda:0')
            children = torch.empty(4, dtype=torch.int64, device='cuda:0')
            self.ok(self.call(ctx, View(parents.data_ptr(), 4, 0, 0), 1,
                              View(indices.data_ptr(), 1, 0, 0), View(moves.data_ptr(), 1, 0, 0),
                              0, 1, View(children.data_ptr(), 4, 0, 0), stream.cuda_stream))
        self.assertEqual(children.cpu().tolist(), [1, 2, 4, 9])
        host_result = reference.array([0] * 4)
        self.ok(self.lib.beam_cnot_domain_download_children(ctx, host_result, 4))
        self.assertEqual(list(host_result), [1, 3, 4, 8])
        del parents, indices, moves, children

    def test_previous_device_restored(self):
        if os.environ.get("BEAM_TEST_SECOND_GPU") != "1":
            self.skipTest("second GPU was not requested")
        ctx = self.context(4, parents=1, children=1)
        with torch.cuda.device(0):
            stream = torch.cuda.Stream(device=0)
            parents = torch.tensor([1, 2, 4, 8], dtype=torch.int64, device='cuda:0')
            indices = torch.zeros(1, dtype=torch.int64, device='cuda:0')
            moves = torch.zeros(1, dtype=torch.uint16, device='cuda:0')
            children = torch.empty(4, dtype=torch.int64, device='cuda:0')
            stream.wait_stream(torch.cuda.current_stream(0))
        with torch.cuda.device(1):
            self.ok(self.call(ctx, View(parents.data_ptr(), 4, 0, 0), 1,
                              View(indices.data_ptr(), 1, 0, 0), View(moves.data_ptr(), 1, 0, 0),
                              1, 1, View(children.data_ptr(), 4, 0, 0), stream.cuda_stream))
            self.assertEqual(torch.cuda.current_device(), 1)
        del parents, indices, moves, children


if __name__ == '__main__':
    unittest.main(verbosity=2)
