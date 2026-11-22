"""Existing full-matrix CNOT distinctness through borrowed CUDA views."""
import ctypes as c
import gc
import os
import threading
import unittest

import torch

import cnot_owned_domain_test as reference
from cnot_borrowed_device_test import View


class Workspace(c.Structure):
    _fields_ = [("ordinals", View), ("flags", View), ("selected", View),
                ("count", View), ("scratch", View)]


def view(tensor, device):
    return View(tensor.data_ptr(), tensor.numel(), device, 0)


def canonical(row, n):
    words = (n + 63) // 64
    mask = (1 << (n % 64)) - 1 if n % 64 else reference.MASK64
    return tuple((x & reference.MASK64) & (mask if i % words == words - 1 else reference.MASK64)
                 for i, x in enumerate(row))


def oracle(closed, children, n):
    seen = {canonical(row, n) for row in closed}
    chosen = []
    for index, row in enumerate(children):
        key = canonical(row, n)
        if key not in seen:
            seen.add(key)
            chosen.append(index)
    return chosen


class BorrowedUnseenTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("BEAM_CNOT_OWNED_LIBRARY")
        if not path:
            raise unittest.SkipTest("standalone CUDA domain library was not supplied")
        cls.lib = c.CDLL(path)
        cls.lib.beam_cnot_domain_error.restype = c.c_char_p
        cls.lib.beam_cnot_domain_create_device.argtypes = [c.POINTER(reference.Config), c.POINTER(c.c_void_p)]
        cls.lib.beam_cnot_domain_destroy.argtypes = [c.c_void_p]
        cls.lib.beam_cnot_domain_unseen_scratch_bytes.argtypes = [
            c.c_void_p, c.c_uint64, c.c_uint64, c.POINTER(c.c_uint64)]
        cls.lib.beam_cnot_domain_first_unseen_device.argtypes = [
            c.c_void_p, c.POINTER(View), c.c_uint64, c.POINTER(View), c.c_uint64,
            c.POINTER(Workspace), c.c_void_p]

    @classmethod
    def tearDownClass(cls):
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

    def context(self, n, device=0, children=8, closed=4):
        cfg = reference.Config(1, c.sizeof(reference.Config), device, n, 7, 0,
                               0, children, closed)
        ctx = c.c_void_p()
        self.ok(self.lib.beam_cnot_domain_create_device(c.byref(cfg), c.byref(ctx)))
        self.addCleanup(lambda: self.ok(self.lib.beam_cnot_domain_destroy(ctx)))
        return ctx

    def query(self, ctx, closed, children):
        size = c.c_uint64(123)
        self.ok(self.lib.beam_cnot_domain_unseen_scratch_bytes(ctx, closed, children, c.byref(size)))
        return size.value

    def buffers(self, ctx, device, closed, children):
        size = self.query(ctx, closed, children)
        tensors = [torch.empty(closed + children if children else 0, dtype=torch.int32, device=device),
                   torch.empty(children, dtype=torch.uint8, device=device),
                   torch.empty(children, dtype=torch.int64, device=device),
                   torch.full((1,), -1, dtype=torch.int32, device=device),
                   torch.empty(size, dtype=torch.uint8, device=device)]
        return tensors, Workspace(*(view(t, device) for t in tensors))

    def call(self, ctx, closed, closed_count, children, child_count, buffers, stream):
        return self.lib.beam_cnot_domain_first_unseen_device(
            ctx, c.byref(closed), closed_count, c.byref(children), child_count,
            c.byref(buffers), c.c_void_p(stream))

    def exercise(self, n, device, default=False):
        stride = n * ((n + 63) // 64)
        rows = [[0] * stride for _ in range(5)]
        rows[1][0], rows[2][0], rows[3][0] = 2, 3, 1
        # Distinguish two matrices solely by their last logical word.
        rows[4] = list(rows[3])
        rows[4][-1] = reference.signed((rows[4][-1] & reference.MASK64) ^ 1)
        closed = [rows[0], rows[1]]
        dirty = list(rows[2])
        if n % 64:
            dirty[-1] = reference.signed((dirty[-1] & reference.MASK64) ^ (1 << (n % 64)))
        children = [rows[0], rows[2], dirty, rows[3], rows[4], rows[1], rows[3]]
        expected = oracle(closed, children, n)
        self.assertEqual(expected, [1, 3, 4])
        ctx = self.context(n, device=device)
        stream = torch.cuda.default_stream(device) if default else torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            a = torch.tensor(closed, dtype=torch.int64, device=device)
            b = torch.tensor(children, dtype=torch.int64, device=device)
            tensors, workspace = self.buffers(ctx, device, len(closed), len(children))
            self.ok(self.call(ctx, view(a, device), len(closed), view(b, device), len(children),
                              workspace, stream.cuda_stream))
            self.assertTrue(stream.query())
            self.assertEqual(tensors[3].cpu().tolist()[0], len(expected))
            self.assertEqual(tensors[2][:len(expected)].cpu().tolist(), expected)
            # Reuse precisely the same scratch for a different, shorter batch.
            self.ok(self.call(ctx, view(a, device), 0, view(b, device), 2, workspace, stream.cuda_stream))
            self.assertEqual(tensors[3].cpu().tolist()[0], 2)
            self.assertEqual(tensors[2][:2].cpu().tolist(), [0, 1])
            self.assertEqual(a.cpu().tolist(), closed)
            self.assertEqual(b.cpu().tolist(), children)
        del a, b, tensors

    def test_full_matrix_order_tail_and_reuse(self):
        for n in (2, 4, 64, 65, 144, 256):
            with self.subTest(n=n):
                self.exercise(n, 0, default=n == 2)
        if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
            self.exercise(65, 1)

    def test_empty_and_invalid_buffers(self):
        ctx = self.context(4, children=3, closed=1)
        stream = torch.cuda.Stream(device=0)
        with torch.cuda.device(0), torch.cuda.stream(stream):
            closed = torch.tensor([1, 2, 4, 8], dtype=torch.int64, device=0)
            children = torch.tensor([1, 2, 4, 8, 2, 3, 4, 8], dtype=torch.int64, device=0)
            tensors, w = self.buffers(ctx, 0, 1, 2)
            a, b = view(closed, 0), view(children, 0)
            args = (ctx, a, 1, b, 2)
            self.fails(self.lib.beam_cnot_domain_unseen_scratch_bytes(ctx, 2, 2, c.byref(c.c_uint64())),
                       "required=2", "available=1")
            self.fails(self.lib.beam_cnot_domain_unseen_scratch_bytes(ctx, 1, 4, c.byref(c.c_uint64())),
                       "required=4", "available=3")
            self.fails(self.lib.beam_cnot_domain_unseen_scratch_bytes(ctx, 1, 2, None), "null")
            self.fails(self.call(*args, Workspace(View(w.ordinals.data, 2, 0, 0), w.flags, w.selected,
                                                   w.count, w.scratch), stream.cuda_stream),
                       "required=3", "available=2")
            self.fails(self.call(*args, Workspace(w.ordinals, View(w.flags.data, 1, 0, 0), w.selected,
                                                   w.count, w.scratch), stream.cuda_stream),
                       "required=2", "available=1")
            self.fails(self.call(*args, Workspace(w.ordinals, w.flags, View(w.selected.data, 1, 0, 0),
                                                   w.count, w.scratch), stream.cuda_stream),
                       "required=2", "available=1")
            self.fails(self.call(*args, Workspace(w.ordinals, w.flags, w.selected, w.count,
                                                   View(w.scratch.data, w.scratch.elements - 1, 0, 0)),
                                 stream.cuda_stream), "scratch", "required=", "available=")
            self.fails(self.call(*args, Workspace(w.ordinals, w.flags, View(a.data, 2, 0, 0),
                                                   w.count, w.scratch), stream.cuda_stream), "overlap")
            self.fails(self.call(*args, Workspace(w.ordinals, w.flags, w.selected,
                                                   View(w.ordinals.data, 1, 0, 0), w.scratch),
                                 stream.cuda_stream), "overlap")
            self.fails(self.call(*args, Workspace(View(w.ordinals.data + 1, 3, 0, 0), w.flags,
                                                   w.selected, w.count, w.scratch), stream.cuda_stream), "align")
            self.fails(self.call(*args, Workspace(w.ordinals, w.flags, w.selected, w.count,
                                                   View(w.scratch.data + 1, w.scratch.elements, 0, 0)),
                                 stream.cuda_stream), "align")
            if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
                other = torch.empty(8, dtype=torch.int64, device='cuda:1')
                other_stream = torch.cuda.Stream(device=1)
                self.fails(self.call(ctx, View(other.data_ptr(), 4, 0, 0), 1, b, 2,
                                     w, stream.cuda_stream), "device")
                self.fails(self.call(*args, w, other_stream.cuda_stream), "stream")
                del other
            self.ok(self.call(*args, w, stream.cuda_stream))
            self.assertEqual(tensors[3].cpu().tolist()[0], 1)
            self.assertEqual(tensors[2][:1].cpu().tolist(), [1])
            self.ok(self.call(ctx, a, 1, a, 1, w, stream.cuda_stream))
            self.assertEqual(tensors[3].cpu().tolist()[0], 0)
            self.assertEqual(self.query(ctx, 1, 0), 0)
            empty = View(None, 0, 0, 0)
            zeros = Workspace(empty, empty, empty, w.count, empty)
            self.ok(self.call(ctx, a, 1, empty, 0, zeros, stream.cuda_stream))
            self.assertEqual(tensors[3].cpu().tolist()[0], 0)
        del closed, children, tensors

    def test_wrong_thread_and_device_restoration(self):
        ctx = self.context(4, children=0, closed=0)
        result = []
        def worker():
            count = c.c_uint64()
            code = self.lib.beam_cnot_domain_unseen_scratch_bytes(ctx, 0, 0, c.byref(count))
            result.append((code, self.lib.beam_cnot_domain_error().decode()))
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        self.assertEqual(result[0][0], 4)
        self.assertIn("different thread", result[0][1])
        if os.environ.get("BEAM_TEST_SECOND_GPU") == "1":
            with torch.cuda.device(1):
                self.assertEqual(self.query(ctx, 0, 0), 0)
                self.assertEqual(torch.cuda.current_device(), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
