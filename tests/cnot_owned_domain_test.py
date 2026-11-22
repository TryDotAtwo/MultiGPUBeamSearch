"""Standalone CNOT domain ABI checks; run with an explicitly built CUDA library.

The oracle below uses Python integers and never calls the CUDA/Torch bridge.
The library under test is selected with BEAM_CNOT_OWNED_LIBRARY.
"""
import ctypes as c
import os
import random
import threading
import unittest


class Config(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                ("device", c.c_int32), ("n", c.c_int32), ("world", c.c_int32),
                ("reserved", c.c_uint32), ("parents", c.c_uint64),
                ("children", c.c_uint64), ("closed", c.c_uint64)]


I64P = c.POINTER(c.c_int64)
U64P = c.POINTER(c.c_uint64)
MASK64 = (1 << 64) - 1


def signed(x):
    x &= MASK64
    return x if x < (1 << 63) else x - (1 << 64)


def mix(x):
    x &= MASK64
    x = ((x ^ (x >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
    x = ((x ^ (x >> 27)) * 0x94d049bb133111eb) & MASK64
    return x ^ (x >> 31)


def canonical(state, n):
    w = (n + 63) // 64
    tail = (1 << (n % 64)) - 1 if n % 64 else MASK64
    return tuple((word & MASK64) & (tail if i % w == w - 1 else MASK64)
                 for i, word in enumerate(state))


def child(parent, n, move):
    words = list(canonical(parent, n))
    width = (n + 63) // 64
    control, offset = divmod(move, n - 1)
    target = offset + (offset >= control)
    for j in range(width):
        words[target * width + j] ^= words[control * width + j]
    return canonical(words, n)


def owner(state, n, world):
    lo = hi = 0
    for i, value in enumerate(canonical(state, n)):
        k = i + 1
        lo ^= mix(value ^ ((k * 0x9e3779b97f4a7c15) & MASK64))
        hi ^= mix(value + k * 0xd6e8feb86659fd93)
    lo ^= n
    hi ^= n << 32
    rotated = ((hi << 32) | (hi >> 32)) & MASK64
    key = lo ^ rotated ^ 0x243f6a8885a308d3
    key ^= mix(hi + 0x9e3779b97f4a7c15)
    return mix(key) % world


def array(values):
    return (c.c_int64 * len(values))(*(signed(v) for v in values))


class DomainTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = os.environ.get("BEAM_CNOT_OWNED_LIBRARY")
        if not path:
            raise unittest.SkipTest("standalone CUDA domain library was not supplied")
        cls.lib = c.CDLL(path)
        cls.lib.beam_cnot_domain_abi_version.restype = c.c_uint32
        cls.lib.beam_cnot_domain_error.restype = c.c_char_p
        cls.lib.beam_cnot_domain_create.argtypes = [c.POINTER(Config), c.POINTER(c.c_void_p)]
        cls.lib.beam_cnot_domain_destroy.argtypes = [c.c_void_p]
        for name in ("parents", "closed"):
            getattr(cls.lib, "beam_cnot_domain_set_" + name).argtypes = [c.c_void_p, I64P, c.c_uint64, c.c_uint64]
        cls.lib.beam_cnot_domain_set_descriptors.argtypes = [c.c_void_p, I64P, c.c_uint64, I64P, c.c_uint64]
        for name in ("expand", "owners"):
            getattr(cls.lib, "beam_cnot_domain_" + name).argtypes = [c.c_void_p]
        cls.lib.beam_cnot_domain_first_unseen.argtypes = [c.c_void_p, U64P]
        for name in ("children", "owners", "selected"):
            getattr(cls.lib, "beam_cnot_domain_download_" + name).argtypes = [c.c_void_p, I64P, c.c_uint64]

    def ok(self, result):
        self.assertEqual(result, 0, self.lib.beam_cnot_domain_error().decode())

    def fails(self, result, *parts):
        self.assertNotEqual(result, 0)
        error = self.lib.beam_cnot_domain_error().decode()
        for part in parts:
            self.assertIn(part, error)

    def make(self, n, device=0, world=7, parents=5, children=7, closed=4):
        config = Config(1, c.sizeof(Config), device, n, world, 0, parents, children, closed)
        handle = c.c_void_p()
        self.ok(self.lib.beam_cnot_domain_create(c.byref(config), c.byref(handle)))
        self.addCleanup(lambda: self.ok(self.lib.beam_cnot_domain_destroy(handle)))
        return handle

    def load(self, ctx, name, states, stride):
        values = array([v for state in states for v in state])
        self.ok(getattr(self.lib, "beam_cnot_domain_set_" + name)(ctx, values, len(values), len(states)))

    def descriptors(self, ctx, indices, moves):
        a, b = array(indices), array(moves)
        self.ok(self.lib.beam_cnot_domain_set_descriptors(ctx, a, len(a), b, len(b)))

    def download(self, ctx, name, length):
        out = array([0] * length)
        self.ok(getattr(self.lib, "beam_cnot_domain_download_" + name)(ctx, out, length))
        return [x & MASK64 for x in out]

    def check_batch(self, n, device=0, world=7):
        rng = random.Random(2030 + n)
        width = (n + 63) // 64
        stride = n * width
        parents = [tuple(rng.getrandbits(64) for _ in range(stride)) for _ in range(4)]
        moves = [0, n - 1, n * (n - 1) - 1, 0, 1, 1, n - 1]
        indices = [0, 1, 2, 0, 3, 3, 2]
        expected = [child(parents[i], n, m) for i, m in zip(indices, moves)]
        # Duplicate an existing child in closed, with only padding altered.
        already = list(expected[1])
        if n % 64:
            already[width - 1] |= 1 << 63
        distinct = list(expected[2])
        distinct[-1] ^= 1
        closed = [already, distinct]
        ctx = self.make(n, device=device, world=world)
        self.load(ctx, "parents", parents, stride)
        self.load(ctx, "closed", closed, stride)
        self.descriptors(ctx, indices, moves)
        self.ok(self.lib.beam_cnot_domain_expand(ctx))
        actual = self.download(ctx, "children", len(expected) * stride)
        self.assertEqual(actual, [word for state in expected for word in state])
        self.ok(self.lib.beam_cnot_domain_owners(ctx))
        self.assertEqual(self.download(ctx, "owners", len(expected)),
                         [owner(state, n, world) for state in expected])
        count = c.c_uint64(999)
        self.ok(self.lib.beam_cnot_domain_first_unseen(ctx, c.byref(count)))
        seen = {canonical(row, n) for row in closed}
        expected_ids = []
        for i, state in enumerate(expected):
            if state not in seen:
                expected_ids.append(i)
                seen.add(state)
        self.assertEqual(count.value, len(expected_ids))
        self.assertEqual(self.download(ctx, "selected", count.value), expected_ids)
        # Existing allocations and stream remain usable after a full batch.
        self.ok(self.lib.beam_cnot_domain_expand(ctx))
        self.ok(self.lib.beam_cnot_domain_first_unseen(ctx, c.byref(count)))
        self.assertEqual(self.download(ctx, "selected", count.value), expected_ids)

    def test_small_wide_and_exact_capacities(self):
        self.assertEqual(self.lib.beam_cnot_domain_abi_version(), 1)
        for n in (2, 4, 64, 144, 256):
            with self.subTest(n=n):
                self.check_batch(n)
        self.check_batch(4, world=65536)

    def test_second_device_separate_context_when_requested(self):
        if os.environ.get("BEAM_TEST_SECOND_GPU") != "1":
            self.skipTest("second GPU was not requested")
        self.check_batch(4, device=1)

    def test_device_restoration_and_independent_contexts(self):
        if os.environ.get("BEAM_TEST_SECOND_GPU") != "1":
            self.skipTest("second GPU was not requested")
        runtime = c.CDLL(os.environ.get("BEAM_CUDART", "libcudart.so.12"))
        runtime.cudaSetDevice.argtypes = [c.c_int]
        runtime.cudaGetDevice.argtypes = [c.POINTER(c.c_int)]
        current = c.c_int(-1)
        self.assertEqual(runtime.cudaSetDevice(1), 0)
        self.addCleanup(lambda: self.assertEqual(runtime.cudaSetDevice(0), 0))
        first = self.make(4, device=0)
        second = self.make(4, device=1)
        for ctx in (first, second):
            self.load(ctx, "parents", [[1, 2, 4, 8]], 4)
            self.load(ctx, "closed", [], 4)
            self.descriptors(ctx, [0], [0])
            self.ok(self.lib.beam_cnot_domain_expand(ctx))
            self.assertEqual(self.download(ctx, "children", 4), [1, 3, 4, 8])
            self.assertEqual(runtime.cudaGetDevice(c.byref(current)), 0)
            self.assertEqual(current.value, 1)

    def test_zero_invalidation_and_errors(self):
        ctx = self.make(4, parents=1, children=1, closed=1)
        self.load(ctx, "parents", [[1, 2, 4, 8]], 4)
        self.load(ctx, "closed", [], 4)
        self.descriptors(ctx, [], [])
        self.ok(self.lib.beam_cnot_domain_expand(ctx))
        self.ok(self.lib.beam_cnot_domain_owners(ctx))
        count = c.c_uint64(42)
        self.ok(self.lib.beam_cnot_domain_first_unseen(ctx, c.byref(count)))
        self.assertEqual(count.value, 0)
        self.assertEqual(self.download(ctx, "children", 0), [])
        self.descriptors(ctx, [0], [0])
        self.ok(self.lib.beam_cnot_domain_expand(ctx))
        self.fails(self.lib.beam_cnot_domain_download_children(ctx, array([]), 0), "required=4", "available=0")
        self.fails(self.lib.beam_cnot_domain_set_parents(ctx, array([0] * 8), 8, 2), "required=2", "available=1")
        self.fails(self.lib.beam_cnot_domain_set_parents(ctx, array([0] * 3), 3, 1), "required=4", "available=3")
        self.fails(self.lib.beam_cnot_domain_set_descriptors(ctx, array([1]), 1, array([0]), 1), "parent")
        self.fails(self.lib.beam_cnot_domain_set_descriptors(ctx, array([-1]), 1, array([0]), 1), "parent")
        self.fails(self.lib.beam_cnot_domain_set_descriptors(ctx, array([0]), 1, array([12]), 1), "move")
        self.fails(self.lib.beam_cnot_domain_set_descriptors(ctx, array([0]), 1, array([-1]), 1), "move")
        self.fails(self.lib.beam_cnot_domain_set_descriptors(ctx, array([0]), 1, array([]), 0), "length")
        self.fails(self.lib.beam_cnot_domain_set_parents(ctx, None, 4, 1), "null")
        self.fails(self.lib.beam_cnot_domain_set_descriptors(ctx, array([0, 0]), 2, array([0, 0]), 2), "required=2", "available=1")
        self.load(ctx, "parents", [[1, 2, 4, 8]], 4)
        self.fails(self.lib.beam_cnot_domain_download_children(ctx, array([0] * 4), 4), "invalid")
        self.fails(self.lib.beam_cnot_domain_expand(ctx), "descriptor")
        self.descriptors(ctx, [0], [0])
        self.ok(self.lib.beam_cnot_domain_expand(ctx))
        self.ok(self.lib.beam_cnot_domain_first_unseen(ctx, c.byref(count)))
        self.load(ctx, "closed", [[1, 3, 4, 8]], 4)
        self.fails(self.lib.beam_cnot_domain_download_selected(ctx, array([0]), 1), "invalid")

        empty = self.make(4, parents=0, children=0, closed=0)
        self.load(empty, "parents", [], 4)
        self.load(empty, "closed", [], 4)
        self.descriptors(empty, [], [])
        self.ok(self.lib.beam_cnot_domain_expand(empty))
        self.ok(self.lib.beam_cnot_domain_first_unseen(empty, c.byref(count)))
        self.assertEqual(count.value, 0)

    def test_rejects_bad_configuration_and_wrong_thread(self):
        for config in (Config(2, c.sizeof(Config), 0, 4, 1, 0, 1, 1, 1),
                       Config(1, c.sizeof(Config) - 1, 0, 4, 1, 0, 1, 1, 1),
                       Config(1, c.sizeof(Config), 0, 1, 1, 0, 1, 1, 1),
                       Config(1, c.sizeof(Config), 0, 4, 0, 0, 1, 1, 1),
                       Config(1, c.sizeof(Config), 0, 4, 65537, 0, 1, 1, 1),
                       Config(1, c.sizeof(Config), 0, 4, 1, 1, 1, 1, 1),
                       Config(1, c.sizeof(Config), 0, 4, 1, 0, 1, 2**31, 1)):
            result = c.c_void_p()
            self.fails(self.lib.beam_cnot_domain_create(c.byref(config), c.byref(result)))
            self.assertFalse(result.value)
        config = Config(1, c.sizeof(Config), 0, 4, 1, 0, 1, 2**31, 1)
        result = c.c_void_p()
        self.fails(self.lib.beam_cnot_domain_create(c.byref(config), c.byref(result)),
                   "required=2147483648", "available=2147483647")
        overflow = Config(1, c.sizeof(Config), 0, 4, 1, 0, 2**64 - 1, 1, 1)
        self.fails(self.lib.beam_cnot_domain_create(c.byref(overflow), c.byref(result)),
                   "size overflow", "required=", "available=")
        ctx = self.make(4)
        outcomes = []
        def other_thread():
            code = self.lib.beam_cnot_domain_destroy(ctx)
            outcomes.append((code, self.lib.beam_cnot_domain_error().decode()))
        thread = threading.Thread(target=other_thread)
        thread.start()
        thread.join()
        self.assertNotEqual(outcomes[0][0], 0)
        self.assertIn("thread", outcomes[0][1])
        self.load(ctx, "parents", [], 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
