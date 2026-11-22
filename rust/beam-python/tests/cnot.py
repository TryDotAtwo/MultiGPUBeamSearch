"""Compare the real PyO3 CNOT contract with independent Python integer rows."""
import importlib.util
import gc
import random
import sys


spec = importlib.util.spec_from_file_location("_beam_runtime", sys.argv[1])
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def rows_from_gates(n, gates):
    rows = [1 << i for i in range(n)]
    for move in gates:
        control, slot = divmod(move, n - 1)
        target = slot + (slot >= control)
        rows[target] ^= rows[control]
    return rows


def pack(n, rows):
    width = 8 * ((n + 63) // 64)
    return b"".join(row.to_bytes(width, "little") for row in rows)


randomizer = random.Random(724041)
for n in (2, 3, 63, 64, 65, 127, 128, 129, 144, 255, 256):
    for _ in range(8):
        gates = [randomizer.randrange(n * (n - 1)) for _ in range(35)]
        target = pack(n, rows_from_gates(n, gates))
        matrix = runtime.CnotMatrix(n, target)
        assert matrix.verifies_construction(gates)
        assert not matrix.verifies_construction(gates + [0])
        assert matrix.verifies_construction(gates + [0, 0])
        for invalid in (n * (n - 1), 2**32 - 1):
            try:
                matrix.verifies_construction([invalid])
            except ValueError:
                pass
            else:
                raise AssertionError((n, invalid))
        for malformed in (target[:-1], target + b"\x00"):
            try:
                runtime.CnotMatrix(n, malformed)
            except ValueError:
                pass
            else:
                raise AssertionError((n, "length"))
        singular = bytearray(target)
        width = 8 * ((n + 63) // 64)
        singular[width : 2 * width] = singular[:width]
        try:
            runtime.CnotMatrix(n, bytes(singular))
        except ValueError as error:
            assert "singular" in str(error)
        else:
            raise AssertionError((n, "singular"))
        if n % 64:
            padding = bytearray(target)
            padding[width - 1] |= 0x80
            try:
                runtime.CnotMatrix(n, bytes(padding))
            except ValueError as error:
                assert "padding" in str(error)
            else:
                raise AssertionError((n, "padding"))

target = runtime.CnotMatrix(144, pack(144, rows_from_gates(144, [20591, 20306])))
assert target.verifies_construction([20591, 20306])
assert not target.verifies_construction([20306, 20591])

# Exhaust all 3x3 binary matrices with an independent Python elimination
# oracle, including nonzero dependent rows.
accepted = 0
for encoded in range(1 << 9):
    rows = [(encoded >> (3 * i)) & 7 for i in range(3)]
    pivots = {}
    for original in rows:
        value = original
        while value and value.bit_length() in pivots:
            value ^= pivots[value.bit_length()]
        if value:
            pivots[value.bit_length()] = value
    expected = len(pivots) == 3
    try:
        runtime.CnotMatrix(3, pack(3, rows))
    except ValueError:
        assert not expected
    else:
        assert expected
        accepted += 1
assert accepted == 168

# The retained Rust value does not borrow Python's temporary byte buffer.
raw = pack(144, rows_from_gates(144, [20591, 20306]))
retained = runtime.CnotMatrix(144, raw)
del raw
gc.collect()
assert retained.verifies_construction([20591, 20306])
print("PYO3_CNOT_INDEPENDENT_REPLAY_OK")
