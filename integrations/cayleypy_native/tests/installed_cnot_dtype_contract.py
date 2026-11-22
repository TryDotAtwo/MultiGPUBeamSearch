"""Exercise PyO3 CNOT dtype checks from an installed CPU wheel."""
import torch

from multigpubeamsearch.rust_control import extension


runtime = extension()
root = torch.zeros((1, 64, 1), dtype=torch.int64)
rows = torch.arange(64, dtype=torch.int64)
root[0, rows, 0] = torch.bitwise_left_shift(torch.ones_like(rows), rows)
assert int(root[0, 63, 0]) < 0  # signed int64 still carries the logical high bit

for dtype in (torch.float64, torch.int32):
    with torch.no_grad():
        try:
            runtime.prepare_cnot_input(root.to(dtype), 1, 4032, 1, "qhat")
        except ValueError:
            pass
        else:
            raise AssertionError(f"root dtype {dtype} must be rejected")

runtime.prepare_cnot_input(root, 1, 4032, 1, "qhat")

# Exercise the exact PyO3 CPU batch primitive used by NativeStages. At n=256,
# 65279 is the largest legal directed gate ID and touches the signed top bit.
n = 256
words = (n + 63) // 64
parents = torch.zeros((1, n, words), dtype=torch.int64)
indices = torch.arange(n, dtype=torch.int64)
parents[0, indices, indices // 64] = torch.bitwise_left_shift(
    torch.ones_like(indices), indices % 64
)
parent_index = torch.tensor([0], dtype=torch.int64)
move_id = 65279

for move in (torch.tensor([move_id], dtype=torch.int64),
             torch.tensor([move_id], dtype=torch.uint16)):
    child = runtime._cnot_cpu_children(parents, parent_index, move, n, words)
    assert child.dtype is torch.int64
    assert int(child[0, 254, 3]) == -(1 << 62)
    assert int(child[0, 255, 3]) == -(1 << 63)

for bad_indices, bad_moves in (
    (torch.tensor([0], dtype=torch.int32), torch.tensor([0], dtype=torch.int64)),
    (parent_index, torch.tensor([0.0], dtype=torch.float64)),
):
    try:
        runtime._cnot_cpu_children(parents, bad_indices, bad_moves, n, words)
    except ValueError as error:
        assert "int64 CNOT states and indices" in str(error)
    else:
        raise AssertionError("unsupported CNOT descriptor dtype must be rejected")

try:
    runtime._cnot_cpu_children(
        parents, parent_index, torch.tensor([-1], dtype=torch.int64), n, words
    )
except ValueError as error:
    assert "outside the physical alphabet" in str(error)
else:
    raise AssertionError("negative int64 CNOT move must be rejected")

print("CNOT_PYO3_DTYPE_CONTRACT_OK")
