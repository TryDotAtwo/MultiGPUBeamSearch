"""The C++ radix pass reproduces the original histogram and mask mutations."""
import os
import pytest
import torch

from multigpubeamsearch import cnot_native
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_rust_selection import SelectionStages
from multigpubeamsearch.cnot_selection import global_winners
from multigpubeamsearch.errors import NativeUnavailable
from multigpubeamsearch.rust_control import extension


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shift", [0, 8, 24, 56])
@pytest.mark.parametrize("remaining", [1, 2, 3])
def test_native_radix_pass_matches_reference(shift, remaining):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    keys = torch.tensor([0, -1, 2**62, -(2**63) + 2, 5], dtype=torch.int64, device=device)
    boundary = torch.tensor([True, False, True, True, False], device=device)
    selected = torch.tensor([False, True, False, False, False], device=device)
    expected_boundary, expected_selected = boundary.clone(), selected.clone()
    expected_digit = (keys >> shift) & 255
    expected_hist = torch.bincount(expected_digit[expected_boundary], minlength=256)
    prefix = torch.cumsum(expected_hist, 0)
    cutoff = int(torch.searchsorted(prefix, remaining).item())
    below = int(prefix[cutoff - 1].item()) if cutoff else 0
    expected_selected |= expected_boundary & (expected_digit < cutoff)
    expected_boundary &= expected_digit == cutoff
    equal = int(expected_hist[cutoff].item())
    ops = cnot_native._cuda_ops()
    digit, histogram = ops.radix_histogram(keys, boundary, shift)
    assert torch.equal(digit, expected_digit)
    assert torch.equal(histogram, expected_hist)
    assert tuple(ops.radix_update(digit, histogram, selected, boundary, remaining)) == (below, equal)
    assert torch.equal(boundary, expected_boundary)
    assert torch.equal(selected, expected_selected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_old_operator_library_retains_reference_radix(monkeypatch):
    try:
        runner = getattr(extension(), "run_cnot_selection", None)
    except NativeUnavailable:
        runner = None
    if runner is None:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            pytest.fail("matching Rust CNOT selection controller is required")
        pytest.skip("matching Rust CNOT selection controller is unavailable")
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    scores = torch.tensor([0., -2., 3., -0.], dtype=torch.float64, device=device)
    words = torch.tensor([7, 5, 3, 1], dtype=torch.int64, device=device)
    world = World(device)
    expected = global_winners(world, scores, words, 2)
    ops = cnot_native._cuda_ops()

    class OlderLibrary:
        def __getattr__(self, name):
            if name in ("radix_histogram", "radix_update"):
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    actual = runner(SelectionStages(world, scores, words), 2)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
