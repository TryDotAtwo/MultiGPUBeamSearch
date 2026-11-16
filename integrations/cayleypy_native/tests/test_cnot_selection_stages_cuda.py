"""Native selection preparation and ordering match the Python reference."""
import os

import pytest
import torch

from multigpubeamsearch import cnot_native
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_rust_selection import SelectionStages
from multigpubeamsearch.cnot_selection import global_winners
from multigpubeamsearch.errors import NativeUnavailable
from multigpubeamsearch.rust_control import extension


def runner():
    try:
        select = getattr(extension(), "run_cnot_selection", None)
    except NativeUnavailable:
        select = None
    if select is None:
        if os.environ.get("REQUIRE_RUST_CNOT") == "1":
            pytest.fail("matching Rust CNOT selection controller is required")
        pytest.skip("matching Rust CNOT selection controller is unavailable")
    return select


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("values,words,width", [
    ([], [], 1),
    ([0.], [7], 1),
    ([0., -0., -2., 3., 3.], [9, 2, 5, 1, 8], 1),
    ([0., -0., -2., 3., 3.], [9, 2, 5, 1, 8], 3),
    ([0., -0., -2., 3., 3.], [9, 2, 5, 1, 8], 5),
    ([0., -0., -2., 3., 3.], [9, 2, 5, 1, 8], 9),
    ([-1e300, 1e300, 1e-280, -1e-280], [4, 3, 2, 1], 2),
])
def test_native_selection_stages_match_reference(values, words, width):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    scores = torch.tensor(values, dtype=torch.float64, device=device)
    keys = torch.tensor(words, dtype=torch.int64, device=device)
    world = World(device)
    expected = global_winners(world, scores, keys, width)
    stages = SelectionStages(world, scores, keys)
    actual = runner()(stages, width)
    assert stages.selection_ops is not None
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_old_operator_library_retains_selection_stages(monkeypatch):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    scores = torch.tensor([0., -2., 3., -0.], dtype=torch.float64, device=device)
    words = torch.tensor([7, 5, 3, 1], dtype=torch.int64, device=device)
    world = World(device)
    expected = global_winners(world, scores, words, 2)
    ops = cnot_native._cuda_ops()

    class OlderLibrary:
        def __getattr__(self, name):
            if name in ("selection_prepare", "selection_records", "selection_order"):
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    stages = SelectionStages(world, scores, words)
    actual = runner()(stages, 2)
    assert stages.selection_ops is None
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
