"""C++ CNOT distinct preparation retains full-matrix and generation semantics."""
import pytest
import torch

from multigpubeamsearch import cnot, cnot_native
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_rust_stages import CnotStages


class Scorer:
    def child_values(self, children):
        return torch.arange(len(children), dtype=torch.float64, device=children.device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n", [4, 65, 144])
@pytest.mark.parametrize("empty", [False, True])
def test_cuda_distinct_matches_reference(n, empty):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    words = (n + 63) // 64
    states = torch.zeros((5, n, words), dtype=torch.int64, device=device)
    states[0, 0, 0] = states[1, 0, 0] = 1
    states[2, -1, -1] = 1
    states[3, 0, 0] = 2
    states[4, -1, -1] = 1
    # The public tensor helper makes the closed set contiguous before launch.
    closed = torch.stack((states[3], states[2], states[3], states[2]))[::2]
    assert not closed.is_contiguous()
    meta = torch.tensor([[10, 20, 5, 0], [11, 21, 1, 1], [12, 22, 7, 2],
                         [13, 23, 0, 3], [14, 24, 3, 4]], device=device)
    incoming = torch.cat((meta, states.reshape(5, -1)), 1)
    if empty:
        incoming = incoming[:0]
    world, scorer = World(device), Scorer()
    expected = cnot._distinct(world, incoming, closed, n, words, scorer)
    stages = CnotStages(world, states[:1], None, None, scorer, "qhat", False)
    stages.incoming, stages.closed = incoming, closed
    actual = stages._distinct()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_old_operator_library_retains_reference_distinct(monkeypatch):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    states = torch.zeros((1, 4, 1), dtype=torch.int64, device=device)
    incoming = torch.cat((torch.zeros((1, 4), dtype=torch.int64, device=device), states.reshape(1, -1)), 1)
    closed = states[:0]
    world, scorer = World(device), Scorer()
    expected = cnot._distinct(world, incoming, closed, 4, 1, scorer)
    ops = cnot_native._cuda_ops()

    class OlderLibrary:
        def __getattr__(self, name):
            if name == "distinct_candidates":
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    stages = CnotStages(world, states, None, None, scorer, "qhat", False)
    stages.incoming, stages.closed = incoming, closed
    actual = stages._distinct()
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
