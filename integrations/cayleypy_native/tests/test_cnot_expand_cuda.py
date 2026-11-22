"""Batch expansion must preserve action order, records and full-matrix owners."""
import pytest
import torch

from multigpubeamsearch import cnot, cnot_native
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_rust_stages import CnotStages


class TiedScorer:
    def parent_scores(self, parents, mode):
        alphabet = parents.size(1) * (parents.size(1) - 1)
        return (torch.arange(alphabet, dtype=torch.float64, device=parents.device) % 7).expand(len(parents), -1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n,count,k", [(4, 0, 3), (4, 3, 3), (4, 3, 12), (65, 2, 5), (144, 2, 5)])
@pytest.mark.parametrize("mode", ["qhat", "proposal"])
def test_cuda_expansion_matches_reference(n, count, k, mode):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    words = (n + 63) // 64
    parents = torch.arange(count * n * words, dtype=torch.int64, device=device).reshape(count, n, words)
    lex = torch.arange(count, dtype=torch.int64, device=device).flip(0)
    scorer = TiedScorer()
    world = World(device)
    world.rank, world.size = 1, 3
    expected, expected_owners = cnot._expand(world, parents, lex, 7, scorer, mode, k)
    scores = scorer.parent_scores(parents, mode) if count and k < n * (n - 1) else None
    actual, actual_owners = cnot_native._cuda_ops().expand_candidates(
        parents, lex, scores, 7, world.rank, world.size, k, mode == "proposal")
    assert torch.equal(actual, expected)
    assert torch.equal(actual_owners, expected_owners)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_old_operator_library_retains_reference_expansion(monkeypatch):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    parents = torch.arange(8, dtype=torch.int64, device=device).reshape(2, 4, 1)
    lex = torch.tensor([1, 0], dtype=torch.int64, device=device)
    scorer, world = TiedScorer(), World(device)
    world.rank, world.size = 1, 3
    expected = cnot._expand(world, parents, lex, 7, scorer, "qhat", 3)
    ops = cnot_native._cuda_ops()

    class OlderLibrary:
        def __getattr__(self, name):
            if name == "expand_candidates":
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    stages = CnotStages(world, parents, None, None, scorer, "qhat", False)
    stages.parents, stages.lex = parents, lex
    actual = stages._expand(7, 3)
    assert all(torch.equal(a, b) for a, b in zip(actual, expected))
