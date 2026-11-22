"""The native local-goal stage matches the existing full-matrix reference."""
import pytest
import torch

from multigpubeamsearch import cnot, cnot_native
from multigpubeamsearch.cnot_collectives import World
from multigpubeamsearch.cnot_rust_stages import CnotStages


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("n", [4, 65, 144])
@pytest.mark.parametrize("count", [0, 1, 4])
def test_local_goal_matches_reference(n, count):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    identity = cnot_native.cnot_identity(n, device=device)
    alphabet = n * (n - 1)
    moves = torch.tensor([0, alphabet - 1, 1, 2], dtype=torch.int64, device=device)
    roots = identity.expand(4, -1, -1).contiguous()
    parents = cnot_native.cnot_children(roots, torch.arange(4, device=device), moves)
    parents[0] = identity[0]  # An identity state must not be treated as a one-step hit.
    parents = torch.stack((parents, parents), 1).reshape(8, n, -1)[::2][:count]
    lex = torch.tensor([6, 3, 1, 5], dtype=torch.int64, device=device)[:count]
    world = World(device)
    expected = cnot._goal(world, parents, lex, 7, identity)
    stages = CnotStages(world, identity, identity, None, None, "qhat", False)
    stages.parents, stages.lex = parents, lex
    assert stages.goal(7) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_old_operator_library_retains_goal_fallback(monkeypatch):
    device = torch.device("cuda", torch.cuda.device_count() - 1)
    torch.cuda.set_device(device)
    identity = cnot_native.cnot_identity(4, device=device)
    parents = cnot_native.cnot_children(identity, torch.tensor([0], device=device),
                                         torch.tensor([0], device=device))
    lex, world = torch.tensor([0], device=device), World(device)
    expected = cnot._goal(world, parents, lex, 0, identity)
    ops = cnot_native._cuda_ops()

    class OlderLibrary:
        def __getattr__(self, name):
            if name == "local_goal":
                raise AttributeError(name)
            return getattr(ops, name)

    monkeypatch.setattr(cnot_native, "_cuda_ops", lambda: OlderLibrary())
    stages = CnotStages(world, identity, identity, None, None, "qhat", False)
    stages.parents, stages.lex = parents, lex
    assert stages.goal(0) == expected
