import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
import torch
from cayleypy import CayleyGraph, CayleyGraphDef, MatrixGroups, PermutationGroups
from cayleypy.models.checkpoint import graph_hash

from cayleypy_native.contracts import GraphContract
from cayleypy_native.errors import NativeUnavailable


def graph(center=None, names=None):
    definition = CayleyGraphDef.create(
        [[1, 2, 3, 0], [3, 0, 1, 2]], central_state=center,
        generator_names=names or ['name with "quotes".and\nnewlines', "-inverse"],
    )
    return CayleyGraph(definition, device="cpu", random_seed=11)


def test_graph_contract_preserves_math_and_synthetic_native_order():
    g = graph()
    c = GraphContract.from_graph(g, torch.tensor([[1, 2, 3, 0]]))
    assert c.state_len == 4 and c.move_count == 2 and c.num_classes == 4
    assert c.graph_hash == graph_hash(g.definition)
    assert c.generator_names == tuple(g.definition.generator_names)
    info = json.loads(json.dumps(c.to_puzzle_info()))
    assert list(info["generators"]) == ["m0", "m1"]
    assert info["generators"]["m0"] == [1, 2, 3, 0]
    assert info["central_state"] == [0, 1, 2, 3]
    assert c.replay([1])
    assert not c.replay([0])
    assert torch.equal(g.apply_path(c.start, [1])[0], g.central_state)
    with pytest.raises(FrozenInstanceError):
        c.state_len = 5


def test_colored_state_is_supported_without_renumbering():
    c = GraphContract.from_graph(graph([0, 0, 1, 1]), np.array([1, 1, 0, 0], dtype=np.uint8))
    assert c.num_classes == 2
    assert c.start == (1, 1, 0, 0)
    assert c.replay((0, 0))


@pytest.mark.parametrize("start", [
    [0.0, 1.0, 2.0, 3.0], [False, 1, 2, 3], ["0", "1", "2", "3"],
    torch.tensor([0., 1., 2., 3.]), np.array([0., 1., 2., 3.]),
    [[0, 1, 2, 3], [0, 1, 2, 3]], [[[0, 1, 2, 3]]], [0, 1, 2],
    [0, 1, 2, 128], [0, 1, 2, -1], [0, 1, 1, 3],
])
def test_invalid_state_rejected_without_coercion(start):
    with pytest.raises(NativeUnavailable):
        GraphContract.from_graph(graph(), start)


@pytest.mark.parametrize("path", [[-1], [2], [True], [0.0], ["1"], None])
def test_replay_rejects_invalid_move_ids(path):
    assert not GraphContract.from_graph(graph(), [1, 2, 3, 0]).replay(path)


def test_matrix_and_state_storage_limit_rejected():
    m = CayleyGraph(MatrixGroups.heisenberg(), device="cpu")
    with pytest.raises(NativeUnavailable, match="permutation"):
        GraphContract.from_graph(m, m.central_state)
    g = CayleyGraph(PermutationGroups.lrx(121), device="cpu")
    with pytest.raises(NativeUnavailable, match="120"):
        GraphContract.from_graph(g, g.central_state)


def test_hash_ignores_names_but_binds_order_and_center():
    a = GraphContract.from_graph(graph(), [0, 1, 2, 3])
    b = GraphContract.from_graph(graph(names=["x", "y"]), [0, 1, 2, 3])
    assert a.graph_hash == b.graph_hash
    reversed_def = CayleyGraphDef.create(list(reversed(graph().definition.generators)))
    reverse = CayleyGraph(reversed_def, device="cpu")
    assert GraphContract.from_graph(reverse, reverse.central_state).graph_hash != a.graph_hash
