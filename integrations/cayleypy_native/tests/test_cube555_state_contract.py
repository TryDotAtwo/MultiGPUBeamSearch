"""Colored 150-facelet states must retain positions beyond the old 120-byte limit."""
from types import SimpleNamespace

import pytest

from multigpubeamsearch.build import shape_contract
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.errors import NativeUnavailable


def colored_graph(size):
    center = tuple(i % 6 for i in range(size))
    move = list(range(size))
    move[0], move[-1] = move[-1], move[0]
    definition = SimpleNamespace(
        state_size=size, central_state=center,
        generators_permutations=[move], generator_names=['swap'],
        is_permutation_group=lambda: True,
    )
    return SimpleNamespace(definition=definition), tuple(center[i] for i in move)


def test_cube555_preserves_tail_positions_and_replays():
    graph, start = colored_graph(150)
    contract = GraphContract.from_graph(graph, start)
    assert contract.state_len == 150
    assert contract.generators[0][0] == 149
    assert contract.replay([0])
    assert not contract.replay([])
    assert shape_contract(contract) == dict(state_len=150, storage_len=160,
                                           move_count=1, alignment=16)


def test_state_beyond_supported_cube555_extent_is_rejected():
    graph, start = colored_graph(151)
    with pytest.raises(NativeUnavailable, match='150'):
        GraphContract.from_graph(graph, start)


def test_cube555_unique_sticker_ids_keep_unsigned_tail():
    graph, _ = colored_graph(150)
    graph.definition.central_state = tuple(range(150))
    start = tuple(graph.definition.central_state[i]
                  for i in graph.definition.generators_permutations[0])
    contract = GraphContract.from_graph(graph, start)
    assert contract.num_classes == 150
    assert contract.start[0] == 149
    assert contract.replay([0])
