import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from multigpubeamsearch import rust_control
from multigpubeamsearch.contracts import GraphContract
from multigpubeamsearch.errors import NativeUnavailable


def graph(*, names=None, generators=None):
    definition = SimpleNamespace(
        is_permutation_group=lambda: True,
        state_size=4,
        central_state=[0, 1, 2, 3],
        generators_permutations=generators or [[1, 2, 3, 0], [3, 0, 1, 2]],
        generator_names=names or ["forward", "inverse"],
    )
    return SimpleNamespace(definition=definition)


def expected_graph_hash(definition):
    data = {
        "generators_type": "PERMUTATION",
        "generators": [list(row) for row in definition.generators_permutations],
        "central_state": list(definition.central_state),
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def test_rust_builder_matches_public_graph_contract_and_hash():
    assert callable(getattr(rust_control.extension(), "build_graph_contract", None))
    source = graph()
    contract = GraphContract.from_graph(source, torch.tensor([[1, 2, 3, 0]]))

    assert contract.state_len == 4
    assert contract.num_classes == 4
    assert contract.center == (0, 1, 2, 3)
    assert contract.start == (1, 2, 3, 0)
    assert contract.generators == ((1, 2, 3, 0), (3, 0, 1, 2))
    assert contract.generator_names == ("forward", "inverse")
    assert contract.graph_hash == expected_graph_hash(source.definition)

    renamed = GraphContract.from_graph(graph(names=["a", "b"]), [1, 2, 3, 0])
    reordered = GraphContract.from_graph(
        graph(generators=[[3, 0, 1, 2], [1, 2, 3, 0]]), [1, 2, 3, 0]
    )
    assert renamed.graph_hash == contract.graph_hash
    assert reordered.graph_hash != contract.graph_hash


def test_graph_contract_keeps_python_fallback_for_older_extensions(monkeypatch):
    monkeypatch.setattr(rust_control, "build_graph_contract", lambda *args: None)
    source = graph()

    contract = GraphContract.from_graph(source, np.asarray([1, 2, 3, 0], dtype=np.uint8))

    assert contract.graph_hash == expected_graph_hash(source.definition)


@pytest.mark.parametrize(
    "source,start,message",
    [
        (graph(), [0, 0, 2, 3], "start_state multiset"),
        (
            graph(generators=[[0, 0, 2, 3]], names=["move"]),
            [0, 1, 2, 3],
            "complete gather permutation",
        ),
    ],
)
def test_rust_graph_contract_rejects_invalid_state_and_generator(source, start, message):
    with pytest.raises(NativeUnavailable, match=message):
        GraphContract.from_graph(source, start)
