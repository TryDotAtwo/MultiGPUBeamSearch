import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cayleypy import CayleyGraph, CayleyGraphDef

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


@pytest.mark.parametrize("center", [None, [0, 0, 1, 1]])
def test_cayleypy_contract_is_accepted_and_stale_contents_are_rejected(center, tmp_path):
    runtime = pytest.importorskip("_beam_runtime")
    definition = CayleyGraphDef.create(
        [[1, 2, 3, 0], [3, 0, 1, 2]],
        central_state=center,
        generator_names=['quoted "name"', "second\nname"],
    )
    graph_value = CayleyGraph(definition, device="cpu")
    contract = GraphContract.from_graph(graph_value, graph_value.central_state)
    identity = dict(
        graph=dict(center=contract.center, generators=contract.generators),
        graph_hash=contract.graph_hash,
        model_artifact_hash="b" * 64,
        model_backend="mlp",
        runtime_build=dict(binary_sha256="c" * 64),
        devices=[0], beam_width=64, max_steps=1, options={},
        cuda_visible_devices=None, cuda_device_order=None,
    )
    launch = dict(command=[], protocol_command=[], environment={}, cwd=str(tmp_path))
    owner = runtime.SearchSession()
    missing = tmp_path / "must-not-exist"
    result = json.loads(owner.run(json.dumps(identity), json.dumps(launch),
                                  list(contract.center), missing, 1.0, "{}"))
    assert result["path"] == []
    assert not missing.exists()

    identity["graph"]["generators"] = tuple(reversed(contract.generators))
    with pytest.raises(RuntimeError, match="graph hash"):
        owner.run(json.dumps(identity), json.dumps(launch),
                  list(contract.center), missing, 1.0, "{}")
    assert owner.pids == [] and not missing.exists()
    owner.close()


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


def _twelve_move_graph():
    definition = SimpleNamespace(
        is_permutation_group=lambda: True,
        state_size=2,
        central_state=[0, 1],
        generators_permutations=[[1, 0] for _ in range(12)],
        generator_names=[f"move-{index}" for index in range(12)],
    )
    return SimpleNamespace(definition=definition)


def test_rust_writer_emits_legacy_runner_files_in_move_order(tmp_path):
    contract = GraphContract.from_graph(_twelve_move_graph(), [1, 0])

    assert rust_control.write_search_inputs(contract, tmp_path) is True
    expected = (
        '{"central_state": [0, 1], "generators": {'
        '"m0": [1, 0], "m1": [1, 0], "m2": [1, 0], "m3": [1, 0], '
        '"m4": [1, 0], "m5": [1, 0], "m6": [1, 0], "m7": [1, 0], '
        '"m8": [1, 0], "m9": [1, 0], "m10": [1, 0], "m11": [1, 0]}}\n'
    )
    assert (tmp_path / "puzzle_info.json").read_text(encoding="utf-8") == expected
    assert (tmp_path / "test.csv").read_text(encoding="utf-8") == (
        'initial_state_id,initial_state\n0,"1,0"\n'
    )


def test_rust_writer_allows_python_fallback_for_older_extensions(monkeypatch, tmp_path):
    contract = GraphContract.from_graph(_twelve_move_graph(), [1, 0])
    monkeypatch.setattr(rust_control, "extension", lambda: SimpleNamespace())

    assert rust_control.write_search_inputs(contract, tmp_path) is False
    assert not (tmp_path / "puzzle_info.json").exists()
    assert not (tmp_path / "test.csv").exists()

