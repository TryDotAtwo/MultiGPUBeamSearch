"""CayleyPy and the actual Rust session must agree on graph identity."""
import json

import pytest
from cayleypy import CayleyGraph, CayleyGraphDef

from multigpubeamsearch.contracts import GraphContract


@pytest.mark.parametrize("center", [None, [0, 0, 1, 1]])
def test_cayleypy_contract_is_accepted_and_stale_contents_are_rejected(center, tmp_path):
    runtime = pytest.importorskip("_beam_runtime")
    definition = CayleyGraphDef.create(
        [[1, 2, 3, 0], [3, 0, 1, 2]],
        central_state=center,
        generator_names=['quoted "name"', "second\nname"],
    )
    graph = CayleyGraph(definition, device="cpu")
    contract = GraphContract.from_graph(graph, graph.central_state)
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
