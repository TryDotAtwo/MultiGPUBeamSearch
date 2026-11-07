import csv
import hashlib
import json
import zipfile

import pytest

import cayleypy as cp
import multigpubeamsearch as beam


def tiny_graph():
    definition = cp.CayleyGraphDef.create(
        [[1, 0]], generator_names=["swap"], central_state=[0, 1]
    )
    return cp.CayleyGraph(definition, device="cpu")


def test_load_graph_accepts_a_cayleypy_definition_dict(tmp_path):
    graph_info = {
        "generators": {"swap": [1, 0]},
        "central_state": [0, 1],
    }

    graph = beam.load_graph(graph_info, device="cpu")

    assert graph.definition.generator_names == ["swap"]
    assert list(graph.definition.central_state) == [0, 1]


def test_read_states_and_named_reflection_paths_use_graph_moves(tmp_path):
    graph = tiny_graph()
    states = tmp_path / "states.csv"
    states.write_text("id,state\ncase-a,1;0\n", encoding="utf-8")
    paths = tmp_path / "paths.csv"
    paths.write_text("id,solution\ncase-a,swap\n", encoding="utf-8")

    start_states = beam.read_states(states, id_column="id", state_column="state", separator=";")
    reflection_paths = beam.read_paths(paths, graph, id_column="id", path_column="solution")

    assert start_states == {"case-a": [1, 0]}
    assert reflection_paths == {"case-a": (0,)}
    assert beam.replay_path(graph, start_states["case-a"], reflection_paths["case-a"])


def test_read_states_rejects_duplicate_ids(tmp_path):
    states = tmp_path / "states.csv"
    states.write_text("id,state\na,1,0\na,0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate"):
        beam.read_states(states, id_column="id", state_column="state")


def test_download_bundle_checks_manifest_before_returning_inputs(tmp_path):
    contents = b'{"generators":{"swap":[1,0]},"central_state":[0,1]}'
    manifest = {"graph.json": {"bytes": len(contents), "sha256": hashlib.sha256(contents).hexdigest()}}
    archive = tmp_path / "bundle.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("manifest.json", json.dumps(manifest))
        zf.writestr("graph.json", contents)

    extracted = beam.download_bundle(archive.as_uri(), tmp_path / "cache")

    assert (extracted / "graph.json").read_bytes() == contents
    assert beam.load_graph(extracted / "graph.json", device="cpu").definition.generator_names == ["swap"]


def test_save_result_replays_and_persists_a_generic_graph(tmp_path):
    graph = tiny_graph()
    result = type("Result", (), {"path_found": True, "path": ["swap"], "solutions": (["swap"],), "native_metadata": {"backend": "native"}})()

    saved = beam.save_result(result, graph, [1, 0], tmp_path, metadata={"problem": "tiny"})
    record = json.loads(saved.read_text(encoding="utf-8"))

    assert record["path"] == [0]
    assert record["graph_hash"]
    assert record["metadata"] == {"problem": "tiny"}
    assert record["replay_valid"] is True


def test_save_result_rejects_a_path_marked_not_found(tmp_path):
    graph = tiny_graph()
    result = type("Result", (), {"path_found": False, "path": [0], "native_metadata": {}})()

    with pytest.raises(ValueError, match="path_found is false"):
        beam.save_result(result, graph, [1, 0], tmp_path)

    assert not list(tmp_path.iterdir())
