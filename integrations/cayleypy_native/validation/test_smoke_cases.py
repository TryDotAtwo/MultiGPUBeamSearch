import json
import os
from pathlib import Path

import pytest
import torch
from cayleypy import CayleyGraph, PermutationGroups, Predictor

from validation.smoke_cases import (make_cases, make_model, rank_evidence,
                                    replay, run_acceptance, run_case, exported_weights_evidence)


def test_cases_have_exact_distances_and_no_beam_truncation():
    definition = PermutationGroups.lrx(8)
    cases, oracle = make_cases(definition)
    assert [case.name for case in cases] == ["already_goal", "one_step", "multi_step",
        "zero_budget", "not_found", "multi_step_warm"]
    assert oracle["exact_layer_sizes"] == [1, 3, 7, 15]
    assert all(case.beam_width >= max(oracle["exact_layer_sizes"][:case.max_steps], default=1) for case in cases)
    for case in cases:
        assert replay(case.start, case.reference_solution, definition.generators) == tuple(definition.central_state)
        assert len(case.reference_solution) == case.exact_distance
        assert case.expected_found == (case.exact_distance <= case.max_steps)


def test_model_is_deterministic_without_mutating_rng_and_cpu_baseline_scores():
    before = torch.random.get_rng_state().clone()
    model = make_model()
    assert torch.equal(before, torch.random.get_rng_state())
    second = make_model()
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in model.state_dict().items())
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu", random_seed=1729)
    case = make_cases(graph.definition)[0][2]
    predictor = Predictor(graph, model)
    result = graph.beam_search(start_state=list(case.start), predictor=predictor,
                              beam_width=case.beam_width, max_steps=case.max_steps, return_path=True)
    assert result.path_found and result.path_length == 3
    assert replay(case.start, result.path, graph.definition.generators) == tuple(graph.definition.central_state)
    assert result.debug_scores  # beam==7 causes real scoring, but retains every depth-2 state.


def test_rank_evidence_requires_workers_not_only_requested_devices():
    text = "WORLD_SIZE=2\nLOCAL_RANK=0\nCUDA_DEVICE_LOCAL_RANK=0\nWORLD_SIZE=2\nLOCAL_RANK=1\nCUDA_DEVICE_LOCAL_RANK=1\n"
    evidence = rank_evidence(text, (0, 1))
    assert evidence["passed"]
    assert evidence["observed_rank_ids"] == [0, 1]
    assert not rank_evidence(text.replace("LOCAL_RANK=1\n", ""), (0, 1))["passed"]
    assert not rank_evidence("requested devices=(0,1) WORLD_SIZE=2", (0, 1))["passed"]
    assert not rank_evidence(text.replace("WORLD_SIZE=2", "WORLD_SIZE=1"), (0, 1))["passed"]


def test_bad_replay_rejects_noninteger_negative_and_out_of_range_moves():
    generators = PermutationGroups.lrx(8).generators
    for path in [[True], [-1], [3], [0.0]]:
        with pytest.raises(ValueError):
            replay(tuple(range(8)), path, generators)


def test_smoke_model_uses_real_cpu_exporter(tmp_path):
    from cayleypy_native.contracts import GraphContract
    from cayleypy_native.models import prepare_model
    from cayleypy_native.options import NativeOptions

    source_dir = Path(os.environ.get("CAYLEYPY_NATIVE_TEST_SOURCE_DIR", Path(__file__).resolve().parents[3]))
    if not (source_dir / "tools" / "export_stream1_mlp.py").is_file():
        pytest.skip("native source exporter is not adjacent to this staged unit-test payload")
    graph = CayleyGraph(PermutationGroups.lrx(8), device="cpu", random_seed=1729)
    contract = GraphContract.from_graph(graph, list(range(8)))
    prepared = prepare_model(Predictor(graph, make_model()), contract,
                             NativeOptions(source_dir=source_dir), tmp_path / "export")
    assert prepared.manifest["hd1"] == 32 and prepared.manifest["hd2"] == 16
    assert prepared.manifest["output_dim"] == 1 and prepared.manifest["dtype"] == "fp16"
    assert prepared.manifest["graph_hash"] == contract.graph_hash
    assert (tmp_path / "export" / "model_export.log").is_file()


def test_numerical_model_identity_ignores_only_export_provenance(tmp_path):
    manifest = {"dtype": "fp16", "hd1": 32, "source_weights": "/tmp/run1/checkpoint.pt"}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    blob = tmp_path / "output_bias.fp16"
    blob.write_bytes(b"\x00\x01")
    original = exported_weights_evidence(tmp_path)["numerical_model_sha256"]
    manifest["source_weights"] = "/tmp/run2/checkpoint.pt"
    path.write_text(json.dumps(manifest))
    assert exported_weights_evidence(tmp_path)["numerical_model_sha256"] == original
    blob.write_bytes(b"\x00\x02")
    assert exported_weights_evidence(tmp_path)["numerical_model_sha256"] != original


def test_public_search_exception_is_preserved_without_fallback(tmp_path):
    class FailedGraph:
        definition = PermutationGroups.lrx(8)

        def beam_search(self, **kwargs):
            assert kwargs["backend"] == "native"
            raise RuntimeError("fixture failure; log=/tmp/worker.log")

    case = make_cases(FailedGraph.definition)[0][1]
    record = run_case(FailedGraph(), object(), case, backend="native", devices=(0, 1),
                      cache_dir=tmp_path, synchronize=lambda: None)
    assert record["status"] == "failed"
    assert record["exception"]["type"] == "RuntimeError"
    assert "/tmp/worker.log" in record["exception"]["message"]
    assert "Traceback" in record["exception"]["traceback"]
    assert record["wall_seconds"] >= 0
    json.dumps(record, allow_nan=False)


def test_cpu_preflight_writes_failure_report_without_claiming_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_acceptance(source_dir=tmp_path, cutlass_dir=tmp_path, output_dir=tmp_path / "out")
    assert not report["passed"]
    assert report["status"] == "failed"
    assert report["cases"] == []
    saved = json.loads((tmp_path / "out" / "acceptance_report.json").read_text())
    assert saved["exception"]["type"] == "RuntimeError"
