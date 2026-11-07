import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from validation.real_cases import (fixture_spec, load_inputs, make_cases, replay,
                                   run_real_acceptance, validate_outcome, verify_export)


def mounted_fixture(tmp_path):
    spec = fixture_spec()
    export = tmp_path / "kernel-output" / "cayleypy_public" / "output" / "export"
    export.mkdir(parents=True)
    (export / "manifest.json").write_text(json.dumps(spec["shape"]))
    (export / "output_bias.fp16").write_bytes(b"\x00\x00")
    spec["files"] = {p.name: {"bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                     for p in export.iterdir()}
    competition = tmp_path / "competition"
    competition.mkdir()
    generators = {}
    for i in range(12):
        for sign in (1, -1):
            # Deliberately nonlexical names and inverse-paired generators.
            generators[f"{'z' if sign == 1 else 'a'}{i}"] = [(j + sign * (i + 1)) % 88 for j in range(88)]
    center = list(range(88))
    (competition / "puzzle_info.json").write_text(json.dumps({"central_state": center, "generators": generators}))
    start = list(replay(center, (0, 2), list(generators.values())))
    (competition / "test.csv").write_text('initial_state_id,initial_state\n0,"' + ','.join(map(str, start)) + '"\n')
    return spec, export, competition, generators


def test_pinned_real_export_has_all_local_checksums():
    spec = fixture_spec()
    assert len(spec["files"]) == 23
    assert sum(v["bytes"] for k, v in spec["files"].items() if k.endswith(".fp16")) == 4295682
    assert spec["checkpoint_sha256"] == "10d0ccebd3bf83c8cb7bf50eb4960430acb1e463fbf24a8c32a91cda119528bf"
    root = Path(__file__).resolve().parents[3]
    export = root / ".worktrees/public-cayleypy-notebook/test_results/kaggle_public_tetraminx_v2/cayleypy_public/output/export"
    if export.is_dir():
        assert verify_export(export)["verified"]  # Actual 4.3MB local artifact, never embedded in tests.


def test_inputs_preserve_native_textual_order_and_real_state(tmp_path):
    spec, export, _, generators = mounted_fixture(tmp_path)
    definition, start, selected, evidence = load_inputs(tmp_path, spec)
    assert selected == export
    assert definition.generator_names == list(generators)
    assert definition.generators_permutations == list(generators.values())
    assert start == list(replay(range(88), (0, 2), list(generators.values())))
    assert evidence["export"]["verified"]
    assert len(evidence["contract"]["test_csv_sha256"]) == 64


@pytest.mark.parametrize("change", ["corrupt", "missing", "extra"])
def test_export_tampering_is_rejected_before_execution(tmp_path, change):
    spec, export, _, _ = mounted_fixture(tmp_path)
    if change == "corrupt":
        (export / "output_bias.fp16").write_bytes(b"\x01\x00")
    elif change == "missing":
        (export / "output_bias.fp16").unlink()
    else:
        (export / "unexpected.bf16").write_bytes(b"\x00\x00")
    with pytest.raises(ValueError, match="checksum|file set"):
        load_inputs(tmp_path, spec)


@pytest.mark.parametrize("change", ["duplicate_id", "bad_state", "missing_table", "duplicate_json_key"])
def test_competition_input_failures_are_not_silently_accepted(tmp_path, change):
    spec, _, competition, _ = mounted_fixture(tmp_path)
    table = competition / "test.csv"
    if change == "duplicate_id":
        table.write_text(table.read_text() + table.read_text().splitlines()[1] + "\n")
    elif change == "bad_state":
        table.write_text('initial_state_id,initial_state\n0,"0,0"\n')
    elif change == "missing_table":
        table.unlink()
    else:
        path = competition / "puzzle_info.json"
        path.write_text(path.read_text().replace('"central_state":', '"central_state": [], "central_state":'))
    with pytest.raises((ValueError, FileNotFoundError)):
        load_inputs(tmp_path, spec)


def test_ambiguous_real_contract_is_rejected(tmp_path):
    spec, _, competition, _ = mounted_fixture(tmp_path)
    second = tmp_path / "other-competition"
    second.mkdir()
    (second / "puzzle_info.json").write_bytes((competition / "puzzle_info.json").read_bytes())
    (second / "test.csv").write_text('initial_state_id,initial_state\n0,"' + ','.join(map(str, range(88))) + '"\n')
    with pytest.raises(ValueError, match="multiple different"):
        load_inputs(tmp_path, spec)


def test_real_and_shallow_cases_have_independent_reference_paths(tmp_path):
    spec, _, _, _ = mounted_fixture(tmp_path)
    definition, start, _, _ = load_inputs(tmp_path, spec)
    cases = make_cases(definition, start)
    assert cases[0]["start"] == start and not cases[0]["must_solve"]
    for case in cases[1:]:
        assert case["must_solve"]
        assert replay(case["start"], case["reference_solution"], definition.generators_permutations) == tuple(range(88))
        assert len(case["reference_solution"]) == case["max_steps"]


def outcome_fixture(tmp_path):
    from cayleypy_native.contracts import GraphContract
    spec, _, _, _ = mounted_fixture(tmp_path)
    definition, start, _, _ = load_inputs(tmp_path, spec)
    graph = SimpleNamespace(definition=definition)
    case = make_cases(definition, start)[1]
    log = tmp_path / "native.log"
    log.write_text("WORLD_SIZE=2\nLOCAL_RANK=0\nCUDA_DEVICE_LOCAL_RANK=0\nWORLD_SIZE=2\nLOCAL_RANK=1\nCUDA_DEVICE_LOCAL_RANK=1\n")
    metadata = {"devices": [0, 1], "requested_beam_width": 65536, "effective_beam_width": 65536,
        "model_artifact_hash": "model", "graph_hash": GraphContract.from_graph(graph, case["start"]).graph_hash,
        "log_path": str(log), "build": {"shape": {"state_len": 88, "storage_len": 96, "alignment": 16, "move_count": 24}}}
    return graph, case, metadata


def test_result_gate_replays_and_requires_real_rank_evidence(tmp_path):
    graph, case, metadata = outcome_fixture(tmp_path)
    record = validate_outcome(case, case["reference_solution"], metadata, graph, (0, 1), 65536, "model")
    assert record["replay_valid"] and record["rank_evidence"]["passed"]
    Path(metadata["log_path"]).write_text("WORLD_SIZE=2\nLOCAL_RANK=0\nCUDA_DEVICE_LOCAL_RANK=0\n")
    with pytest.raises(ValueError, match="rank/device"):
        validate_outcome(case, case["reference_solution"], metadata, graph, (0, 1), 65536, "model")


def test_result_gate_rejects_valid_path_exceeding_depth_budget(tmp_path):
    graph, case, metadata = outcome_fixture(tmp_path)
    case["max_steps"] = 0
    assert replay(case["start"], case["reference_solution"], graph.definition.generators_permutations) == tuple(graph.definition.central_state)
    with pytest.raises(ValueError, match="budget"):
        validate_outcome(case, case["reference_solution"], metadata, graph, (0, 1), 65536, "model")


@pytest.mark.parametrize("change", ["replay", "no_solution", "beam", "model", "graph", "devices", "shape"])
def test_result_gate_rejects_wrong_work_or_wrong_solution(tmp_path, change):
    graph, case, metadata = outcome_fixture(tmp_path)
    path = case["reference_solution"]
    if change == "replay":
        path = []
    elif change == "no_solution":
        path = None
    elif change == "beam":
        metadata["effective_beam_width"] = 32768
    elif change == "model":
        metadata["model_artifact_hash"] = "changed"
    elif change == "graph":
        metadata["graph_hash"] = "changed"
    elif change == "devices":
        metadata["devices"] = [1, 0]
    else:
        metadata["build"]["shape"]["state_len"] = 120
    with pytest.raises(ValueError):
        validate_outcome(case, path, metadata, graph, (0, 1), 65536, "model")


def test_cpu_failure_is_a_saved_report_without_native_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_real_acceptance(tmp_path, tmp_path, tmp_path, tmp_path / "out")
    assert report["status"] == "failed" and not report["passed"] and report["cases"] == []
    assert report["torch_baseline"] == "not_run"
    saved = json.loads((tmp_path / "out/real_acceptance_report.json").read_text())
    assert saved["exception"]["type"] == "RuntimeError"


@pytest.mark.parametrize("failure", [None, "worker", "validation"])
def test_orchestration_uses_same_work_and_stops_on_worker_failure(tmp_path, monkeypatch, failure):
    from cayleypy import CayleyGraph
    from cayleypy_native import backend, dispatch, models
    from cayleypy_native.options import NativeOutcome
    import validation.real_cases as module

    spec, _, _, _ = mounted_fixture(tmp_path)
    inputs = load_inputs(tmp_path, spec)
    monkeypatch.setattr(module, "load_inputs", lambda _: inputs)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "Tesla T4")
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _: (14 * 1024**3, 15 * 1024**3))
    monkeypatch.setattr(dispatch, "runtime_devices", lambda *_: (0, 1))
    prepared = SimpleNamespace(artifact_hash="model", manifest={"output_dim": 1}, backend="mlp")
    monkeypatch.setattr(models, "prepare_model", lambda *args: prepared)
    monkeypatch.setattr(dispatch, "prepare_model", lambda *args: prepared)
    preparation_options, invocations = [], []

    def prepare(contract, model, options, directory, devices):
        preparation_options.append(options)
        Path(directory).mkdir(parents=True, exist_ok=True)
        return SimpleNamespace()

    def execute(contract, model, options, beam, depth, directory, devices, *, runtime):
        invocations.append((contract, model, options, beam, depth, tuple(devices)))
        if failure == "worker" and len(invocations) == 2:
            raise RuntimeError("bounded fixture worker failure; must not retry")
        directory = Path(directory)
        log = directory / "native.log"
        log.write_text("WORLD_SIZE=2\nLOCAL_RANK=0\nCUDA_DEVICE_LOCAL_RANK=0\nWORLD_SIZE=2\nLOCAL_RANK=1\nCUDA_DEVICE_LOCAL_RANK=1\n")
        move = next(i for i in range(len(contract.generators)) if replay(contract.start, [i], contract.generators) == contract.center)
        metadata = {"model_artifact_hash": "model", "graph_hash": contract.graph_hash,
            "devices": list(devices), "requested_beam_width": beam, "log_path": str(log),
            "native_seconds": 0.005, "build": {"shape": {"state_len": 88, "storage_len": 96, "alignment": 16, "move_count": 24}}}
        return NativeOutcome((move,), 0.01, beam, directory, metadata)

    monkeypatch.setattr(backend, "prepare_runtime", prepare)
    monkeypatch.setattr(dispatch, "prepare_runtime", prepare)
    monkeypatch.setattr(backend, "run_native", execute)
    monkeypatch.setattr(dispatch, "run_native", execute)
    if failure == "validation":
        def reject_valid_outcome(*args):
            raise ValueError("fixture result validation failure")
        monkeypatch.setattr(module, "validate_outcome", reject_valid_outcome)
    original = CayleyGraph.beam_search
    report = run_real_acceptance(tmp_path, tmp_path, tmp_path, tmp_path / "out", warm_repeats=2)
    assert CayleyGraph.beam_search is original and dispatch._ORIGINAL is None
    assert len({id(row[2]) for row in invocations}) == 1
    assert all(row[1] is prepared and row[3] == 65536 and row[5] == (0, 1) for row in invocations)
    assert all(options is invocations[0][2] for options in preparation_options)
    if failure == "validation":
        assert not report["passed"] and len(invocations) == len(report["cases"]) == 1
        record = report["cases"][0]
        assert record["status"] == "failed"
        assert Path(record["native_metadata"]["log_path"]).is_file()
        saved = json.loads((tmp_path / "out/real_acceptance_report.json").read_text())
        assert saved["cases"][0]["native_metadata"] == record["native_metadata"]
        assert "validation failure" in report["exception"]["message"]
    elif failure == "worker":
        assert not report["passed"] and len(invocations) == len(report["cases"]) == 2
        assert report["cases"][-1]["status"] == "failed"
        assert "must not retry" in report["exception"]["message"]
    else:
        assert report["passed"] and len(invocations) == 8
        assert [r["route"] for r in report["cases"][:5]] == ["wrapper", "direct", "wrapper", "wrapper", "direct"]
        assert all(row["result"]["replay_valid"] for row in report["cases"])
        assert report["torch_baseline"] == "not_run"
