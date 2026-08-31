import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from cayleypy import CayleyGraph, PermutationGroups

from cayleypy_native import NativeModel, NativeOptions, prepare_native
from cayleypy_native.contracts import GraphContract
from cayleypy_native.errors import NativeBackendError, NativeUnavailable
from cayleypy_native.models import prepare_model
import cayleypy_native.preparation as preparation


def make_artifact(directory):
    directory.mkdir()
    manifest = dict(state_len=4, num_classes=4, hd1=16, hd2=8, nrd=1,
                    output_dim=1, dtype="fp16", normalization="batchnorm_folded")
    (directory / "manifest.json").write_text(json.dumps(manifest))
    counts = dict(input_weight_hxk=256, input_bias=16, hidden_weight_hxk=128,
                  hidden_bias=8, output_weight_hxk=8, output_bias=1)
    for fc in (1, 2):
        counts[f"residual0_fc{fc}_weight_hxk"] = 64
        counts[f"residual0_fc{fc}_bias"] = 8
    for name, count in counts.items():
        (directory / f"{name}.fp16").write_bytes(bytes(count * 2))
    return directory


@pytest.fixture
def configured(tmp_path, monkeypatch):
    graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu", random_seed=1729)
    weights = make_artifact(tmp_path / "source_weights")
    runner = tmp_path / "fake_production_runner"
    runner.write_bytes(b"unit-test fake binary; never executed")
    runner.chmod(runner.stat().st_mode | 0o111)
    metadata = {"schema_version": 1, "shape": {"state_len": 4, "storage_len": 16, "move_count": 3, "alignment": 16},
                "backend": "mlp", "cuda_architectures": [75], "build_key": "test-fixture",
                "binary_sha256": hashlib.sha256(runner.read_bytes()).hexdigest()}
    calls = []
    monkeypatch.setattr(preparation, "runtime_devices", lambda *args: (0, 1))

    def runtime(contract, model, options, run_dir, devices):
        calls.append(contract)
        assert contract.start == contract.center
        assert devices == (0, 1)
        return SimpleNamespace(runner=runner, build_metadata=metadata, architectures=(75,))

    monkeypatch.setattr(preparation, "prepare_runtime", runtime)
    options = NativeOptions(cache_dir=tmp_path / "cache", source_dir=tmp_path / "source",
                            cutlass_dir=tmp_path / "cutlass", devices=(0, 1))
    return graph, NativeModel.for_graph(graph, weights), options, calls


def test_prepare_creates_owned_pinned_snapshot_without_search_or_patch(configured):
    graph, model, options, calls = configured
    original = CayleyGraph.beam_search
    prepared = prepare_native(graph, model, native_options=options)
    assert CayleyGraph.beam_search is original
    assert len(calls) == 1
    assert prepared.model.weights_dir != model.weights_dir
    assert prepared.model.fallback is None
    assert prepared.options.runner_path.parent.parent == prepared.preparation_dir
    assert prepared.options.devices == (0, 1)
    assert prepared.options.source_dir is None and prepared.options.cutlass_dir is None
    assert prepared.model.expected_artifact_hash and len(prepared.runner_sha256) == 64
    assert (prepared.preparation_dir / "native-preparation.json").is_file()
    contract = GraphContract.from_graph(graph, [1, 0, 2, 3])
    # Modifying the caller's old weights cannot modify the prepared snapshot.
    (model.weights_dir / "output_bias.fp16").write_bytes(b"\x01\x00")
    observed = prepare_model(prepared.model, contract, prepared.options, prepared.preparation_dir / "unused")
    assert observed.artifact_hash == prepared.model.expected_artifact_hash


def test_same_size_snapshot_mutation_is_error_not_fallback(configured):
    graph, model, options, _ = configured
    prepared = prepare_native(graph, model, native_options=options)
    (prepared.model.weights_dir / "output_bias.fp16").write_bytes(b"\x01\x00")
    with pytest.raises(NativeBackendError, match="hash"):
        prepare_model(prepared.model, GraphContract.from_graph(graph, [1, 0, 2, 3]), prepared.options,
                      prepared.preparation_dir / "unused")


def test_fallback_is_only_retained_when_explicitly_passed(configured):
    graph, model, options, _ = configured
    fallback = object()
    prepared = prepare_native(graph, model, native_options=options, fallback=fallback)
    assert prepared.model.fallback is fallback
    assert model.fallback is None


def test_unknown_runtime_is_strict_and_does_not_export(configured, monkeypatch):
    graph, model, options, calls = configured
    def unavailable(*args):
        raise NativeUnavailable("fixture no CUDA")
    monkeypatch.setattr(preparation, "runtime_devices", unavailable)
    monkeypatch.setattr(preparation, "prepare_model", lambda *args: pytest.fail("export after runtime rejection"))
    with pytest.raises(NativeUnavailable, match="no CUDA"):
        prepare_native(graph, model, native_options=options, fallback=object())
    assert not calls
    assert not options.cache_dir.exists()


def test_loaded_model_export_runs_once_and_repeated_validation_uses_snapshot(configured, monkeypatch):
    graph, source_model, options, calls = configured
    loaded_predictor = object()
    original_prepare = preparation.prepare_model
    exported = []
    def prepare(predictor, contract, current_options, run_dir):
        if predictor is loaded_predictor:
            exported.append(predictor)
            # Labelled exporter boundary fixture; real exporter is covered by test_models.
            weights = make_artifact(run_dir / "weights")
            return original_prepare(NativeModel.for_graph(graph, weights), contract, current_options, run_dir)
        return original_prepare(predictor, contract, current_options, run_dir)
    monkeypatch.setattr(preparation, "prepare_model", prepare)
    prepared = prepare_native(graph, loaded_predictor, native_options=options)
    contract = GraphContract.from_graph(graph, [1, 0, 2, 3])
    for index in range(2):
        original_prepare(prepared.model, contract, prepared.options, prepared.preparation_dir / f"run{index}")
    assert len(exported) == 1 and len(calls) == 1
    assert prepared.model.fallback is None


def test_snapshot_binary_is_checked_after_copy(configured, monkeypatch):
    graph, model, options, _ = configured
    original_copy = preparation.shutil.copy2
    def corrupt_binary(source, destination, *args, **kwargs):
        result = original_copy(source, destination, *args, **kwargs)
        if "fake_production_runner" == Path(source).name:
            Path(destination).write_bytes(b"corrupted copied binary")
        return result
    monkeypatch.setattr(preparation.shutil, "copy2", corrupt_binary)
    with pytest.raises(NativeBackendError, match="SHA256"):
        prepare_native(graph, model, native_options=options)
    errors = list(options.cache_dir.glob("prepared/*/native-preparation-error.json"))
    assert len(errors) == 1


@pytest.mark.parametrize("pin", ["", "abc", "z" * 64, 3])
def test_artifact_hash_pin_must_be_sha256(tmp_path, pin):
    with pytest.raises(ValueError, match="expected_artifact_hash"):
        NativeModel(tmp_path, "graph", None, "mlp", pin)
