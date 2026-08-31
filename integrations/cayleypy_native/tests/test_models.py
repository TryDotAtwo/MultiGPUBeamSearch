import json
import os
from pathlib import Path

import pytest
import torch
from torch import nn
from cayleypy import CayleyGraph, PermutationGroups, Predictor, ModelConfig

from cayleypy_native.contracts import GraphContract
from cayleypy_native.errors import NativeBackendError, NativeUnavailable
from cayleypy_native.models import NativeModel, prepare_model
from cayleypy_native.options import NativeOptions

SOURCE_ROOT = Path(os.environ.get("CAYLEYPY_NATIVE_TEST_SOURCE_DIR", Path(__file__).resolve().parents[3]))


@pytest.fixture
def contract():
    g = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    return GraphContract.from_graph(g, [1, 0, 2, 3])


def artifact(path, *, normalization="batchnorm_folded", classes=4, **overrides):
    path.mkdir()
    manifest = dict(state_len=4, num_classes=classes, hd1=16, hd2=8, nrd=1,
                    output_dim=1, dtype="fp16", normalization=normalization)
    manifest.update(overrides)
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    sizes = dict(input_weight_hxk=4 * classes * 16, input_bias=16,
                 hidden_weight_hxk=16 * 8, hidden_bias=8,
                 output_weight_hxk=8, output_bias=1)
    for fc in (1, 2):
        sizes[f"residual0_fc{fc}_weight_hxk"] = 8 * 8
        sizes[f"residual0_fc{fc}_bias"] = 8
    if normalization == "layernorm":
        for prefix, width in [("input", 16), ("hidden", 8), ("residual0_fc1", 8), ("residual0_fc2", 8)]:
            sizes[f"{prefix}_ln_gamma"] = width
            sizes[f"{prefix}_ln_beta"] = width
    for name, count in sizes.items():
        (path / f"{name}.fp16").write_bytes(b"\x00" * (count * 2))
    return path


def test_valid_artifact_bound_to_graph_and_content_hash(tmp_path, contract):
    weights = artifact(tmp_path / "weights")
    model = NativeModel(weights, contract.graph_hash, fallback=object())
    prepared = prepare_model(model, contract, NativeOptions(), tmp_path / "run")
    assert prepared.backend == "mlp"
    assert prepared.weights_dir == weights.resolve()
    assert prepared.manifest["state_len"] == 4
    before = prepared.artifact_hash
    (weights / "output_bias.fp16").write_bytes(b"\x01\x00")
    assert prepare_model(model, contract, NativeOptions(), tmp_path / "run2").artifact_hash != before


def test_artifact_graph_mismatch_is_prelaunch_unavailable(tmp_path, contract):
    with pytest.raises(NativeUnavailable, match="graph"):
        prepare_model(NativeModel(tmp_path, "different"), contract, NativeOptions(), tmp_path / "run")


def test_for_graph_explicitly_declares_binding_without_loading_weights(tmp_path, contract):
    graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    fallback = object()
    model = NativeModel.for_graph(graph, tmp_path / "not-yet-exported", fallback=fallback)
    assert model.graph_hash == contract.graph_hash
    assert model.weights_dir == (tmp_path / "not-yet-exported").resolve()
    assert model.fallback is fallback and model.backend == "mlp"
    assert not model.weights_dir.exists()


@pytest.mark.parametrize("changes", [dict(dtype="fp32"), dict(state_len=5), dict(num_classes=3),
    dict(output_dim=2), dict(hd1=8, hd2=16), dict(hd2=7), dict(nrd=0),
    dict(hd1=True), dict(normalization="batchnorm"), dict(nrd=-1)])
def test_corrupt_declared_manifest_is_backend_error(tmp_path, contract, changes):
    path = artifact(tmp_path / "bad", **changes)
    with pytest.raises(NativeBackendError):
        prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run")


@pytest.mark.parametrize("name", ["hidden_weight_hxk.fp16", "output_bias.fp16", "manifest.json"])
def test_missing_declared_blob_is_backend_error(tmp_path, contract, name):
    path = artifact(tmp_path / "bad")
    (path / name).unlink()
    with pytest.raises(NativeBackendError):
        prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run")


def test_short_blob_is_backend_error(tmp_path, contract):
    path = artifact(tmp_path / "bad")
    (path / "input_bias.fp16").write_bytes(b"\x00")
    with pytest.raises(NativeBackendError, match="input_bias"):
        prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run")


def test_layernorm_artifact_requires_norm_blobs(tmp_path, contract):
    path = artifact(tmp_path / "ln", normalization="layernorm")
    prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run")
    (path / "residual0_fc2_ln_gamma.fp16").unlink()
    with pytest.raises(NativeBackendError):
        prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run2")


@pytest.mark.parametrize("model", [None, lambda x: x, ModelConfig("MLP", 4, 4, [16, 8]).build_model()])
def test_unknown_predictors_do_not_launch_export(tmp_path, contract, monkeypatch, model):
    import cayleypy_native.models as module
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("unsupported predictor launched process"))
    with pytest.raises(NativeUnavailable):
        prepare_model(model, contract, NativeOptions(), tmp_path / "run")


class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(8, 8), nn.Linear(8, 8)
        self.bn1, self.bn2 = nn.BatchNorm1d(8), nn.BatchNorm1d(8)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0)

    def forward(self, x):
        return self.relu(x + self.bn2(self.fc2(self.dropout(self.relu(self.bn1(self.fc1(x)))))))


class Pilgrim(nn.Module):
    def __init__(self):
        super().__init__()
        self.state_size, self.num_classes, self.z_add, self.output_dim = 4, 4, 0, 1
        self.input_layer, self.hidden_layer, self.output_layer = nn.Linear(16, 16), nn.Linear(16, 8), nn.Linear(8, 1)
        self.bn1, self.bn2 = nn.BatchNorm1d(16), nn.BatchNorm1d(8)
        self.relu, self.dropout = nn.ReLU(), nn.Dropout(0)
        self.residual_blocks = nn.ModuleList([ResidualBlock()])

    def forward(self, states):
        x = nn.functional.one_hot(states.long(), self.num_classes).float().flatten(1)
        x = self.relu(self.bn1(self.input_layer(x)))
        x = self.relu(self.bn2(self.hidden_layer(x)))
        for block in self.residual_blocks:
            x = block(x)
        return self.output_layer(x).squeeze(-1)


@pytest.mark.parametrize("customization", ["subclass_call", "subclass_children", "__call__", "score_children", "predict_batched", "_predict_as_tensor", "foreign_wrapper"])
def test_custom_predictor_scoring_is_not_bypassed(tmp_path, contract, monkeypatch, customization):
    from types import SimpleNamespace
    import cayleypy_native.models as module
    graph = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    if customization == "subclass_call":
        class CustomPredictor(Predictor):
            def __call__(self, states):
                return super().__call__(states) + 7
        predictor = CustomPredictor(graph, Pilgrim())
    elif customization == "subclass_children":
        class CustomPredictor(Predictor):
            def score_children(self, states):
                return super().score_children(states) * -1
        predictor = CustomPredictor(graph, Pilgrim())
    elif customization == "foreign_wrapper":
        predictor = SimpleNamespace(graph=graph, predict=Pilgrim().eval())
    else:
        predictor = Predictor(graph, Pilgrim())
        setattr(predictor, customization, lambda states: torch.zeros(len(states)))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("custom predictor launched exporter"))
    with pytest.raises(NativeUnavailable, match="Predictor"):
        prepare_model(predictor, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_predictor_graph_fingerprint_must_match(tmp_path, contract, monkeypatch):
    import cayleypy_native.models as module
    graph = CayleyGraph(PermutationGroups.lrx(4).with_central_state([1, 0, 2, 3]), device="cpu")
    predictor = Predictor(graph, Pilgrim())
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("mismatched predictor launched exporter"))
    with pytest.raises(NativeUnavailable, match="graph"):
        prepare_model(predictor, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_predictor_class_monkeypatch_is_not_bypassed(tmp_path, contract, monkeypatch):
    import cayleypy_native.models as module
    predictor = Predictor(CayleyGraph(PermutationGroups.lrx(4), device="cpu"), Pilgrim())
    monkeypatch.setattr(Predictor, "score_children", lambda self, states: torch.zeros(len(states), 3))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("modified predictor class launched exporter"))
    with pytest.raises(NativeUnavailable, match="Predictor"):
        prepare_model(predictor, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_older_predictor_surface_imports_and_unwraps_scalar(monkeypatch, contract):
    """PyPI 0.1 lacks child/batched helper methods and n_outputs metadata."""
    import importlib.util
    import sys
    import cayleypy_native.models as current_models
    for method in ("score_children", "predict_batched", "_predict_as_tensor"):
        monkeypatch.delattr(Predictor, method)
    name = "cayleypy_native._old_surface_test_models"
    spec = importlib.util.spec_from_file_location(name, current_models.__file__)
    old_surface_models = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, old_surface_models)
    spec.loader.exec_module(old_surface_models)
    model = Pilgrim().eval()
    predictor = Predictor(CayleyGraph(PermutationGroups.lrx(4), device="cpu"), model)
    del predictor.n_outputs
    assert old_surface_models._unwrap_predictor(predictor, contract) is model


def test_known_batchnorm_model_exports_real_tensor_checkpoint(tmp_path, contract):
    source = SOURCE_ROOT
    model = Pilgrim().eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    g = CayleyGraph(PermutationGroups.lrx(4), device="cpu")
    prepared = prepare_model(Predictor(g, model), contract, NativeOptions(source_dir=source), tmp_path / "run")
    assert prepared.manifest["normalization"] == "batchnorm_folded"
    assert prepared.manifest["graph_hash"] == contract.graph_hash
    checkpoint = torch.load(tmp_path / "run" / "model_state.pt", weights_only=True)
    assert checkpoint and all(isinstance(value, torch.Tensor) for value in checkpoint.values())
    assert not model.training
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("mutation", ["eps", "shift", "training", "forward", "activation"])
def test_similar_but_semantically_different_model_rejected(tmp_path, contract, mutation):
    model = Pilgrim().eval()
    if mutation == "eps":
        model.bn1.eps = 1e-3
    elif mutation == "shift":
        model.z_add = 1
    elif mutation == "training":
        model.train()
    elif mutation == "activation":
        model.relu = nn.SiLU()
    else:
        model.forward = lambda states: torch.zeros(states.shape[0])
    with pytest.raises(NativeUnavailable):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_class_level_forward_override_cannot_hide_beyond_finite_probes(tmp_path, contract):
    class ClassForwardOverride(Pilgrim):
        def forward(self, states):
            score = super().forward(states)
            unseen = torch.tensor([2, 1, 0, 3], device=states.device)
            return score + (states == unseen).all(dim=1).to(score.dtype)

    # The adapter selects supported schemas by their public class names. This
    # class deliberately preserves that name while changing deeper-state scores.
    ClassForwardOverride.__name__ = "Pilgrim"
    model = ClassForwardOverride().eval()
    with pytest.raises(NativeUnavailable, match="class-level forward"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


class LayerNormBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1, self.lin2 = nn.Linear(8, 8), nn.Linear(8, 8)
        self.ln1, self.ln2 = nn.LayerNorm(8), nn.LayerNorm(8)

    def forward(self, x):
        return torch.relu(x + self.ln2(self.lin2(torch.relu(self.ln1(self.lin1(x))))))


class ResMLPDistance(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(4, 16)
        self.input_stack = nn.Sequential(nn.Linear(64, 16), nn.LayerNorm(16), nn.ReLU(),
                                         nn.Linear(16, 8), nn.LayerNorm(8), nn.ReLU())
        self.res_blocks = nn.ModuleList([LayerNormBlock()])
        self.head = nn.Linear(8, 3)

    def forward(self, states):
        x = self.input_stack(self.embedding(states.long()).flatten(1))
        for block in self.res_blocks:
            x = block(x)
        return self.head(x)


def test_known_resmlp_q_model_exports_layernorm_and_folded_embedding(tmp_path, contract):
    model = ResMLPDistance().eval()
    prepared = prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")
    assert prepared.manifest["output_dim"] == contract.move_count == 3
    assert prepared.manifest["normalization"] == "layernorm"
    assert (prepared.weights_dir / "input_ln_gamma.fp16").stat().st_size == 32
    assert (prepared.weights_dir / "input_weight_hxk.fp16").stat().st_size == 4 * 4 * 16 * 2
    assert (prepared.weights_dir / "output_weight_hxk.fp16").stat().st_size == 8 * 3 * 2


def test_export_process_failure_cannot_fall_back(tmp_path, contract, monkeypatch):
    from types import SimpleNamespace
    import cayleypy_native.models as module
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=9, stdout="export failed"))
    with pytest.raises(NativeBackendError, match="exit code 9"):
        prepare_model(Pilgrim().eval(), contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")
    assert (tmp_path / "run" / "model_export.log").read_text() == "export failed"
