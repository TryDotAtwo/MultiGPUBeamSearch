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
        (path / f"{name}.{manifest['dtype']}").write_bytes(b"\x00" * (count * 2))
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


@pytest.mark.parametrize("dtype,nonfinite", [("fp16", b"\x00\x7c"), ("bf16", b"\x80\x7f")])
def test_nonfinite_native_weight_blob_is_rejected(tmp_path, contract, dtype, nonfinite):
    path = artifact(tmp_path / "bad", dtype=dtype)
    blob = path / f"input_bias.{dtype}"
    data = bytearray(blob.read_bytes())
    data[:2] = nonfinite
    blob.write_bytes(data)
    with pytest.raises(NativeBackendError, match="non-finite.*input_bias"):
        prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run")


def test_layernorm_artifact_requires_norm_blobs(tmp_path, contract):
    path = artifact(tmp_path / "ln", normalization="layernorm")
    prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run")
    (path / "residual0_fc2_ln_gamma.fp16").unlink()
    with pytest.raises(NativeBackendError):
        prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run2")


@pytest.mark.parametrize("variant", ["duplicate", "nested", "escaped"])
def test_manifest_runtime_keys_must_be_unique_literal_top_level_keys(tmp_path, contract, variant):
    path = artifact(tmp_path / "ambiguous", normalization="layernorm")
    manifest_path = path / "manifest.json"
    text = manifest_path.read_text(encoding="utf-8")
    if variant == "duplicate":
        text = '{"normalization":"none",' + text[1:]
    elif variant == "nested":
        text = '{"metadata":{"normalization":"none"},' + text[1:]
    else:
        text = text.replace('"normalization"', '"normali\\u007aation"')
    manifest_path.write_text(text, encoding="utf-8")
    with pytest.raises(NativeBackendError, match="manifest.*runtime key"):
        prepare_model(NativeModel(path, contract.graph_hash), contract, NativeOptions(), tmp_path / "run")


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


@pytest.mark.parametrize("inner_width", [4, 16])
def test_pilgrim_nonsquare_residual_is_unavailable_before_export(
        tmp_path, contract, monkeypatch, inner_width):
    import cayleypy_native.models as module

    model = Pilgrim()
    block = model.residual_blocks[0]
    block.fc1 = nn.Linear(8, inner_width)
    block.bn1 = nn.BatchNorm1d(inner_width)
    block.fc2 = nn.Linear(inner_width, 8)
    block.bn2 = nn.BatchNorm1d(8)
    model.eval()
    assert tuple(model(torch.tensor([[0, 1, 2, 3]])).shape) == (1,)
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: pytest.fail("unsupported residual shape launched exporter"))
    with pytest.raises(NativeUnavailable, match="residual.*dimensions"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


@pytest.mark.parametrize("hidden", ["first", "second"])
def test_pilgrim_unaligned_hidden_width_is_unavailable_before_export(
        tmp_path, contract, monkeypatch, hidden):
    import cayleypy_native.models as module

    model = Pilgrim()
    if hidden == "first":
        model.input_layer = nn.Linear(16, 12)
        model.bn1 = nn.BatchNorm1d(12)
        model.hidden_layer = nn.Linear(12, 8)
    else:
        model.hidden_layer = nn.Linear(16, 10)
        model.bn2 = nn.BatchNorm1d(10)
        model.residual_blocks = nn.ModuleList([ResidualBlock()])
        block = model.residual_blocks[0]
        block.fc1, block.fc2 = nn.Linear(10, 10), nn.Linear(10, 10)
        block.bn1, block.bn2 = nn.BatchNorm1d(10), nn.BatchNorm1d(10)
        model.output_layer = nn.Linear(10, 1)
    model.eval()
    assert tuple(model(torch.tensor([[0, 1, 2, 3]])).shape) == (1,)
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: pytest.fail("unaligned hidden width launched exporter"))
    with pytest.raises(NativeUnavailable, match="multiples of 8"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_excessive_residual_count_is_unavailable_before_export(tmp_path, contract, monkeypatch):
    import cayleypy_native.models as module

    model = Pilgrim()
    model.residual_blocks = nn.ModuleList([ResidualBlock() for _ in range(1025)])
    model.eval()
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: pytest.fail("excessive residual count launched exporter"))
    with pytest.raises(NativeUnavailable, match="at most 1024"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


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


def test_class_level_call_override_cannot_hide_beyond_finite_probes(tmp_path, contract):
    class ClassCallOverride(Pilgrim):
        def __call__(self, states):
            score = super().__call__(states)
            unseen = torch.tensor([2, 1, 0, 3], device=states.device)
            return score + (states == unseen).all(dim=1).to(score.dtype)

    ClassCallOverride.__name__ = "Pilgrim"
    model = ClassCallOverride().eval()
    with pytest.raises(NativeUnavailable, match="class-level __call__"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


@pytest.mark.parametrize("method", ["__getattribute__", "__getattr__"])
def test_custom_model_attribute_dispatch_is_rejected(tmp_path, contract, method):
    if method == "__getattribute__":
        class AttributeOverride(Pilgrim):
            def __getattribute__(self, name):
                return super().__getattribute__(name)
    else:
        class AttributeOverride(Pilgrim):
            def __getattr__(self, name):
                return super().__getattr__(name)

    AttributeOverride.__name__ = "Pilgrim"
    with pytest.raises(NativeUnavailable, match="call dispatch"):
        prepare_model(AttributeOverride().eval(), contract,
                      NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


@pytest.mark.parametrize("method", ["modules", "named_modules"])
def test_custom_model_traversal_is_rejected(tmp_path, contract, monkeypatch, method):
    import cayleypy_native.models as module

    if method == "modules":
        class TraversalOverride(Pilgrim):
            def modules(self):
                return super().modules()
    else:
        class TraversalOverride(Pilgrim):
            def named_modules(self, *args, **kwargs):
                return super().named_modules(*args, **kwargs)

    TraversalOverride.__name__ = "Pilgrim"
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: pytest.fail("custom module traversal launched exporter"))
    with pytest.raises(NativeUnavailable, match="module traversal"):
        prepare_model(TraversalOverride().eval(), contract,
                      NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


@pytest.mark.parametrize("override", ["class", "instance"])
def test_residual_block_call_dispatch_is_rejected(tmp_path, contract, monkeypatch, override):
    import cayleypy_native.models as module

    model = Pilgrim()
    if override == "class":
        class DispatchBlock(ResidualBlock):
            def _call_impl(self, features):
                return super()._call_impl(features)

        model.residual_blocks[0] = DispatchBlock()
    else:
        block = model.residual_blocks[0]
        original = block._call_impl
        block._call_impl = lambda features: original(features)
    model.eval()
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: pytest.fail("custom child dispatch launched exporter"))
    with pytest.raises(NativeUnavailable, match="call dispatch"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


@pytest.mark.parametrize("override", ["class", "instance", "hook", "child"])
def test_custom_state_serialization_cannot_hide_beyond_finite_probes(
        tmp_path, contract, monkeypatch, override):
    import cayleypy_native.models as module

    probes = [contract.center, contract.start]
    probes.extend(tuple(contract.start[j] for j in generator) for generator in contract.generators[:8])
    used_columns = {position * 4 + value for state in probes for position, value in enumerate(state)}
    unseen_column = next(column for column in range(16) if column not in used_columns)

    def changed(state):
        weight = state["input_layer.weight"].clone()
        weight[:, unseen_column] += 1.0
        state["input_layer.weight"] = weight
        return state

    if override == "class":
        class StateOverride(Pilgrim):
            def state_dict(self, *args, **kwargs):
                return changed(super().state_dict(*args, **kwargs))

        StateOverride.__name__ = "Pilgrim"
        model = StateOverride()
    elif override != "child":
        model = Pilgrim()
        if override == "instance":
            original = model.state_dict
            model.state_dict = lambda *args, **kwargs: changed(original(*args, **kwargs))
        else:
            def alter_state(_module, state, _prefix, _metadata):
                changed(state)

            model.register_state_dict_post_hook(alter_state)
    else:
        class StateBlock(ResidualBlock):
            def state_dict(self, *args, **kwargs):
                return super().state_dict(*args, **kwargs)

        model = Pilgrim()
        model.residual_blocks[0] = StateBlock()
    model.eval()
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: pytest.fail("custom serialization launched exporter"))
    with pytest.raises(NativeUnavailable, match="state serialization"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


@pytest.mark.parametrize("override", ["class", "instance", "compiled"])
def test_call_impl_override_cannot_hide_beyond_finite_probes(tmp_path, contract, override):
    def changed(score, states):
        unseen = torch.tensor([2, 1, 0, 3], device=states.device)
        return score + (states == unseen).all(dim=1).to(score.dtype)

    if override == "class":
        class CallImplOverride(Pilgrim):
            def _call_impl(self, states):
                return changed(super()._call_impl(states), states)

        CallImplOverride.__name__ = "Pilgrim"
        model = CallImplOverride().eval()
    else:
        model = Pilgrim().eval()
        original = model._call_impl
        replacement = lambda states: changed(original(states), states)
        if override == "instance":
            model._call_impl = replacement
        else:
            model._compiled_call_impl = replacement
    with pytest.raises(NativeUnavailable, match="call dispatch"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_custom_child_forward_cannot_hide_beyond_finite_probes(tmp_path, contract):
    class ConditionalLinear(nn.Linear):
        def forward(self, features):
            output = super().forward(features)
            unseen = features[:, 2] * features[:, 5] * features[:, 8] * features[:, 15]
            return output + unseen.unsqueeze(1)

    model = Pilgrim().eval()
    custom = ConditionalLinear(16, 16)
    custom.load_state_dict(model.input_layer.state_dict())
    model.input_layer = custom
    model.eval()
    with pytest.raises(NativeUnavailable, match="child module"):
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


@pytest.mark.parametrize("inner_width", [4, 16])
def test_resmlp_nonsquare_residual_is_unavailable_before_export(
        tmp_path, contract, monkeypatch, inner_width):
    import cayleypy_native.models as module

    model = ResMLPDistance()
    block = model.res_blocks[0]
    block.lin1 = nn.Linear(8, inner_width)
    block.ln1 = nn.LayerNorm(inner_width)
    block.lin2 = nn.Linear(inner_width, 8)
    block.ln2 = nn.LayerNorm(8)
    model.eval()
    assert tuple(model(torch.tensor([[0, 1, 2, 3]])).shape) == (1, 3)
    monkeypatch.setattr(module.subprocess, "run",
                        lambda *a, **k: pytest.fail("unsupported residual shape launched exporter"))
    with pytest.raises(NativeUnavailable, match="residual.*dimensions"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_resmlp_bias_free_first_linear_exports_zero_bias(tmp_path, contract):
    model = ResMLPDistance()
    first = nn.Linear(64, 16, bias=False)
    with torch.no_grad():
        first.weight.copy_(model.input_stack[0].weight)
    model.input_stack[0] = first
    prepared = prepare_model(model.eval(), contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")
    assert (prepared.weights_dir / "input_bias.fp16").read_bytes() == b"\x00" * (16 * 2)


def test_fp16_conversion_overflow_is_rejected_after_export(tmp_path, contract):
    model = Pilgrim().eval()
    with torch.no_grad():
        model.input_layer.weight[0, 0] = 70000.0
    with pytest.raises(NativeBackendError, match="non-finite.*input_weight"):
        prepare_model(model, contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")


def test_export_process_failure_cannot_fall_back(tmp_path, contract, monkeypatch):
    from types import SimpleNamespace
    import cayleypy_native.models as module
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=9, stdout="export failed"))
    with pytest.raises(NativeBackendError, match="exit code 9"):
        prepare_model(Pilgrim().eval(), contract, NativeOptions(source_dir=SOURCE_ROOT), tmp_path / "run")
    assert (tmp_path / "run" / "model_export.log").read_text() == "export failed"
