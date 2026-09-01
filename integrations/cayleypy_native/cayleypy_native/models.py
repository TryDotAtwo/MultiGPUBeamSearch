"""Validated native artifacts and conservative export of known inference schemas."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid

from cayleypy import Predictor

from .contracts import GraphContract
from .errors import NativeBackendError, NativeUnavailable
from .options import NativeOptions


_PREDICTOR_METHODS = {name: getattr(Predictor, name, None) for name in (
    "__init__", "__call__", "score_children", "predict_batched", "_predict_as_tensor",
)}
_NATIVE_RUNTIME_MANIFEST_KEYS = frozenset({
    "state_len", "num_classes", "hidden1", "hd1", "hidden2", "hd2",
    "residual_count", "nrd", "output_dim", "dtype", "normalization",
})


class _ObjectPairs(list):
    """JSON object representation that preserves duplicate keys for validation."""


@dataclass(frozen=True)
class NativeModel:
    weights_dir: Path
    graph_hash: str
    fallback: object = None
    backend: str = "mlp"
    expected_artifact_hash: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "weights_dir", Path(self.weights_dir).expanduser().resolve())
        if self.expected_artifact_hash is not None and (
            not isinstance(self.expected_artifact_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.expected_artifact_hash) is None
        ):
            raise ValueError("expected_artifact_hash must be a lowercase SHA256 hex digest")

    @classmethod
    def for_graph(cls, graph, weights_dir, *, fallback=None, backend="mlp") -> "NativeModel":
        """Declare that an exported artifact belongs to this ordered graph and center.

        This is a caller declaration, not proof of the graph used for training.
        It neither loads nor exports weights; prepare_model validates the artifact.
        """
        contract = GraphContract.from_graph(graph, graph.definition.central_state)
        return cls(weights_dir, contract.graph_hash, fallback, backend)


@dataclass(frozen=True)
class PreparedModel:
    weights_dir: Path
    manifest: dict
    backend: str
    artifact_hash: str
    artifact_files: tuple[str, ...]


def verify_prepared_model(model: PreparedModel, contract: GraphContract) -> None:
    """Fail if any manifest/blob byte changed after preparation."""
    current = _validate_artifact(model.weights_dir, contract, model.backend)
    if current.artifact_hash != model.artifact_hash:
        raise NativeBackendError("native model artifact changed before launch")


def _manifest_int(manifest: dict, primary: str, alias: str | None = None) -> int:
    value = manifest.get(primary, manifest.get(alias) if alias else None)
    if type(value) is not int or not 0 < value < 2**32:
        raise NativeBackendError(f"native manifest {primary} must be a positive uint32")
    if alias and primary in manifest and alias in manifest and manifest[alias] != value:
        raise NativeBackendError(f"native manifest has conflicting {primary}/{alias}")
    return value


def _load_unambiguous_manifest(manifest_path: Path) -> tuple[dict, bytes]:
    """Accept only runtime keys the native text parser reads identically."""
    raw = manifest_path.read_bytes()
    text = raw.decode("utf-8")
    root = json.loads(text, object_pairs_hook=_ObjectPairs)
    if not isinstance(root, _ObjectPairs):
        raise NativeBackendError("native manifest must be a JSON object")
    runtime_keys = set()

    def validate(value, depth=0):
        if isinstance(value, _ObjectPairs):
            local_keys = set()
            for key, child in value:
                if key in local_keys:
                    if key in _NATIVE_RUNTIME_MANIFEST_KEYS:
                        raise NativeBackendError(
                            f"native manifest runtime key {key!r} must be a unique literal top-level key"
                        )
                    raise NativeBackendError(f"native manifest contains duplicate JSON key {key!r}")
                local_keys.add(key)
                if key in _NATIVE_RUNTIME_MANIFEST_KEYS:
                    if depth != 0 or key in runtime_keys:
                        raise NativeBackendError(
                            f"native manifest runtime key {key!r} must be a unique literal top-level key"
                        )
                    runtime_keys.add(key)
                validate(child, depth + 1)
        elif isinstance(value, list):
            for child in value:
                validate(child, depth)

    def materialize(value):
        if isinstance(value, _ObjectPairs):
            return {key: materialize(child) for key, child in value}
        if isinstance(value, list):
            return [materialize(child) for child in value]
        return value

    validate(root)
    manifest = materialize(root)
    for key in runtime_keys:
        # The native reader searches raw bytes for an exact quoted key and does
        # not decode JSON key escapes before choosing the first occurrence.
        if text.count(json.dumps(key)) != 1:
            raise NativeBackendError(
                f"native manifest runtime key {key!r} must be a unique literal top-level key"
            )
    for key in ("dtype", "normalization"):
        if key not in manifest:
            continue
        quoted_key = json.dumps(key)
        key_pos = text.find(quoted_key)
        colon_pos = text.find(":", key_pos + len(quoted_key))
        opening_quote = text.find('"', colon_pos + 1)
        closing_quote = text.find('"', opening_quote + 1)
        if (
            key_pos < 0
            or colon_pos < 0
            or opening_quote < 0
            or closing_quote < 0
            or text[opening_quote + 1:closing_quote] != manifest[key]
        ):
            raise NativeBackendError(
                f"native manifest {key} string value must use a native-compatible literal encoding"
            )
    return manifest, raw


def _validate_artifact(path: Path, contract: GraphContract, backend: str) -> PreparedModel:
    if backend != "mlp":
        raise NativeUnavailable(f"native model backend {backend!r} is not supported by this adapter")
    path = path.resolve()
    try:
        manifest, manifest_bytes = _load_unambiguous_manifest(path / "manifest.json")
        size = _manifest_int(manifest, "state_len")
        classes = _manifest_int(manifest, "num_classes")
        h1 = _manifest_int(manifest, "hidden1", "hd1")
        h2 = _manifest_int(manifest, "hidden2", "hd2")
        blocks = _manifest_int(manifest, "residual_count", "nrd")
        outputs = _manifest_int(manifest, "output_dim")
        if size != contract.state_len or classes < max(contract.num_classes, size):
            raise NativeBackendError("native manifest state_len/num_classes do not match graph/runtime")
        if outputs not in (1, contract.move_count):
            raise NativeBackendError("native output_dim must be 1 or graph move_count")
        if h1 < h2 or h1 % 8 or h2 % 8:
            raise NativeBackendError("native hidden dimensions require hidden1>=hidden2 and multiples of 8")
        if blocks > 1024:
            raise NativeBackendError("native adapter supports at most 1024 residual blocks")
        dtype = manifest.get("dtype")
        if dtype not in ("fp16", "bf16"):
            raise NativeBackendError("native manifest dtype must be fp16 or bf16")
        norm = manifest.get("normalization", "none")
        if norm not in ("none", "batchnorm_folded", "layernorm"):
            raise NativeBackendError("native normalization must be none, batchnorm_folded, or layernorm")
        if manifest.get("graph_hash", contract.graph_hash) != contract.graph_hash:
            raise NativeBackendError("native manifest graph_hash does not match declared graph")
        counts = {"input_weight_hxk": size * classes * h1, "input_bias": h1,
                  "hidden_weight_hxk": h1 * h2, "hidden_bias": h2,
                  "output_weight_hxk": h2 * outputs, "output_bias": outputs}
        norm_layers = [("input", h1), ("hidden", h2)]
        for block in range(blocks):
            for fc in (1, 2):
                prefix = f"residual{block}_fc{fc}"
                counts[f"{prefix}_weight_hxk"] = h2 * h2
                counts[f"{prefix}_bias"] = h2
                norm_layers.append((prefix, h2))
        if norm == "layernorm":
            for prefix, width in norm_layers:
                counts[f"{prefix}_ln_gamma"] = width
                counts[f"{prefix}_ln_beta"] = width
        digest = hashlib.sha256(manifest_bytes)
        digest.update(contract.graph_hash.encode("ascii"))
        import torch
        blob_dtype = torch.float16 if dtype == "fp16" else torch.bfloat16
        artifact_files = ["manifest.json"]
        for name, count in sorted(counts.items()):
            blob = path / f"{name}.{dtype}"
            expected = count * 2
            if not blob.is_file() or blob.stat().st_size != expected:
                raise NativeBackendError(f"native blob {blob.name} missing or wrong length (expected {expected} bytes)")
            artifact_files.append(blob.name)
            digest.update(blob.name.encode("ascii"))
            with blob.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    values = torch.frombuffer(bytearray(chunk), dtype=blob_dtype)
                    if not torch.isfinite(values).all().item():
                        raise NativeBackendError(f"native artifact contains non-finite values in {blob.name}")
        return PreparedModel(path, manifest, backend, digest.hexdigest(), tuple(artifact_files))
    except NativeBackendError:
        raise
    except (OSError, ValueError, TypeError, OverflowError, RuntimeError) as error:
        raise NativeBackendError(f"invalid native model artifact {path}: {error}") from error


def _snapshot_artifact(model: PreparedModel, contract: GraphContract, run_dir: Path) -> PreparedModel:
    """Copy only verified runtime inputs into a private, immutable-by-caller run path."""
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    target = run_dir / "weights"
    if model.weights_dir == target:
        return model
    if target.exists() or target.is_symlink():
        raise NativeBackendError(f"private native weight snapshot already exists at {target}")

    staging = run_dir / f".weights-snapshot-{uuid.uuid4().hex}"
    published = False
    try:
        staging.mkdir()
        for name in model.artifact_files:
            shutil.copyfile(model.weights_dir / name, staging / name)
        snapshot = _validate_artifact(staging, contract, model.backend)
        if snapshot.artifact_hash != model.artifact_hash:
            raise NativeBackendError("native model artifact changed while its private snapshot was copied")
        staging.replace(target)
        published = True
        return PreparedModel(target, snapshot.manifest, snapshot.backend, snapshot.artifact_hash,
                             snapshot.artifact_files)
    except NativeBackendError:
        raise
    except OSError as error:
        raise NativeBackendError(f"could not create private native model snapshot: {error}") from error
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def _require_no_global_forward_hooks() -> None:
    from torch.nn.modules import module as torch_module

    registries = (
        "_global_forward_pre_hooks",
        "_global_forward_hooks",
        "_global_forward_hooks_always_called",
        "_global_forward_hooks_with_kwargs",
    )
    if any(getattr(torch_module, name, None) for name in registries):
        raise NativeUnavailable("native auto-export does not accept global forward hooks")


def _known_model(model, contract: GraphContract):
    """Return exporter format and plain CPU tensors after checking exact inference semantics."""
    import torch
    from torch import nn
    from torch.nn import functional as F

    _require_no_global_forward_hooks()
    if not isinstance(model, nn.Module) or model.__class__.__name__ not in ("Pilgrim", "ResMLPDistance"):
        raise NativeUnavailable("predictor is not a supported Pilgrim/ResMLPDistance native model; use NativeModel for an exported artifact")
    if (model.__class__.__getattribute__ is not nn.Module.__getattribute__
            or model.__class__.__getattr__ is not nn.Module.__getattr__):
        raise NativeUnavailable("native auto-export does not accept custom nn.Module call dispatch")
    if model.__class__.__call__ is not nn.Module.__call__:
        raise NativeUnavailable("native auto-export does not accept a class-level __call__ override")
    if (model.__class__._call_impl is not nn.Module._call_impl
            or "_call_impl" in model.__dict__
            or getattr(model, "_compiled_call_impl", None) is not None):
        raise NativeUnavailable("native auto-export does not accept custom nn.Module call dispatch")
    if "forward" in model.__dict__:
        raise NativeUnavailable("native auto-export does not accept an overridden forward callable")
    if (model.__class__.modules is not nn.Module.modules
            or model.__class__.named_modules is not nn.Module.named_modules
            or "modules" in model.__dict__ or "named_modules" in model.__dict__):
        raise NativeUnavailable("native auto-export does not accept custom module traversal")
    if model.__class__.state_dict is not nn.Module.state_dict or "state_dict" in model.__dict__:
        raise NativeUnavailable("native auto-export does not accept custom state serialization")

    modules, pending, seen = [], [model], set()
    while pending:
        module = pending.pop()
        if id(module) in seen:
            continue
        seen.add(id(module))
        modules.append(module)
        registered = object.__getattribute__(module, "__dict__").get("_modules", {})
        pending.extend(child for child in registered.values() if child is not None)
    if any(module.training for module in modules):
        raise NativeUnavailable("native auto-export requires an inference model in eval mode")
    if getattr(model, "z_add", 0) != 0:
        raise NativeUnavailable("native auto-export does not support shifted input labels")

    def require_child(label, module, expected):
        if type(module) is not expected or "forward" in module.__dict__:
            raise NativeUnavailable(
                f"native auto-export requires exact built-in child module {label} ({expected.__name__})"
            )

    for module in modules:
        attributes = object.__getattribute__(module, "__dict__")
        if (module.__class__.__getattribute__ is not nn.Module.__getattribute__
                or module.__class__.__getattr__ is not nn.Module.__getattr__
                or module.__class__.__call__ is not nn.Module.__call__
                or module.__class__._call_impl is not nn.Module._call_impl
                or "_call_impl" in attributes
                or getattr(module, "_compiled_call_impl", None) is not None):
            raise NativeUnavailable("native auto-export does not accept custom nn.Module call dispatch")
        if "forward" in attributes:
            raise NativeUnavailable("native auto-export does not accept an overridden forward callable")
        if any(attributes.get(name) for name in ("_forward_hooks", "_forward_pre_hooks")):
            raise NativeUnavailable("native auto-export does not accept forward hooks")
        if (module.__class__.state_dict is not nn.Module.state_dict
                or module.__class__._save_to_state_dict is not nn.Module._save_to_state_dict
                or "state_dict" in attributes
                or any(attributes.get(name) for name in ("_state_dict_pre_hooks", "_state_dict_hooks"))):
            raise NativeUnavailable("native auto-export does not accept custom state serialization")
    for module in modules:
        if isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
            affine = module.affine if isinstance(module, nn.BatchNorm1d) else module.elementwise_affine
            if module.eps != 1e-5 or not affine:
                raise NativeUnavailable("native normalization requires affine parameters and eps=1e-5")
        if isinstance(module, nn.BatchNorm1d) and not module.track_running_stats:
            raise NativeUnavailable("native BatchNorm export requires running statistics")
    try:
        state = model.state_dict()
        if any(type(value) is not torch.Tensor or value.layout != torch.strided for value in state.values()):
            raise NativeUnavailable("native exporter only accepts plain dense tensor state_dict values")
        state = {key: value.detach().cpu().contiguous().clone() for key, value in state.items()}
        if any(value.is_floating_point() and not torch.isfinite(value).all().item() for value in state.values()):
            raise NativeUnavailable("native model has non-finite tensor weights")

        def linear_dimensions(name, output_width, input_width):
            weight = state.get(f"{name}.weight")
            bias = state.get(f"{name}.bias")
            return (weight is not None and tuple(weight.shape) == (output_width, input_width)
                    and (bias is None or tuple(bias.shape) == (output_width,)))

        def affine_dimensions(name, width, *, batchnorm=False):
            suffixes = ["weight", "bias"]
            if batchnorm:
                suffixes.extend(("running_mean", "running_var"))
            return all(
                (value := state.get(f"{name}.{suffix}")) is not None
                and tuple(value.shape) == (width,)
                for suffix in suffixes
            )

        if model.__class__.__name__ == "Pilgrim":
            fmt = "batchnorm-folded"
            for label, module, expected in (
                ("input_layer", model.input_layer, nn.Linear),
                ("hidden_layer", model.hidden_layer, nn.Linear),
                ("output_layer", model.output_layer, nn.Linear),
                ("bn1", model.bn1, nn.BatchNorm1d),
                ("bn2", model.bn2, nn.BatchNorm1d),
                ("relu", model.relu, nn.ReLU),
                ("residual_blocks", model.residual_blocks, nn.ModuleList),
            ):
                require_child(label, module, expected)
            for index, block in enumerate(model.residual_blocks):
                for label, module, expected in (
                    ("fc1", block.fc1, nn.Linear),
                    ("fc2", block.fc2, nn.Linear),
                    ("bn1", block.bn1, nn.BatchNorm1d),
                    ("bn2", block.bn2, nn.BatchNorm1d),
                    ("relu", block.relu, nn.ReLU),
                    ("dropout", block.dropout, nn.Dropout),
                ):
                    require_child(f"residual_blocks.{index}.{label}", module, expected)
            classes = getattr(model, "num_classes", None)
            if type(classes) is not int or classes < max(contract.state_len, contract.num_classes):
                raise NativeUnavailable("native MLP class table must cover state_len and state values")
            if getattr(model, "state_size", None) != contract.state_len:
                raise NativeUnavailable("native model state_size does not match graph")
            h1, inputs = state["input_layer.weight"].shape
            h2, h1_in = state["hidden_layer.weight"].shape
            outputs, h2_in = state["output_layer.weight"].shape
            blocks = len(model.residual_blocks)
            if inputs != contract.state_len * classes or h1_in != h1 or h2_in != h2:
                raise NativeUnavailable("native BatchNorm model has incompatible linear dimensions")
            if (not linear_dimensions("input_layer", h1, inputs)
                    or not linear_dimensions("hidden_layer", h2, h1)
                    or not linear_dimensions("output_layer", outputs, h2)
                    or not affine_dimensions("bn1", h1, batchnorm=True)
                    or not affine_dimensions("bn2", h2, batchnorm=True)):
                raise NativeUnavailable("native Pilgrim input, hidden, and output dimensions are incompatible")
            for index, block in enumerate(model.residual_blocks):
                prefix = f"residual_blocks.{index}"
                if (not linear_dimensions(f"{prefix}.fc1", h2, h2)
                        or not linear_dimensions(f"{prefix}.fc2", h2, h2)
                        or not affine_dimensions(f"{prefix}.bn1", h2, batchnorm=True)
                        or not affine_dimensions(f"{prefix}.bn2", h2, batchnorm=True)):
                    raise NativeUnavailable("native Pilgrim residual block dimensions must equal hidden2")

            def linear(x, name):
                bias = state.get(f"{name}.bias")
                return F.linear(x, state[f"{name}.weight"].float(), None if bias is None else bias.float())

            def bn(x, name):
                return F.batch_norm(x, state[f"{name}.running_mean"].float(), state[f"{name}.running_var"].float(),
                                    state[f"{name}.weight"].float(), state[f"{name}.bias"].float(), training=False, eps=1e-5)

            def reference(x):
                x = F.one_hot(x.long(), classes).float().flatten(1)
                x = F.relu(bn(linear(x, "input_layer"), "bn1"))
                x = F.relu(bn(linear(x, "hidden_layer"), "bn2"))
                for i in range(blocks):
                    p = f"residual_blocks.{i}"
                    residual = bn(linear(F.relu(bn(linear(x, p + ".fc1"), p + ".bn1")), p + ".fc2"), p + ".bn2")
                    x = F.relu(x + residual)
                return linear(x, "output_layer")
        else:
            fmt = "resmlp-layernorm"
            require_child("embedding", model.embedding, nn.Embedding)
            require_child("input_stack", model.input_stack, nn.Sequential)
            require_child("res_blocks", model.res_blocks, nn.ModuleList)
            require_child("head", model.head, nn.Linear)
            if len(model.input_stack) != 6:
                raise NativeUnavailable("native ResMLPDistance requires the canonical six-layer input stack")
            for index, expected in enumerate((nn.Linear, nn.LayerNorm, nn.ReLU, nn.Linear, nn.LayerNorm, nn.ReLU)):
                require_child(f"input_stack.{index}", model.input_stack[index], expected)
            for index, block in enumerate(model.res_blocks):
                for label, module, expected in (
                    ("lin1", block.lin1, nn.Linear),
                    ("lin2", block.lin2, nn.Linear),
                    ("ln1", block.ln1, nn.LayerNorm),
                    ("ln2", block.ln2, nn.LayerNorm),
                ):
                    require_child(f"res_blocks.{index}.{label}", module, expected)
            if (model.embedding.padding_idx is not None or model.embedding.max_norm is not None
                    or model.embedding.scale_grad_by_freq or model.embedding.sparse):
                raise NativeUnavailable("native ResMLPDistance requires a canonical plain embedding")
            classes, embedding = state["embedding.weight"].shape
            h1, inputs = state["input_stack.0.weight"].shape
            h2, h1_in = state["input_stack.3.weight"].shape
            outputs, h2_in = state["head.weight"].shape
            blocks = len(model.res_blocks)
            if embedding != 16 or inputs != contract.state_len * 16 or h1_in != h1 or h2_in != h2:
                raise NativeUnavailable("native ResMLPDistance requires embedding_dim=16 and compatible state/hidden shapes")
            if classes < max(contract.state_len, contract.num_classes):
                raise NativeUnavailable("native ResMLPDistance class table is too small")
            if (not linear_dimensions("input_stack.0", h1, inputs)
                    or not linear_dimensions("input_stack.3", h2, h1)
                    or not linear_dimensions("head", outputs, h2)
                    or tuple(model.input_stack[1].normalized_shape) != (h1,)
                    or tuple(model.input_stack[4].normalized_shape) != (h2,)
                    or not affine_dimensions("input_stack.1", h1)
                    or not affine_dimensions("input_stack.4", h2)):
                raise NativeUnavailable("native ResMLPDistance input, hidden, and output dimensions are incompatible")
            for index, block in enumerate(model.res_blocks):
                prefix = f"res_blocks.{index}"
                if (not linear_dimensions(f"{prefix}.lin1", h2, h2)
                        or not linear_dimensions(f"{prefix}.lin2", h2, h2)
                        or tuple(block.ln1.normalized_shape) != (h2,)
                        or tuple(block.ln2.normalized_shape) != (h2,)
                        or not affine_dimensions(f"{prefix}.ln1", h2)
                        or not affine_dimensions(f"{prefix}.ln2", h2)):
                    raise NativeUnavailable("native ResMLPDistance residual block dimensions must equal hidden2")

            def linear(x, name):
                bias = state.get(f"{name}.bias")
                return F.linear(x, state[f"{name}.weight"].float(), None if bias is None else bias.float())

            def ln(x, name):
                return F.layer_norm(x, (x.shape[-1],), state[f"{name}.weight"].float(), state[f"{name}.bias"].float(), eps=1e-5)

            def reference(x):
                x = F.embedding(x.long(), state["embedding.weight"].float()).flatten(1)
                x = F.relu(ln(linear(x, "input_stack.0"), "input_stack.1"))
                x = F.relu(ln(linear(x, "input_stack.3"), "input_stack.4"))
                for i in range(blocks):
                    p = f"res_blocks.{i}"
                    residual = ln(linear(F.relu(ln(linear(x, p + ".lin1"), p + ".ln1")), p + ".lin2"), p + ".ln2")
                    x = F.relu(x + residual)
                return linear(x, "head")
        if outputs not in (1, contract.move_count) or blocks <= 0 or h1 < h2:
            raise NativeUnavailable("native model needs positive residual count, hidden1>=hidden2, and scalar or move-count outputs")
        if h1 % 8 or h2 % 8:
            raise NativeUnavailable("native hidden dimensions must be multiples of 8")
        if blocks > 1024:
            raise NativeUnavailable("native adapter supports at most 1024 residual blocks")
        _require_canonical_forward_graph(model, fmt)
        # Use CPU tensors for both sides without moving or changing the original module.
        probes = [contract.center, contract.start]
        probes.extend(tuple(contract.start[j] for j in g) for g in contract.generators[:8])
        inputs_cpu = torch.tensor(probes, dtype=torch.int64)
        functional_state = {name: value.detach().cpu().clone() for name, value in model.named_parameters()}
        functional_state.update({name: value.detach().cpu().clone() for name, value in model.named_buffers()})
        with torch.no_grad():
            observed = torch.func.functional_call(model, functional_state, (inputs_cpu,))
            expected = reference(inputs_cpu)
        if not isinstance(observed, torch.Tensor) or tuple(observed.shape) not in ((len(probes), outputs), (len(probes),) if outputs == 1 else ()):
            raise NativeUnavailable("native model output shape does not match scalar/Q contract")
        if not torch.allclose(observed.float().reshape_as(expected), expected.float(), rtol=2e-4, atol=2e-5):
            raise NativeUnavailable("model forward does not match native exporter inference semantics on CPU probes")
        _require_no_global_forward_hooks()
        return fmt, state, classes
    except NativeUnavailable:
        raise
    except (AttributeError, KeyError, TypeError, ValueError, RuntimeError, IndexError) as error:
        raise NativeUnavailable(f"model does not match native export schema: {error}") from error


def _fx_signature(module) -> tuple:
    """Normalize an FX graph, including data flow and concrete module types."""
    import torch.fx

    traced = torch.fx.symbolic_trace(module)
    positions, rows = {}, []

    def value(item):
        if isinstance(item, torch.fx.Node):
            return ("node", positions[item])
        if isinstance(item, tuple):
            return ("tuple", tuple(value(part) for part in item))
        if isinstance(item, list):
            return ("list", tuple(value(part) for part in item))
        if isinstance(item, dict):
            return ("dict", tuple((str(key), value(item[key])) for key in sorted(item, key=str)))
        return ("literal", repr(item))

    for node in traced.graph.nodes:
        positions[node] = len(rows)
        target = node.target
        if node.op == "call_module":
            child = traced.get_submodule(str(target))
            target = (str(target), type(child).__module__, type(child).__qualname__)
        elif node.op == "call_function":
            target = (getattr(target, "__module__", ""),
                      getattr(target, "__qualname__", getattr(target, "__name__", repr(target))))
        else:
            target = str(target)
        rows.append((node.op, target, value(node.args), value(node.kwargs)))
    return tuple(rows)


def _require_canonical_forward_graph(model, fmt: str) -> None:
    """Prove the class-level forward graph is the schema exported below."""
    import torch
    from torch import nn

    class PilgrimReference(nn.Module):
        def __init__(self, source, squeeze):
            super().__init__()
            self.num_classes = source.num_classes
            self.input_layer, self.hidden_layer = source.input_layer, source.hidden_layer
            self.output_layer, self.bn1, self.bn2 = source.output_layer, source.bn1, source.bn2
            self.relu, self.residual_blocks = source.relu, source.residual_blocks
            self.squeeze = squeeze

        def forward(self, states):
            x = nn.functional.one_hot(states.long(), self.num_classes).float().flatten(1)
            x = self.relu(self.bn1(self.input_layer(x)))
            x = self.relu(self.bn2(self.hidden_layer(x)))
            for block in self.residual_blocks:
                residual = block.bn2(block.fc2(block.dropout(block.relu(block.bn1(block.fc1(x))))))
                x = block.relu(x + residual)
            output = self.output_layer(x)
            return output.squeeze(-1) if self.squeeze else output

    class ResMLPReference(nn.Module):
        def __init__(self, source, squeeze):
            super().__init__()
            self.embedding, self.input_stack = source.embedding, source.input_stack
            self.res_blocks, self.head = source.res_blocks, source.head
            self.squeeze = squeeze

        def forward(self, states):
            x = self.embedding(states.long()).flatten(1)
            for layer in self.input_stack:
                x = layer(x)
            for block in self.res_blocks:
                residual = block.ln2(block.lin2(torch.relu(block.ln1(block.lin1(x)))))
                x = torch.relu(x + residual)
            output = self.head(x)
            return output.squeeze(-1) if self.squeeze else output

    reference = PilgrimReference if fmt == "batchnorm-folded" else ResMLPReference
    try:
        observed = _fx_signature(model)
        allowed = {_fx_signature(reference(model, squeeze)) for squeeze in (False, True)}
    except Exception as error:
        raise NativeUnavailable(f"native auto-export cannot verify class-level forward graph: {error}") from error
    if observed not in allowed:
        raise NativeUnavailable("native auto-export rejects a customized class-level forward graph")


def _unwrap_predictor(predictor, contract: GraphContract):
    from torch import nn

    if isinstance(predictor, nn.Module):
        return predictor
    if type(predictor) is not Predictor:
        raise NativeUnavailable("native auto-export requires an unmodified CayleyPy Predictor or a direct supported nn.Module")
    for name, original in _PREDICTOR_METHODS.items():
        if (name in predictor.__dict__ or getattr(Predictor, name, None) is not original
                or (original is not None and (original.__module__ != "cayleypy.predictor"
                    or original.__qualname__ != f"Predictor.{name}"))):
            raise NativeUnavailable(f"custom Predictor.{name} cannot be bypassed by native auto-export")
    bound_graph = predictor.graph
    bound_contract = GraphContract.from_graph(bound_graph, bound_graph.definition.central_state)
    if bound_contract.graph_hash != contract.graph_hash:
        raise NativeUnavailable("Predictor graph fingerprint does not match the requested graph and center")
    declared_outputs = getattr(predictor, "n_outputs", 1)
    if type(declared_outputs) is not int or declared_outputs != int(getattr(predictor.predict, "n_outputs", 1)):
        raise NativeUnavailable("custom Predictor.n_outputs cannot be bypassed by native auto-export")
    return predictor.predict


def prepare_model(predictor, contract: GraphContract, options: NativeOptions, run_dir: Path) -> PreparedModel:
    if isinstance(predictor, NativeModel):
        if predictor.graph_hash != contract.graph_hash:
            raise NativeUnavailable("NativeModel graph_hash does not match this ordered graph and center")
        prepared = _validate_artifact(predictor.weights_dir, contract, predictor.backend)
        if predictor.expected_artifact_hash is not None and predictor.expected_artifact_hash != prepared.artifact_hash:
            raise NativeBackendError("native artifact hash changed after the model snapshot was prepared")
        return _snapshot_artifact(prepared, contract, run_dir)
    # Unwrap only the exact unchanged wrapper; arbitrary scoring/preprocessing must not disappear.
    model = _unwrap_predictor(predictor, contract)
    fmt, state, classes = _known_model(model, contract)
    if options.source_dir is None:
        raise NativeUnavailable("source_dir is required to export a recognized native model")
    exporter = options.source_dir / "tools" / "export_stream1_mlp.py"
    if not exporter.is_file():
        raise NativeUnavailable(f"native exporter is unavailable at {exporter}")
    import torch
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = run_dir / "weights"
    checkpoint = run_dir / "model_state.pt"
    command = [sys.executable, "-B", str(exporter), "--weights", str(checkpoint), "--out", str(weights_dir),
               "--format", fmt, "--dtype", "fp16", "--num-classes", str(classes)]
    try:
        torch.save(state, checkpoint)
        # Exporter's legacy weights_only=False reads only this freshly written plain tensor dictionary.
        (run_dir / "model_export_command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", timeout=options.build_timeout_seconds,
                                check=False, shell=False)
        (run_dir / "model_export.log").write_text(result.stdout, encoding="utf-8")
        if result.returncode != 0:
            raise NativeBackendError(f"native model export failed with exit code {result.returncode}; see {run_dir / 'model_export.log'}")
        manifest_path = weights_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["graph_hash"] = contract.graph_hash
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return _validate_artifact(weights_dir, contract, "mlp")
    except NativeBackendError:
        raise
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as error:
        raise NativeBackendError(f"native model export failed: {error}") from error
