"""Explicit prepare-once snapshots; no implicit cache of mutable torch models."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import shutil
import time
import uuid

from .backend import prepare_runtime, runtime_devices, validate_touch_bfs_contract
from .build import validate_runner
from .contracts import GraphContract
from .errors import NativeBackendError
from .models import NativeModel, prepare_model
from .options import NativeOptions


@dataclass(frozen=True)
class PreparedNative:
    """Pass model/options to the existing public graph.beam_search call."""

    model: NativeModel
    options: NativeOptions
    preparation_dir: Path
    runner_sha256: str
    preparation_seconds: float


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def prepare_native(graph, predictor, *, native_options=None, fallback=None) -> PreparedNative:
    """Export and verify an explicit graph/model/runner snapshot once.

    Preparation is strict: it never searches, patches CayleyPy, or falls back.
    ``fallback`` is retained only when explicitly passed, and remains the caller's
    object; it is not a snapshot of torch behavior. Native always uses the frozen
    exported weights, even if the original model is subsequently changed.

    Subsequent searches still verify artifact and executable integrity, device
    capability and each start state. They create fresh workers/history and replay
    paths. There is no persistent CUDA worker or GPU-memory sharing.
    """
    options = NativeOptions() if native_options is None else native_options
    if not isinstance(options, NativeOptions):
        raise TypeError("native_options must be NativeOptions")
    contract = GraphContract.from_graph(graph, graph.definition.central_state)
    validate_touch_bfs_contract(contract, options.touch_bfs_radius, options.touch_bfs_max_entries)
    devices = runtime_devices(graph, options)
    directory = options.cache_dir / "prepared" / uuid.uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        model = prepare_model(predictor, contract, options, directory)
        weights = directory / "weights"
        if model.weights_dir != weights:
            # Declared NativeModel artifacts may belong to another mutable
            # directory. Copy their numeric blobs and manifest before pinning.
            weights.mkdir()
            shutil.copy2(model.weights_dir / "manifest.json", weights / "manifest.json")
            for blob in sorted(model.weights_dir.glob("*." + model.manifest["dtype"])):
                if blob.is_file():
                    shutil.copy2(blob, weights / blob.name)
        snapshot = NativeModel(weights, contract.graph_hash, fallback, model.backend, model.artifact_hash)
        model = prepare_model(snapshot, contract, options, directory)
        # Unlike a zero-step search, explicit preparation builds immediately.
        runtime = prepare_runtime(contract, model, options, directory, devices)
        binary_dir = directory / "bin"
        binary_dir.mkdir()
        runner = binary_dir / runtime.runner.name
        shutil.copy2(runtime.runner, runner)
        _write_json(binary_dir / "native-build.json", runtime.build_metadata)
        verified = validate_runner(runner, contract, model.backend, runtime.architectures)
        prepared_options = replace(options, runner_path=runner, source_dir=None, cutlass_dir=None, devices=devices)
        elapsed = time.perf_counter() - started
        _write_json(directory / "native-preparation.json", {
            "schema_version": 1, "graph_hash": contract.graph_hash, "model_hash": model.artifact_hash,
            "weights_dir": str(weights), "runner_path": str(runner), "runner_sha256": verified["binary_sha256"],
            "devices": list(devices), "build": verified, "preparation_seconds": elapsed,
            "fallback_explicitly_supplied": fallback is not None})
        return PreparedNative(snapshot, prepared_options, directory, verified["binary_sha256"], elapsed)
    except Exception as error:
        try:
            _write_json(directory / "native-preparation-error.json", {"type": type(error).__name__, "error": str(error)})
        except OSError:
            pass
        if isinstance(error, OSError):
            raise NativeBackendError(f"cannot create native prepared snapshot; artifacts={directory}: {error}") from error
        raise
