"""Opt-in dispatch; auto fallback ends before the native worker starts."""
from __future__ import annotations

import inspect
import functools
import json
from pathlib import Path
import threading
import uuid
import warnings

from cayleypy import CayleyGraph
import cayleypy
from cayleypy.algo.beam_search import BeamSearchAlgorithm

from .backend import prepare_runtime, run_native, runtime_devices
from .contracts import GraphContract
from .errors import NativeBackendError, NativeFallbackWarning, NativeUnavailable
from .models import NativeModel, prepare_model
from .options import NativeOptions, NativeOutcome
from .results import NativeBeamSearchResult

_LOCK = threading.RLock()
_ORIGINAL = None
_INSTALLED = None
_SESSION = None
_REGISTRATION = None
_OPTIONS = NativeOptions()
_DEFAULT_BACKEND = "auto"
_MODES = {"auto", "native", "torch"}
_SUPPORTED = {"start_state", "destination_state", "beam_mode", "predictor", "beam_width", "max_steps",
              "history_depth", "return_path", "bfs_result_for_mitm", "use_child_scores", "verbose"}


def _registry_api():
    """Feature-detect the public hook; published CayleyPy 0.1.0 predates it."""
    names = ("register_beam_search_backend", "unregister_beam_search_backend", "get_beam_search_backend",
             "set_default_beam_search_backend", "get_default_beam_search_backend")
    return cayleypy if all(callable(getattr(cayleypy, name, None)) for name in names) else None


def _original_search():
    if _ORIGINAL is not None:
        return _ORIGINAL
    api = _registry_api()
    return api.get_beam_search_backend("torch") if api is not None else CayleyGraph.beam_search


def _check_registration(registration):
    api = registration["api"]
    try:
        owned = all(api.get_beam_search_backend(name) is callback
                    for name, callback in registration["callbacks"].items())
    except (ValueError, KeyError):
        owned = False
    if not owned or api.get_default_beam_search_backend() != registration["default"]:
        raise RuntimeError("CayleyPy backend registration/default was changed by another integration")


def _enable_registry(api, options, default):
    global _REGISTRATION
    if _REGISTRATION is not None:
        _check_registration(_REGISTRATION)
        api.set_default_beam_search_backend(default)
        _REGISTRATION["session"]["options"] = options
        _REGISTRATION["default"] = default
        return
    session = {"options": options}
    original = api.get_beam_search_backend("torch")

    def callback_for(mode):
        def callback(graph, **kwargs):
            with _LOCK:
                selected = kwargs.pop("native_options", session["options"])
            if not isinstance(selected, NativeOptions):
                raise TypeError("native_options must be NativeOptions")
            return _search(original, graph, kwargs, selected, mode)
        return callback

    callbacks = {name: callback_for(name) for name in ("native", "auto")}
    added = []
    try:
        for name, callback in callbacks.items():
            api.register_beam_search_backend(name, callback)
            added.append(name)
        previous = api.set_default_beam_search_backend(default)
    except BaseException:
        for name in reversed(added):
            api.unregister_beam_search_backend(name)
        raise
    _REGISTRATION = {"api": api, "callbacks": callbacks, "session": session,
                     "previous": previous, "default": default}


def _mode(value):
    if not isinstance(value, str) or value not in _MODES:
        raise ValueError("backend must be 'auto', 'native', or 'torch'")
    return value


def _torch_call(original, graph, kwargs):
    params = dict(kwargs)
    predictor = params.get("predictor")
    if isinstance(predictor, NativeModel):
        if predictor.fallback is None:
            raise NativeUnavailable("NativeModel has no CayleyPy fallback predictor; use backend='native' on a supported runtime")
        params["predictor"] = predictor.fallback
    return original(graph, **params)


def _parameters(kwargs):
    # Respect upstream's actual signature; typos must not disappear in **kwargs.
    signature = inspect.signature(BeamSearchAlgorithm.search)
    bound = signature.bind(None, **kwargs)
    bound.apply_defaults()
    params = dict(bound.arguments)
    params.pop("self", None)
    params.setdefault("use_child_scores", False)  # Published CayleyPy 0.1.0 predates this option.
    unknown = set(params) - _SUPPORTED
    if unknown:
        raise NativeUnavailable("native does not implement upstream options: " + ", ".join(sorted(unknown)))
    for name in ("beam_width", "max_steps"):
        value = params[name]
        if type(value) is not int or value < (1 if name == "beam_width" else 0):
            raise ValueError(f"{name} must be an integer {'> 0' if name == 'beam_width' else '>= 0'}")
        maximum = (1 << (64 if name == "beam_width" else 32)) - 1
        if value > maximum:
            raise ValueError(f"{name} exceeds native integer range ({maximum})")
    if params["beam_mode"] != "simple":
        raise NativeUnavailable("native does not implement the upstream advanced mode; use beam_mode='simple'")
    if params["history_depth"] != 0:
        raise NativeUnavailable("upstream history_depth is a taboo policy, not native reconstruction history")
    if params["destination_state"] is not None:
        raise NativeUnavailable("native adapter currently targets graph.central_state; destination_state is not supported")
    if params["bfs_result_for_mitm"] is not None:
        raise NativeUnavailable("upstream BfsResult hashes cannot be reused by native; configure touch_bfs_radius instead")
    for name in ("return_path", "use_child_scores"):
        if type(params.get(name, False)) is not bool:
            raise ValueError(f"{name} must be bool")
    return params


def _write_metadata(path: Path, metadata):
    path.write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")


def _search(original, graph, kwargs, options, mode):
    mode = _mode(mode)
    if mode == "torch":
        return _torch_call(original, graph, kwargs)
    # NativeUnavailable is recoverable only inside the capability/preparation phase.
    try:
        params = _parameters(kwargs)
        contract = GraphContract.from_graph(graph, params["start_state"])
        devices = runtime_devices(graph, options)
        run_dir = options.cache_dir / "runs" / uuid.uuid4().hex
        run_dir.mkdir(parents=True, exist_ok=False)
        model = prepare_model(params["predictor"], contract, options, run_dir)
        if model.manifest["output_dim"] != 1 and not params["use_child_scores"]:
            raise NativeUnavailable("native Q models require use_child_scores=True")
        runtime = None
        if contract.start != contract.center and params["max_steps"] > 0:
            runtime = prepare_runtime(contract, model, options, run_dir, devices)
    except NativeUnavailable as exc:
        if mode == "native":
            raise
        predictor = kwargs.get("predictor")
        if isinstance(predictor, NativeModel) and predictor.fallback is None:
            raise NativeUnavailable(f"{exc}; NativeModel has no fallback predictor") from exc
        if options.warn_on_fallback:
            warnings.warn(f"CayleyPy native unavailable; using torch search: {exc}", NativeFallbackWarning, stacklevel=3)
        return _torch_call(original, graph, kwargs)

    try:
        if contract.start == contract.center:
            outcome = NativeOutcome((), 0.0, None, run_dir, {"execution": "already_at_goal"})
        elif params["max_steps"] == 0:
            outcome = NativeOutcome(None, 0.0, None, run_dir, {"execution": "zero_depth_budget"})
        else:
            outcome = run_native(contract, model, options, params["beam_width"], params["max_steps"], run_dir, devices,
                                 runtime=runtime)
        found = outcome.path is not None
        if found and not contract.replay(outcome.path):
            raise NativeBackendError("native solution failed independent graph replay")
        metadata = dict(outcome.metadata)
        metadata.update({"backend": "native", "graph_hash": contract.graph_hash, "model_hash": model.artifact_hash,
                         "requested_beam_width": params["beam_width"], "effective_beam_width": outcome.effective_beam_width,
                         "devices": list(devices), "elapsed_seconds": outcome.elapsed_seconds,
                         "run_dir": str(outcome.run_dir), "replay_valid": True if found else None,
                         "scoring": "scalar_children" if model.manifest["output_dim"] == 1 else "parent_q",
                         "status": "found" if found else "not_found_within_budget"})
        _write_metadata(run_dir / "adapter-result.json", metadata)
        path = list(outcome.path) if found and (params["return_path"] or not outcome.path) else None
        return NativeBeamSearchResult(found, len(outcome.path) if found else 0, path, {}, graph.definition,
                                      native_metadata=metadata)
    except BaseException as exc:
        try:
            _write_metadata(run_dir / "adapter-error.json", {"type": type(exc).__name__, "error": str(exc)})
        except OSError:
            pass  # Preserve the original failure, including cancellation.
        raise


def beam_search(graph, *, backend="auto", native_options=None, **kwargs):
    """Run dispatch once without patching CayleyPy; the original kwargs remain unchanged on fallback."""
    with _LOCK:
        original = _original_search()
        options = _OPTIONS if native_options is None else native_options
    if not isinstance(options, NativeOptions):
        raise TypeError("native_options must be NativeOptions")
    return _search(original, graph, kwargs, options, backend)


def _make_installed_search(original, session):
    @functools.wraps(original)
    def installed(graph, **kwargs):
        # The session remains valid for methods already bound before disable_native.
        with _LOCK:
            options, default = session["options"], session["default"]
        mode = kwargs.pop("backend", default)
        options = kwargs.pop("native_options", options)
        if not isinstance(options, NativeOptions):
            raise TypeError("native_options must be NativeOptions")
        return _search(original, graph, kwargs, options, mode)
    return installed


def enable_native(options=None, *, default_backend="auto"):
    """Enable explicit backend dispatch, using CayleyPy's public hook when available.

    Older releases use a reversible in-memory wrapper. No download, build,
    CUDA work or file creation happens here.
    """
    global _ORIGINAL, _OPTIONS, _DEFAULT_BACKEND, _INSTALLED, _SESSION
    _mode(default_backend)
    if options is not None and not isinstance(options, NativeOptions):
        raise TypeError("options must be NativeOptions")
    with _LOCK:
        api = _registry_api()
        if api is not None and _ORIGINAL is None:
            selected = options or NativeOptions()
            _enable_registry(api, selected, default_backend)
            _OPTIONS, _DEFAULT_BACKEND = selected, default_backend
            return
        if _ORIGINAL is not None and CayleyGraph.beam_search is not _INSTALLED:
            raise RuntimeError("CayleyGraph.beam_search was replaced by another integration")
        if _ORIGINAL is None:
            _ORIGINAL = CayleyGraph.beam_search
            _SESSION = {}
            _INSTALLED = _make_installed_search(_ORIGINAL, _SESSION)
            CayleyGraph.beam_search = _INSTALLED
        _OPTIONS = options or NativeOptions()
        _DEFAULT_BACKEND = default_backend
        _SESSION.update(options=_OPTIONS, default=_DEFAULT_BACKEND)


def disable_native():
    """Restore the method captured by enable_native; do not overwrite another integration's patch."""
    global _ORIGINAL, _OPTIONS, _DEFAULT_BACKEND, _INSTALLED, _SESSION, _REGISTRATION
    with _LOCK:
        if _REGISTRATION is not None:
            _check_registration(_REGISTRATION)
            api = _REGISTRATION["api"]
            api.set_default_beam_search_backend(_REGISTRATION["previous"])
            for name in reversed(_REGISTRATION["callbacks"]):
                api.unregister_beam_search_backend(name)
            _REGISTRATION = None
        if _ORIGINAL is not None:
            if CayleyGraph.beam_search is not _INSTALLED:
                raise RuntimeError("CayleyGraph.beam_search was replaced by another integration")
            CayleyGraph.beam_search = _ORIGINAL
            _ORIGINAL = None
            _INSTALLED = None
            _SESSION = None
        _OPTIONS = NativeOptions()
        _DEFAULT_BACKEND = "auto"
