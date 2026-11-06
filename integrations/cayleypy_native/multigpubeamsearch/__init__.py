"""Opt-in native beam search for CayleyPy. Importing this package never patches CayleyPy."""
from importlib import import_module

from .errors import NativeBackendError, NativeFallbackWarning, NativeUnavailable
from .options import NativeOptions
from .problem import download_bundle, load_graph, read_paths, read_states, select_ids
from .publication import publish_result
from .sources import setup_sources
from .cluster import plan_cluster, ClusterPlan, run_cluster_rank, scheduler_rank

# Scheduler workers only need the standard library to launch the native solver.
# Avoid loading the Python torch runtime in each process that supervises a GPU.
_LAZY_EXPORTS = {
    "beam_search": "dispatch", "disable_native": "dispatch", "enable_native": "dispatch",
    "NativeModel": "models", "NativeEnsemble": "models",
    "PreparedNative": "preparation", "prepare_native": "preparation",
    "NativeBeamSearchResult": "results", "replay_path": "results",
    "resolve_move_path": "results", "save_result": "results",
    "PancakeGNN": "pancake_gnn", "NeighborConfig": "pancake_gnn",
}


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module("." + _LAZY_EXPORTS[name], __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))

__all__ = ["beam_search", "enable_native", "disable_native", "NativeOptions", "NativeModel", "NativeEnsemble",
           "NativeBeamSearchResult", "replay_path", "resolve_move_path", "save_result",
           "download_bundle", "load_graph", "read_paths", "read_states", "select_ids",
           "publish_result",
           "NativeUnavailable", "NativeBackendError", "NativeFallbackWarning",
           "PreparedNative", "prepare_native", "setup_sources", "plan_cluster", "ClusterPlan",
           "PancakeGNN", "NeighborConfig", "run_cluster_rank", "scheduler_rank"]

from .persistent import NativeSession
__all__.append("NativeSession")
