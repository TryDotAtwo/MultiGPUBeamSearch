"""Opt-in native beam search for CayleyPy. Importing this package never patches CayleyPy."""
from .dispatch import beam_search, disable_native, enable_native
from .errors import NativeBackendError, NativeFallbackWarning, NativeUnavailable
from .models import NativeModel, NativeEnsemble
from .options import NativeOptions
from .preparation import PreparedNative, prepare_native
from .results import NativeBeamSearchResult, replay_path, resolve_move_path, save_result
from .problem import download_bundle, load_graph, read_paths, read_states, select_ids
from .publication import publish_result
from .sources import setup_sources
from .cluster import plan_cluster, ClusterPlan
from .pancake_gnn import PancakeGNN, NeighborConfig

__all__ = ["beam_search", "enable_native", "disable_native", "NativeOptions", "NativeModel", "NativeEnsemble",
           "NativeBeamSearchResult", "replay_path", "resolve_move_path", "save_result",
           "download_bundle", "load_graph", "read_paths", "read_states", "select_ids",
           "publish_result",
           "NativeUnavailable", "NativeBackendError", "NativeFallbackWarning",
           "PreparedNative", "prepare_native", "setup_sources", "plan_cluster", "ClusterPlan",
           "PancakeGNN", "NeighborConfig"]

from .persistent import NativeSession
__all__.append("NativeSession")
