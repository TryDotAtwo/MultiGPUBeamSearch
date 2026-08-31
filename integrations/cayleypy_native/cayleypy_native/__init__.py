"""Opt-in native beam search for CayleyPy. Importing this package never patches CayleyPy."""
from .dispatch import beam_search, disable_native, enable_native
from .errors import NativeBackendError, NativeFallbackWarning, NativeUnavailable
from .models import NativeModel
from .options import NativeOptions
from .preparation import PreparedNative, prepare_native
from .results import NativeBeamSearchResult
from .sources import setup_sources

__all__ = ["beam_search", "enable_native", "disable_native", "NativeOptions", "NativeModel",
           "NativeBeamSearchResult", "NativeUnavailable", "NativeBackendError", "NativeFallbackWarning",
           "PreparedNative", "prepare_native", "setup_sources"]
