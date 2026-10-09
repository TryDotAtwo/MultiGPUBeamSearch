"""A native result remains an ordinary CayleyPy BeamSearchResult."""
from dataclasses import dataclass, field
from typing import Any

from cayleypy.algo.beam_search_result import BeamSearchResult


@dataclass(frozen=True, repr=False)
class NativeBeamSearchResult(BeamSearchResult):
    backend: str = field(default="native", init=False)
    native_metadata: dict[str, Any] = field(default_factory=dict)
