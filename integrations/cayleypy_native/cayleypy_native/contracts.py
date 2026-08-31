"""Exact mathematical boundary; never reuse CayleyPy's private state hashes."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral

from .errors import NativeUnavailable


def _integers(value, name: str, size: int, *, single_row: bool = False) -> tuple[int, ...]:
    # Check element types before conversion: numpy promotion would otherwise hide bools.
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        if value.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise NativeUnavailable(f"{name} must contain integers without coercion")
        value = value.detach().cpu().tolist()
    elif isinstance(value, np.ndarray):
        if value.dtype.kind not in "iu":
            raise NativeUnavailable(f"{name} must contain integers without coercion")
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise NativeUnavailable(f"{name} must be a vector of {size} integers")
    if single_row and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if len(value) != size:
        raise NativeUnavailable(f"{name} must have exactly one state of length {size}")
    if any(isinstance(x, (bool, np.bool_)) or not isinstance(x, Integral) for x in value):
        raise NativeUnavailable(f"{name} must contain integers without coercion")
    return tuple(int(x) for x in value)


@dataclass(frozen=True)
class GraphContract:
    state_len: int
    num_classes: int
    generators: tuple[tuple[int, ...], ...]
    generator_names: tuple[str, ...]
    center: tuple[int, ...]
    start: tuple[int, ...]
    graph_hash: str

    @property
    def move_count(self) -> int:
        return len(self.generators)

    @classmethod
    def from_graph(cls, graph, start_state) -> "GraphContract":
        definition = getattr(graph, "definition", None)
        if definition is None or not definition.is_permutation_group():
            raise NativeUnavailable("native adapter currently supports permutation generators only")
        size = definition.state_size
        if type(size) is not int or not 1 <= size <= 120:
            raise NativeUnavailable("native logical state length must be in [1, 120]")
        center = _integers(definition.central_state, "central_state", size)
        start = _integers(start_state, "start_state", size, single_row=True)
        if any(not 0 <= x <= 127 for x in center + start):
            raise NativeUnavailable("native state values must be integers in [0, 127]")
        if Counter(center) != Counter(start):
            raise NativeUnavailable("start_state multiset must exactly match central_state")
        generators = tuple(_integers(g, "generator", size) for g in definition.generators_permutations)
        if not generators or len(generators) > 255:
            raise NativeUnavailable("native move count must be in [1, 255]")
        expected = tuple(range(size))
        if any(tuple(sorted(g)) != expected for g in generators):
            raise NativeUnavailable("every native generator must be a complete gather permutation")
        names = tuple(definition.generator_names)
        if len(names) != len(generators) or any(not isinstance(name, str) for name in names):
            raise NativeUnavailable("generator_names must contain one string per ordered generator")
        # Same schema and JSON encoding as cayleypy.models.checkpoint.graph_hash.
        data = {"generators_type": "PERMUTATION", "generators": [list(g) for g in generators],
                "central_state": list(center)}
        fingerprint = hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()
        return cls(size, max(center) + 1, generators, names, center, start, fingerprint)

    def to_puzzle_info(self) -> dict:
        # Native's small parser must never parse quotes/newlines/arbitrary user names.
        return {"central_state": list(self.center),
                "generators": {f"m{i}": list(g) for i, g in enumerate(self.generators)}}

    def replay(self, path: list[int] | tuple[int, ...]) -> bool:
        if not isinstance(path, (list, tuple)):
            return False
        state = self.start
        for move in path:
            if type(move) is not int or not 0 <= move < self.move_count:
                return False
            state = tuple(state[source] for source in self.generators[move])
        return state == self.center
