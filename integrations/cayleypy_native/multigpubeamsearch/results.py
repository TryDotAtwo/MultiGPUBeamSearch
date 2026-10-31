"""Generic CayleyPy result validation and persistence."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from cayleypy.algo.beam_search_result import BeamSearchResult


@dataclass(frozen=True, repr=False)
class NativeBeamSearchResult(BeamSearchResult):
    backend: str = field(default="native", init=False)
    native_metadata: dict[str, Any] = field(default_factory=dict)
    solutions: tuple[tuple[int, ...], ...] = field(default_factory=tuple)


def resolve_move_path(graph, path) -> tuple[int, ...]:
    """Convert a path of move indices or graph move names to ordered indices."""
    from .contracts import GraphContract

    contract = GraphContract.from_graph(graph, graph.definition.central_state)
    if isinstance(path, str):
        path = [move for move in path.split(".") if move]
    if not isinstance(path, (list, tuple)):
        raise ValueError("solution path must be a sequence of move indices or names")
    names = contract.generator_names
    if len(set(names)) != len(names):
        raise ValueError("graph move names must be unique to resolve named paths")
    indices = {name: index for index, name in enumerate(names)}
    resolved = []
    for move in path:
        if type(move) is int and 0 <= move < contract.move_count:
            resolved.append(move)
        elif isinstance(move, str) and move in indices:
            resolved.append(indices[move])
        else:
            raise ValueError(f"invalid move in solution path: {move!r}")
    return tuple(resolved)


def replay_path(graph, start_state, path) -> bool:
    """Return whether ``path`` reaches this graph's central state from ``start_state``."""
    from .contracts import GraphContract

    contract = GraphContract.from_graph(graph, start_state)
    return contract.replay(resolve_move_path(graph, path))


def save_result(result, graph, start_state, directory="beam_results", *, metadata=None):
    """Replay-check and atomically save a result for any supported CayleyPy graph."""
    from .contracts import GraphContract

    contract = GraphContract.from_graph(graph, start_state)
    raw_path = getattr(result, "path", None)
    path = resolve_move_path(graph, raw_path) if raw_path is not None else None
    path_found = getattr(result, "path_found", path is not None)
    if type(path_found) is not bool:
        raise ValueError("result.path_found must be a boolean")
    if path_found and path is None:
        raise ValueError("search returned a solution without a replayable path")
    if not path_found and path is not None:
        raise ValueError("search returned a path while path_found is false")
    if path is not None and not contract.replay(path):
        raise ValueError("solution path does not replay to the graph's central state")

    raw_solutions = getattr(result, "solutions", ()) or ()
    solutions = []
    for candidate in raw_solutions:
        resolved = resolve_move_path(graph, candidate)
        if not contract.replay(resolved):
            raise ValueError("collected solution does not replay to the graph's central state")
        if resolved not in solutions:
            solutions.append(resolved)
    if path is not None and path not in solutions:
        solutions.insert(0, path)

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    record = dict(
        schema_version=1,
        graph_hash=contract.graph_hash,
        start_state=list(contract.start),
        target_state=list(contract.center),
        path_found=path_found,
        path=list(path) if path is not None else None,
        replay_valid=True if path is not None else None,
        solutions=[list(solution) for solution in solutions],
        native_metadata=getattr(result, "native_metadata", None),
        metadata={} if metadata is None else metadata,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="solution-",
                                     dir=directory, delete=False, encoding="utf-8") as output:
        temporary = Path(output.name)
        try:
            json.dump(record, output, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        except (TypeError, ValueError):
            output.close()
            temporary.unlink(missing_ok=True)
            raise
    destination = temporary.with_name(temporary.stem + "-verified.json")
    try:
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination
