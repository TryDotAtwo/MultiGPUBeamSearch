"""The public Cube555 example: download a puzzle and load its trained blend."""
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request
import zipfile

from .contracts import GraphContract
from .models import NativeEnsemble

_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch/releases/download/cube555-inputs-20261001/cube555-inputs.zip"
_SHA256 = "49d61a127540cd197db9bf75aacad5712a7ab54d3fbcb39bf2da9ae9bb477286"


def _hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inputs(root):
    root.mkdir(parents=True, exist_ok=True)
    bundle = root / "cube555-inputs.zip"
    if not bundle.is_file() or _hash(bundle) != _SHA256:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as temporary:
            partial = Path(temporary.name)
        try:
            with urllib.request.urlopen(_URL, timeout=120) as response, partial.open("wb") as output:
                shutil.copyfileobj(response, output)
            if _hash(partial) != _SHA256:
                raise ValueError("Cube555 download is incomplete or corrupted")
            partial.replace(bundle)
        finally:
            partial.unlink(missing_ok=True)
    inputs = root / "inputs"
    with zipfile.ZipFile(bundle) as archive:
        for member in archive.infolist():
            if not (inputs / member.filename).resolve().is_relative_to(inputs.resolve()):
                raise ValueError("Unsafe Cube555 archive path")
        archive.extractall(inputs)
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name, entry in manifest.items():
        path = (inputs / name).resolve()
        if not path.is_relative_to(inputs.resolve()):
            raise ValueError("Unsafe Cube555 manifest path")
        if path.stat().st_size != entry["bytes"] or _hash(path) != entry["sha256"]:
            raise ValueError(f"Cube555 input verification failed: {name}")
    return inputs


@dataclass(frozen=True)
class Puzzle:
    definition: object
    start_state: list[int]
    directory: Path
    cache_dir: Path
    model_dir: Path | None = None

    def blend(self, graph, transformer_weight=0.8, *, transformer_checkpoint=None,
              mlp_checkpoint=None, layout_path=None, puzzle_info=None):
        """Load the pretrained Transformer + MLP; weights sum to one."""
        if not 0 <= transformer_weight <= 1:
            raise ValueError("transformer_weight must be between zero and one")
        checkout = Path(__file__).resolve().parents[3]
        if not (checkout / "tools/cube555/export.py").is_file():
            checkout = Path(__file__).resolve().parent / "_native_source"
        if str(checkout) not in sys.path:
            sys.path.insert(0, str(checkout))
        from tools.cube555.export import export_native_models
        exports = self.cache_dir / "exports"
        exports.mkdir(parents=True, exist_ok=True)
        destination = Path(tempfile.mkdtemp(prefix="blend-", dir=exports)) / "models"
        model = self.model_dir or self.directory / "model"
        def checkpoint(value, default):
            value = Path(value) if value is not None else Path(default)
            return value if value.is_absolute() else model / value
        members = export_native_models(
            checkpoint(transformer_checkpoint, "q555_f1_bell2k.pt"), checkpoint(mlp_checkpoint, "q555_2k_BEST.pt"),
            checkpoint(layout_path, "piece_layout_555.json"), checkpoint(puzzle_info, "puzzle_info.json"), destination,
            contract=GraphContract.from_graph(graph, graph.definition.central_state))
        weights = (transformer_weight, float(Decimal(1) - Decimal(str(transformer_weight))))
        return NativeEnsemble(members, weights)


def load(puzzle_id=1020, *, cache_dir=None, model_root=None, competition_root=None):
    """Return a CayleyPy definition and start state from the public puzzle set."""
    from cayleypy import CayleyGraphDef
    root = Path(cache_dir or os.environ.get(
        "MULTIGPUBEAMSEARCH_HOME", Path.home() / ".cache/multigpubeamsearch/cube555")).expanduser().resolve()
    inputs = _inputs(root) if model_root is None or competition_root is None else root / 'inputs'
    model_root = Path(model_root) if model_root is not None else inputs / 'model'
    competition_root = Path(competition_root) if competition_root is not None else inputs / 'competition'
    info = json.loads((model_root / "puzzle_info.json").read_text())
    moves = info["generators"]
    definition = CayleyGraphDef.create(list(moves.values()), generator_names=list(moves),
                                       central_state=info["central_state"])
    with (competition_root / "test.csv").open(newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["initial_state_id"]) == puzzle_id]
    if len(rows) != 1:
        raise ValueError(f"Expected one Cube555 puzzle with index {puzzle_id}; found {len(rows)}")
    return Puzzle(definition, [int(x) for x in rows[0]["initial_state"].split(",")], model_root.parent, root, model_root)


def save_result(result, graph, start_state, directory="cube555_runs"):
    """Replay any returned solution and save the path plus native diagnostics."""
    contract = GraphContract.from_graph(graph, start_state)
    path = list(result.path) if result.path_found and result.path is not None else None
    if result.path_found:
        if path is None:
            raise ValueError("Search returned no replayable solution")
        state = list(start_state)
        for move in path:
            if type(move) is not int or not 0 <= move < len(contract.generators):
                raise ValueError("Solution contains an invalid move")
            state = [state[i] for i in contract.generators[move]]
        if state != list(contract.center):
            raise ValueError("Solution replay failed")
    solutions = getattr(result, 'solutions', ()) or ((tuple(path),) if path is not None else ())
    for solution in solutions:
        state = list(start_state)
        for move in solution:
            if type(move) is not int or not 0 <= move < len(contract.generators): raise ValueError('Invalid collected move')
            state = [state[i] for i in contract.generators[move]]
        if state != list(contract.center): raise ValueError('Collected solution replay failed')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", prefix="solution-",
                                     dir=directory, delete=False, encoding="utf-8") as output:
        json.dump(dict(path_found=result.path_found, path=path,
                       replay_valid=True if result.path_found else None,
                       solutions=solutions, native_metadata=result.native_metadata), output, indent=2, default=str)
    return Path(output.name)


def select_puzzles(start=1020, end=None, *, indices_url=None, random_choice=False):
    """An inclusive range, or the former public CSV list of indices."""
    if indices_url:
        with urllib.request.urlopen(indices_url, timeout=30) as response:
            text = response.read(1_000_001)
        if len(text) > 1_000_000: raise ValueError('Puzzle index list is too large')
        indices = [int(value.strip()) for row in csv.reader(text.decode().splitlines()) for value in row if value.strip()]
    else:
        indices = list(range(start, (start if end is None else end)+1))
    if not indices or any(i < 0 for i in indices): raise ValueError('Expected nonempty nonnegative puzzle indices')
    if random_choice:
        import random
        return [random.choice(indices)]
    return indices


def source_solution(csv_path, puzzle_id, graph):
    """Read a reflection seed using the graph's original move names."""
    if csv_path is None: return None
    with Path(csv_path).open(newline='') as stream:
        rows = [r for r in csv.DictReader(stream) if int(r.get('puzzle_id', r.get('initial_state_id', r.get('id', -1)))) == puzzle_id]
    if not rows: raise ValueError('No reflection source for puzzle '+str(puzzle_id))
    names = GraphContract.from_graph(graph, graph.definition.central_state).generator_names
    paths = [[names.index(move) for move in row.get('solution', row.get('moves', '')).split('.') if move] for row in rows]
    return min(paths, key=len)
