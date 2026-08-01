#!/usr/bin/env python3
"""Build public, configured examples from the universal CayleyPy notebook."""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import build_kaggle_cayleypy_public_notebook as public_notebook

SOLVER_COMMIT = "6d4471c4ab03c528fd7ce1e15c0cc9db11774833"
INGEST_URL = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v1/results"
EXAMPLES: dict[str, dict[str, Any]] = {
    "444": {
        "title": "cayleypy-2xt4-444-example",
        "puzzle_id": 0,
        "slug": "cayleypy-2xt4-444-example",
        "competition": "cayley-py-444-cube",
        "dataset_source": "trydotatwo/cube4-full-transformer-inference",
        "model_source": None,
        "model_root": "/kaggle/input/datasets/trydotatwo/cube4-full-transformer-inference",
        "model_download": None,
        "checkpoint_glob": "model.pth",
        "metadata": "model/model.json",
        "generators": "generators/p002.json",
        "output_dim": 24,
        "description": "Ready-to-run 4x4x4 Cube piece-Transformer example; edit only the USER CONFIG cell.",
    },
    "megaminx": {
        "title": "cayleypy-2xt4-megaminx-example",
        "slug": "cayleypy-2xt4-megaminx-example",
        "puzzle_id": 10,
        "competition": "cayley-py-megaminx",
        "model_source": "trydotatwo/megaminx-output24-p900-t000-q-sym/PyTorch/default/1",
        "model_root": "/kaggle/input/models/trydotatwo/megaminx-output24-p900-t000-q-sym",
        "model_download": "trydotatwo/megaminx-output24-p900-t000-q-sym/PyTorch/default/1",
        "output_dim": 24,
        "description": "Ready-to-run Megaminx output-24 example; edit only the USER CONFIG cell.",
    },
    "ihes": {
        "title": "cayleypy-2xt4-ihes-example",
        "slug": "cayleypy-2xt4-ihes-example",
        "competition": "cayleypy-ihes-cube",
        "model_source": "arabidopsisthalian/ihes-e08192/PyTorch/default/1",
        "puzzle_id": 1,
        "model_root": "/kaggle/input/models/arabidopsisthalian/ihes-e08192",
        "model_download": "arabidopsisthalian/ihes-e08192/PyTorch/default/1",
        "output_dim": 1,
        "description": "Ready-to-run IHES cube example; edit only the USER CONFIG cell.",
    },
    "tetraminx": {
        "title": "cayleypy-2xt4-tetraminx-example",
        "slug": "cayleypy-2xt4-tetraminx-example",
        "competition": "cayley-py-professor-tetraminx-solve-optimally",
        "kernel_source": "rokham/cayleypy-cube-train-and-solve",
        "puzzle_id": 0,
        "model_root": "/kaggle/input/cayleypy-cube-train-and-solve/cayleypy-cube",
        "model_download": None,
        "checkpoint_glob": "p888-t000_1765097793_e01024.pth",
        "discover_model_root_from_checkpoint": True,
        "metadata": "logs/model_p888-t000_1765097793.json",
        "generators": "generators/p888.json",
        "output_dim": 1,
        "description": (
            "Ready-to-run Professor Tetraminx output-1 MLP example using "
            "Rokham's p888 epoch-1024 checkpoint; edit only the USER CONFIG cell."
        ),
    },
}


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected one occurrence of {old!r}, got {source.count(old)}")
    return source.replace(old, new)


def config(spec: dict[str, Any], *, solution_mode: str = "first", slug: str | None = None) -> str:
    s = public_notebook.CONFIG
    root = spec["model_root"]
    download = spec["model_download"]
    slug = spec["slug"] if slug is None else slug
    if solution_mode not in {"first", "collect"}:
        raise ValueError(f"unsupported solution mode: {solution_mode}")
    checkpoint_glob = spec.get("checkpoint_glob")
    if checkpoint_glob:
        if spec.get("discover_model_root_from_checkpoint"):
            checkpoint = (
                f'_CHECKPOINT_MATCHES = sorted(Path("/kaggle/input").rglob("{checkpoint_glob}"))\n'
                f'if len(_CHECKPOINT_MATCHES) != 1:\n'
                f'    raise RuntimeError("expected exactly one {checkpoint_glob} under /kaggle/input; "\n'
                f'                       f"observed={{_CHECKPOINT_MATCHES!r}}")\n'
                f'CHECKPOINT_PATH = _CHECKPOINT_MATCHES[0]\n'
                f'MODEL_ROOT = CHECKPOINT_PATH.parent.parent')
        else:
            checkpoint = (f'MODEL_ROOT = Path("{root}")\n'
                          f'CHECKPOINT_PATH = next(iter(sorted(MODEL_ROOT.rglob("{checkpoint_glob}"))))')
    else:
        checkpoint = (f'MODEL_ROOT = Path("{root}")\n'
                     f'if not any([*MODEL_ROOT.rglob("*.pth"), *MODEL_ROOT.rglob("*.pt")]):\n'
                     f'    import kagglehub\n'
                     f'    MODEL_ROOT = Path(kagglehub.model_download("{download}"))\n'
                     f'CHECKPOINT_PATH = next(iter(sorted([*MODEL_ROOT.rglob("*.pth"), *MODEL_ROOT.rglob("*.pt")])))')
    replacements = {
        'CHECKPOINT_PATH = Path("/kaggle/input/REPLACE_WITH_MODEL/checkpoint.pth")': checkpoint,
        'PUZZLE_INFO_JSON = Path("/kaggle/input/REPLACE_WITH_COMPETITION/puzzle_info.json")': 'COMPETITION_ROOTS = sorted({path.parent for path in Path("/kaggle/input").rglob("puzzle_info.json")})\nif len(COMPETITION_ROOTS) != 1:\n    raise RuntimeError(f"expected exactly one attached CayleyPy input root, observed={COMPETITION_ROOTS!r}")\nCOMPETITION_ROOT = COMPETITION_ROOTS[0]\nPUZZLE_INFO_JSON = COMPETITION_ROOT / "puzzle_info.json"',
        'TEST_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/test.csv")': 'TEST_CSV = COMPETITION_ROOT / "test.csv"',
        'SAMPLE_SUBMISSION_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/sample_submission.csv")': 'SAMPLE_SUBMISSION_CSV = COMPETITION_ROOT / "sample_submission.csv"',
        'BEAM_WIDTH = 2**21': 'BEAM_WIDTH = 2**16',
        'MAX_DEPTH = 100': 'MAX_DEPTH = 100',
        'SOLUTION_MODE = "first"': f'SOLUTION_MODE = "{solution_mode}"',
        'AUTHOR_NAME = "replace-with-author"': f'AUTHOR_NAME = "public-example-{slug}"',
        'COMPETITION = "replace-with-competition"': f'COMPETITION = "{spec["competition"]}"',
        'KAGGLE_OWNER = "replace-with-kaggle-owner"': 'KAGGLE_OWNER = "trydotatwo"',
        'PUZZLE_ID_START = 0': f'PUZZLE_ID_START = {spec["puzzle_id"]}',
        'PUZZLE_ID_END = 0': f'PUZZLE_ID_END = {spec["puzzle_id"]}',
        'KAGGLE_SLUG = "replace-with-kaggle-notebook-slug"': f'KAGGLE_SLUG = "{slug}"',
        'KAGGLE_USERNAME = None': 'KAGGLE_USERNAME = "trydotatwo"',
        'SOLVER_COMMIT = "6d4471c4ab03c528fd7ce1e15c0cc9db11774833"': f'SOLVER_COMMIT = "{SOLVER_COMMIT}"',
    }
    for old, new in replacements.items():
        s = replace_once(s, old, new)
    if spec.get("metadata"):
        metadata_path = " / ".join(f'"{part}"' for part in spec["metadata"].split("/"))
        generator_path = " / ".join(f'"{part}"' for part in spec["generators"].split("/"))
        s = replace_once(s, "CHECKPOINT_METADATA_JSON = None", f"CHECKPOINT_METADATA_JSON = MODEL_ROOT / {metadata_path}")
        s = replace_once(s, "CHECKPOINT_GENERATOR_JSON = None", f"CHECKPOINT_GENERATOR_JSON = MODEL_ROOT / {generator_path}")
        s = replace_once(s, "CHECKPOINT_SOURCE_ROOT = None", "CHECKPOINT_SOURCE_ROOT = MODEL_ROOT")

    return s

def cell(kind: str, source: str, ident: str) -> dict[str, Any]:
    out: dict[str, Any] = {"cell_type": kind, "id": ident, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if kind == "code":
        out.update({"execution_count": None, "outputs": []})
    return out


def build(name: str, out_root: Path, *, public: bool = True, solution_mode: str = "first",
          slug_suffix: str = "") -> Path:
    spec = EXAMPLES[name]
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    slug = spec["slug"] + slug_suffix
    header = public_notebook.HEADER + "\n## Ready-to-run example\n\n" + spec["description"] + "\n\nBeam is fixed to `2**16` for a short example. Toggle `SOLUTION_MODE` between `first` and `collect` in the USER CONFIG cell.\n"
    setup = public_notebook.SETUP.replace("__CANONICAL_SOURCES__", repr([header, public_notebook.PREFLIGHT, public_notebook.RUN, public_notebook.DISPLAY]))
    nb = {"cells": [cell("markdown", header, "header"), cell("code", config(spec, solution_mode=solution_mode, slug=slug), "config"), cell("code", public_notebook.DEBUG_CONFIG, "debug"), cell("code", setup, "setup"), cell("code", public_notebook.PREFLIGHT, "preflight"), cell("code", public_notebook.RUN, "run"), cell("code", public_notebook.DISPLAY, "display")], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}}, "nbformat": 4, "nbformat_minor": 5}
    nb_path = out / f'{name}.ipynb'
    nb_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    dataset_sources = [spec["dataset_source"]] if spec.get("dataset_source") else []
    model_sources = [spec["model_source"]] if spec.get("model_source") else []
    kernel_sources = [spec["kernel_source"]] if spec.get("kernel_source") else []
    meta = {"id": f"trydotatwo/{slug}", "title": slug, "code_file": nb_path.name, "language": "python", "kernel_type": "notebook", "is_private": not public, "enable_gpu": True, "enable_internet": True, "dataset_sources": dataset_sources, "competition_sources": [spec["competition"]], "kernel_sources": kernel_sources, "model_sources": model_sources, "machine_shape": "NvidiaTeslaT4"}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return nb_path


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, default=ROOT / "kaggle" / "cayleypy-public-examples")
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()
    for name in EXAMPLES:
        print(build(name, args.out_root, public=not args.private))

if __name__ == "__main__":
    main()
