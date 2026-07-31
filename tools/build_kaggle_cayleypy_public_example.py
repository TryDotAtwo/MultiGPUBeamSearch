#!/usr/bin/env python3
"""Build the fully configured public Megaminx puzzle-10 quick example."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tools import build_kaggle_cayleypy_public_notebook as public_notebook


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPOSITORY_ROOT / "kaggle" / "cayleypy-2xt4-megaminx-puzzle-10-example"
COMPETITION = "cayley-py-megaminx"
COMPETITION_ROOT = Path(f"/kaggle/input/competitions/{COMPETITION}")
MODEL_SOURCE = "arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1"
CHECKPOINT = Path(
    "/kaggle/input/models/arabidopsisthalian/megaminx2048-512-8-e4000/"
    "pytorch/default/1/weights_megaminx2048_512_8_e4000.pth"
)
SLUG = "trydotatwo/cayleypy-2xt4-megaminx-puzzle-10-example"


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    return source.replace(old, new)


def _config_source() -> str:
    source = public_notebook.CONFIG
    replacements = (
        ('CHECKPOINT_PATH = Path("/kaggle/input/REPLACE_WITH_MODEL/checkpoint.pth")', f'CHECKPOINT_PATH = Path("{CHECKPOINT.as_posix()}")'),
        ('PUZZLE_INFO_JSON = Path("/kaggle/input/REPLACE_WITH_COMPETITION/puzzle_info.json")', f'PUZZLE_INFO_JSON = Path("{(COMPETITION_ROOT / "puzzle_info.json").as_posix()}")'),
        ('TEST_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/test.csv")', f'TEST_CSV = Path("{(COMPETITION_ROOT / "test.csv").as_posix()}")'),
        ('SAMPLE_SUBMISSION_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/sample_submission.csv")', f'SAMPLE_SUBMISSION_CSV = Path("{(COMPETITION_ROOT / "sample_submission.csv").as_posix()}")'),
        ("PUZZLE_ID_START = 0", "PUZZLE_ID_START = 10"),
        ("PUZZLE_ID_END = 0", "PUZZLE_ID_END = 10"),
        ("BEAM_WIDTH = 2**21", "BEAM_WIDTH = 1024"),
        ("MAX_DEPTH = 100", "MAX_DEPTH = 40"),
        ('AUTHOR_NAME = "replace-with-author"', 'AUTHOR_NAME = "TryDotAtwo public example"'),
        ('COMPETITION = "replace-with-competition"', f'COMPETITION = "{COMPETITION}"'),
        ('KAGGLE_OWNER = "replace-with-kaggle-owner"', 'KAGGLE_OWNER = "trydotatwo"'),
        ('KAGGLE_SLUG = "replace-with-kaggle-notebook-slug"', 'KAGGLE_SLUG = "cayleypy-2xt4-megaminx-puzzle-10-example"'),
        ("KAGGLE_USERNAME = None", 'KAGGLE_USERNAME = "trydotatwo"'),
    )
    for old, new in replacements:
        source = _replace_once(source, old, new)
    return source


def _cell(cell_type: str, source: str, cell_id: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "id": cell_id, "metadata": {}, "source": source.strip().splitlines(keepends=True)}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def build_example(out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = out_dir / "cayleypy-2xt4-megaminx-puzzle-10-example.ipynb"
    metadata_path = out_dir / "kernel-metadata.json"
    header = public_notebook.HEADER + "\n## Ready-to-run example\n\nThis version already attaches the Megaminx competition and output-1 checkpoint. It solves puzzle 10 with beam 1024 and max depth 40.\n"
    sources = [header, public_notebook.PREFLIGHT, public_notebook.RUN, public_notebook.DISPLAY]
    setup = public_notebook.SETUP.replace("__CANONICAL_SOURCES__", repr(sources))
    notebook = {
        "cells": [
            _cell("markdown", header, "contract"),
            _cell("code", _config_source(), "example-config"),
            _cell("code", public_notebook.DEBUG_CONFIG, "debug-config"),
            _cell("code", setup, "setup"),
            _cell("code", public_notebook.PREFLIGHT, "preflight"),
            _cell("code", public_notebook.RUN, "run"),
            _cell("code", public_notebook.DISPLAY, "artifacts"),
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    metadata = {
        "id": SLUG,
        "title": "CayleyPy 2xT4 Megaminx Puzzle 10 Example",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [MODEL_SOURCE],
    }
    notebook_path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return notebook_path, metadata_path


if __name__ == "__main__":
    print(*build_example())