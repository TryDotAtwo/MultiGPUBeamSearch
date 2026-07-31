#!/usr/bin/env python3
"""Build private, pinned Kaggle acceptance packages for the public CayleyPy launcher.

The packages deliberately reuse the public notebook's config/setup/run/display
contract.  They differ only in verified, static acceptance inputs and never
embed a second solver or alter CUDA/C++ code.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from tools import build_kaggle_cayleypy_public_notebook as public_notebook


OUT_ROOT = REPOSITORY_ROOT / "kaggle"
SOLVER_COMMIT = "65aecb1b0946a81a0386d6c1d7509de2d859216a"
COMPETITION = "cayley-py-megaminx"
COMPETITION_ROOT = Path(f"/kaggle/input/competitions/{COMPETITION}")
OUTPUT1_MODEL_SOURCE = "arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1"
OUTPUT1_CHECKPOINT = Path(
    "/kaggle/input/models/arabidopsisthalian/megaminx2048-512-8-e4000/"
    "pytorch/default/1/weights_megaminx2048_512_8_e4000.pth"
)
OUTPUT24_MODEL_SOURCE = "trydotatwo/megaminx-output24-p900-t000-q-sym/PyTorch/default/1"
OUTPUT24_CHECKPOINT = Path(
    "/kaggle/input/models/trydotatwo/megaminx-output24-p900-t000-q-sym/"
    "pytorch/default/1/p900-t000-q-sym_1777988767_best.pth"
)


@dataclass(frozen=True)
class AcceptanceScenario:
    name: str
    title: str
    checkpoint: Path
    model_source: str
    expected_output_dim: int
    beam_width: int
    max_depth: int

    @property
    def slug(self) -> str:
        return f"trydotatwo/cayleypy-public-acceptance-{self.name}"

    @property
    def directory(self) -> Path:
        return OUT_ROOT / f"cayleypy-public-acceptance-{self.name}"


SCENARIOS = {
    "smoke-output1": AcceptanceScenario(
        "smoke-output1", "CayleyPy Public Acceptance Smoke output1",
        OUTPUT1_CHECKPOINT, OUTPUT1_MODEL_SOURCE, 1, 2**16, 8,
    ),
    "final-output1": AcceptanceScenario(
        "final-output1", "CayleyPy Public Acceptance Final output1",
        OUTPUT1_CHECKPOINT, OUTPUT1_MODEL_SOURCE, 1, 2**21, 100,
    ),
    "final-output24": AcceptanceScenario(
        "final-output24", "CayleyPy Public Acceptance Final output24",
        OUTPUT24_CHECKPOINT, OUTPUT24_MODEL_SOURCE, 24, 2**21, 100,
    ),
}


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise ValueError(f"expected exactly one occurrence of {old!r}")
    return source.replace(old, new)


def _config_source(scenario: AcceptanceScenario) -> str:
    source = public_notebook.CONFIG
    source = _replace_once(
        source,
        'CHECKPOINT_PATH = Path("/kaggle/input/REPLACE_WITH_MODEL/checkpoint.pth")',
        f'CHECKPOINT_PATH = Path("{scenario.checkpoint.as_posix()}")',
    )
    source = _replace_once(
        source,
        'PUZZLE_INFO_JSON = Path("/kaggle/input/REPLACE_WITH_COMPETITION/puzzle_info.json")',
        f'PUZZLE_INFO_JSON = Path("{(COMPETITION_ROOT / "puzzle_info.json").as_posix()}")',
    )
    source = _replace_once(
        source,
        'TEST_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/test.csv")',
        f'TEST_CSV = Path("{(COMPETITION_ROOT / "test.csv").as_posix()}")',
    )
    source = _replace_once(
        source,
        'SAMPLE_SUBMISSION_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/sample_submission.csv")',
        f'SAMPLE_SUBMISSION_CSV = Path("{(COMPETITION_ROOT / "sample_submission.csv").as_posix()}")',
    )
    source = _replace_once(source, 'BEAM_WIDTH = 2**21', f'BEAM_WIDTH = {scenario.beam_width}')
    source = _replace_once(source, 'MAX_DEPTH = 100', f'MAX_DEPTH = {scenario.max_depth}')
    source = _replace_once(source, 'AUTHOR_NAME = "replace-with-author"', 'AUTHOR_NAME = "acceptance-test"')
    source = _replace_once(source, 'PUBLISH_RESULTS = True', 'PUBLISH_RESULTS = False')
    source = _replace_once(source, 'RESULTS_INGEST_URL = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v1/results"', 'RESULTS_INGEST_URL = ""')
    source = _replace_once(source, 'COMPETITION = "replace-with-competition"', f'COMPETITION = "{COMPETITION}"')
    source = _replace_once(source, 'KAGGLE_OWNER = "replace-with-kaggle-owner"', 'KAGGLE_OWNER = "trydotatwo"')
    source = _replace_once(source, 'KAGGLE_SLUG = "replace-with-kaggle-notebook-slug"', f'KAGGLE_SLUG = "cayleypy-public-acceptance-{scenario.name}"')
    source = _replace_once(source, 'KAGGLE_USERNAME = None', 'KAGGLE_USERNAME = "trydotatwo"')
    source = _replace_once(source, 'SOLVER_COMMIT = "65aecb1b0946a81a0386d6c1d7509de2d859216a"', f'SOLVER_COMMIT = "{SOLVER_COMMIT}"')
    return source


def _cell(cell_type: str, source: str, cell_id: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "id": cell_id, "metadata": {}}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    cell["source"] = source.strip().splitlines(keepends=True)
    return cell


def _notebook(scenario: AcceptanceScenario) -> dict[str, Any]:
    config = _config_source(scenario)
    setup_template = public_notebook.SETUP
    sources = [public_notebook.HEADER, public_notebook.PREFLIGHT, public_notebook.RUN, public_notebook.DISPLAY]
    setup = setup_template.replace("__CANONICAL_SOURCES__", repr(sources))
    return {
        "cells": [
            _cell("markdown", public_notebook.HEADER, "contract"),
            _cell("code", config, "acceptance-config"),
            _cell("code", public_notebook.DEBUG_CONFIG, "debug-config"),
            _cell("code", setup, "pinned-setup"),
            _cell("code", public_notebook.PREFLIGHT, "preflight"),
            _cell("code", public_notebook.RUN, "run"),
            _cell("code", public_notebook.DISPLAY, "artifacts"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_package(scenario: AcceptanceScenario, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = scenario.directory if out_dir is None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = out_dir / f"{scenario.name}.ipynb"
    metadata_path = out_dir / "kernel-metadata.json"
    notebook_path.write_text(json.dumps(_notebook(scenario), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": scenario.slug,
        "title": scenario.title,
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [COMPETITION],
        "kernel_sources": [],
        "model_sources": [scenario.model_source],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return notebook_path, metadata_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", choices=tuple(SCENARIOS) + ("all",), default="all")
    parser.add_argument("--out-root", type=Path, default=None)
    args = parser.parse_args(argv)
    selected = tuple(SCENARIOS) if args.scenario == "all" else (args.scenario,)
    for name in selected:
        scenario = SCENARIOS[name]
        out_dir = None if args.out_root is None else args.out_root / scenario.name
        print(*build_package(scenario, out_dir))


if __name__ == "__main__":
    main()
