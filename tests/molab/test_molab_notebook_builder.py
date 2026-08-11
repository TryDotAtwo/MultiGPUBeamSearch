from __future__ import annotations

import ast
from pathlib import Path

from tools.build_molab_cayleypy_notebooks import EXAMPLES, build_all


def test_builder_materializes_one_main_and_four_example_marimo_notebooks(tmp_path: Path) -> None:
    outputs = build_all(tmp_path)

    assert set(outputs) == {"main", "444", "megaminx", "ihes", "tetraminx"}
    assert all(path.suffix == ".py" and path.is_file() for path in outputs.values())
    assert all("marimo.App" in path.read_text(encoding="utf-8") for path in outputs.values())
    for path in outputs.values():
        ast.parse(path.read_text(encoding="utf-8"))


def test_main_notebook_is_molab_native_and_checkpoint_only(tmp_path: Path) -> None:
    source = build_all(tmp_path)["main"].read_text(encoding="utf-8")

    assert 'PLATFORM = "molab"' in source
    assert 'REPOSITORY_BRANCH = "molab/notebooks"' in source
    assert 'MODEL_SOURCE_KIND = "checkpoint"' in source
    assert 'MODEL_DTYPE = "auto"' in source
    assert 'CHECKPOINT_FORMAT = "auto"' in source
    assert 'SOLUTION_MODE = "first"' in source
    assert 'TOUCH_BFS_RADIUS = 4' in source
    assert 'PUBLISH_RESULTS = True' in source
    assert "/kaggle/input" not in source
    assert "exactly two T4" not in source


def test_examples_pin_the_same_four_public_workloads_as_kaggle(tmp_path: Path) -> None:
    outputs = build_all(tmp_path)

    for name, spec in EXAMPLES.items():
        source = outputs[name].read_text(encoding="utf-8")
        assert f'COMPETITION = "{spec["competition"]}"' in source
        assert f'PUZZLE_ID_START = {spec["puzzle_id"]}' in source
        assert f'PUZZLE_ID_END = {spec["puzzle_id"]}' in source
        assert 'BEAM_WIDTH = 2**16' in source
        assert 'SOLUTION_MODE = "first"' in source
        assert spec["asset_kind"] in source
        assert spec["asset_ref"] in source


def test_notebooks_use_one_wrapped_execution_cell_and_persistent_outputs(tmp_path: Path) -> None:
    for path in build_all(tmp_path).values():
        source = path.read_text(encoding="utf-8")
        assert "def run_molab_cayleypy_v1(" in source
        assert 'OUTPUT_DIR = Path("/tmp/cayleypy_molab/output")' in source
        assert 'LOG_PATH = Path("/tmp/cayleypy_molab/molab-run.log")' in source
        assert "run_summary.json" in source
        assert "publish_status.json" in source
        assert "submission.csv" in source


def test_examples_are_complete_and_have_unique_titles() -> None:
    assert set(EXAMPLES) == {"444", "megaminx", "ihes", "tetraminx"}
    assert len({spec["title"] for spec in EXAMPLES.values()}) == 4
    assert all(spec["output_dim"] in {1, 24} for spec in EXAMPLES.values())
