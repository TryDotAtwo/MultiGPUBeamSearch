from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.build_kaggle_cayleypy_public_acceptance import (
    COMPETITION,
    OUTPUT1_CHECKPOINT,
    OUTPUT1_MODEL_SOURCE,
    OUTPUT24_CHECKPOINT,
    OUTPUT24_MODEL_SOURCE,
    SCENARIOS,
    SOLVER_COMMIT,
    build_package,
)


def _source(notebook: dict, cell_id: str) -> str:
    return "".join(next(cell for cell in notebook["cells"] if cell["id"] == cell_id)["source"])


def _literal_config(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    namespace: dict[str, object] = {}
    exec(compile(tree, "acceptance-config", "exec"), namespace)
    return namespace


def test_acceptance_packages_are_private_pinned_and_parseable(tmp_path: Path) -> None:
    for name, scenario in SCENARIOS.items():
        notebook_path, metadata_path = build_package(scenario, tmp_path / name)
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        assert metadata["id"] == scenario.slug
        assert metadata["is_private"] is True
        assert metadata["machine_shape"] == "NvidiaTeslaT4"
        assert metadata["competition_sources"] == [COMPETITION]
        assert metadata["model_sources"] == [scenario.model_source]
        assert metadata["code_file"] == notebook_path.name
        assert len(notebook["cells"]) == 7
        assert all(not cell.get("outputs") for cell in notebook["cells"] if cell["cell_type"] == "code")
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

        config = _literal_config(_source(notebook, "acceptance-config"))
        assert config["CHECKPOINT_PATH"] == scenario.checkpoint
        assert config["PUZZLE_INFO_JSON"] == Path(f"/kaggle/input/competitions/{COMPETITION}/puzzle_info.json")
        assert config["PUZZLE_ID_START"] == config["PUZZLE_ID_END"] == 0
        assert config["BEAM_WIDTH"] == scenario.beam_width
        assert config["MAX_DEPTH"] == scenario.max_depth
        assert config["TOUCH_BFS_RADIUS"] == (0 if scenario.beam_width >= 2**26 else 4)
        assert config["PUBLISH_RESULTS"] is False
        assert config["RESULTS_INGEST_URL"] == ""
        assert config["SOLVER_COMMIT"] == SOLVER_COMMIT
        setup = _source(notebook, "pinned-setup")
        assert '"git", "fetch", "--depth", "1", "origin", SOLVER_COMMIT' in setup
        assert '"git", "checkout", "--detach", "FETCH_HEAD"' in setup
        assert "pinned commit mismatch" in setup


def test_acceptance_scenarios_preserve_evidenced_models_and_stage_order() -> None:
    smoke = SCENARIOS["smoke-output1"]
    output1 = SCENARIOS["final-output1"]
    output24 = SCENARIOS["final-output24"]
    p26_output1 = SCENARIOS["p26-output1"]
    p26_output24 = SCENARIOS["p26-output24"]
    assert (smoke.beam_width, smoke.max_depth, smoke.expected_output_dim) == (2**16, 8, 1)
    assert (output1.beam_width, output1.max_depth, output1.expected_output_dim) == (2**21, 100, 1)
    assert (output24.beam_width, output24.max_depth, output24.expected_output_dim) == (2**21, 100, 24)
    assert (p26_output1.beam_width, p26_output1.max_depth, p26_output1.expected_output_dim) == (2**26, 9, 1)
    assert (p26_output24.beam_width, p26_output24.max_depth, p26_output24.expected_output_dim) == (2**26, 9, 24)
    assert (smoke.checkpoint, output1.checkpoint) == (OUTPUT1_CHECKPOINT, OUTPUT1_CHECKPOINT)
    assert (smoke.model_source, output1.model_source) == (OUTPUT1_MODEL_SOURCE, OUTPUT1_MODEL_SOURCE)
    assert (output24.checkpoint, output24.model_source) == (OUTPUT24_CHECKPOINT, OUTPUT24_MODEL_SOURCE)
