from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.build_kaggle_cayleypy_public_examples import EXAMPLES, build


def _source(notebook: dict, cell_id: str) -> str:
    return "".join(next(cell for cell in notebook["cells"] if cell["id"] == cell_id)["source"])


def test_cube4_example_uses_the_accepted_piece_transformer_bundle(tmp_path: Path) -> None:
    notebook_path = build("444", tmp_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = json.loads((notebook_path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = _source(notebook, "config")

    assert metadata["dataset_sources"] == ["trydotatwo/cube4-full-transformer-inference"]
    assert metadata["model_sources"] == []
    assert 'Path("/kaggle/input/datasets/trydotatwo/cube4-full-transformer-inference")' in config
    assert 'rglob("model.pth")' in config
    assert 'CHECKPOINT_METADATA_JSON = MODEL_ROOT / "model" / "model.json"' in config
    assert 'CHECKPOINT_GENERATOR_JSON = MODEL_ROOT / "generators" / "p002.json"' in config
    assert 'CHECKPOINT_SOURCE_ROOT = MODEL_ROOT' in config


def test_all_public_examples_share_the_universal_contract(tmp_path: Path) -> None:
    for name, spec in EXAMPLES.items():
        notebook_path = build(name, tmp_path)
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        metadata = json.loads((notebook_path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                assert not cell["outputs"]
        config = _source(notebook, "config")
        assert "BEAM_WIDTH = 2**16" in config
        assert f'PUZZLE_ID_START = {spec["puzzle_id"]}' in config
        assert f'PUZZLE_ID_END = {spec["puzzle_id"]}' in config
        assert 'SOLUTION_MODE = "first"' in config
        assert f'COMPETITION = "{spec["competition"]}"' in config
        assert metadata["id"] == f'trydotatwo/{spec["slug"]}'
        assert 'Path("/kaggle/input").rglob("puzzle_info.json")' in config
        assert "if len(COMPETITION_ROOTS) != 1:" in config
        assert "PUZZLE_INFO_JSON = COMPETITION_ROOT / \"puzzle_info.json\"" in config
        assert "TEST_CSV = COMPETITION_ROOT / \"test.csv\"" in config
        assert metadata["is_private"] is False
        assert metadata["machine_shape"] == "NvidiaTeslaT4"


def test_private_collect_package_has_isolated_slug_and_collect_mode(tmp_path: Path) -> None:
    notebook_path = build(
        "ihes", tmp_path, public=False, solution_mode="collect", slug_suffix="-collect"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    metadata = json.loads((notebook_path.parent / "kernel-metadata.json").read_text(encoding="utf-8"))
    config = _source(notebook, "config")

    assert 'SOLUTION_MODE = "collect"' in config
    assert 'KAGGLE_SLUG = "cayleypy-2xt4-ihes-example-collect"' in config
    assert metadata["id"] == "trydotatwo/cayleypy-2xt4-ihes-example-collect"
    assert metadata["is_private"] is True
