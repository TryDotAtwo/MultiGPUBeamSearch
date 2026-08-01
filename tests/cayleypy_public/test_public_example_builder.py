from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.build_kaggle_cayleypy_public_example import build_example


def _source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])


def test_public_megaminx_example_is_fully_configured_and_runnable(tmp_path: Path) -> None:
    notebook_path, metadata_path = build_example(tmp_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(_source(cell))
    setup = _source(notebook["cells"][3])
    assert "Ready-to-run example" in setup
    config = _source(notebook["cells"][1])
    assert "REPLACE_WITH_" not in config
    assert "PUZZLE_ID_START = 10" in config
    assert "PUZZLE_ID_END = 10" in config
    assert "BEAM_WIDTH = 1024" in config
    assert 'SOLUTION_MODE = "first"' in config
    assert 'REFLECT_MODE = "off"' in config
    assert "PUBLISH_RESULTS = True" in config
    assert 'COMPETITION = "cayley-py-megaminx"' in config
    assert "78565a7cf0b89c394e957dc8ca59ae55b1280f27" in config

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "trydotatwo/cayleypy-2xt4-megaminx-puzzle-10-example"
    assert metadata["is_private"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["competition_sources"] == ["cayley-py-megaminx"]
    assert metadata["model_sources"] == [
        "arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1"
    ]