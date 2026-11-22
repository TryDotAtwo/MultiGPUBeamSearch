import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[3] / "kaggle" / "cube555_github" / "cube555-native-2xt4-blend.ipynb"
PACKAGE = Path(__file__).resolve().parents[1] / "multigpubeamsearch"


def test_cube555_notebook_is_short_and_uses_cayleypy_search():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    combined = "\n".join(code)
    for source in code:
        ast.parse(source)
    assert len(notebook["cells"]) == 4
    assert "GRAPH_INFO_PATH" not in combined
    assert "PREDICTOR = None" not in combined
    assert "multigpubeamsearch.cube555" not in combined
    assert "TRANSFORMER_WEIGHT = 0.8" in combined
    assert "SOLUTION_MODE = \"collect\"" in combined
    assert "TOUCH_BFS_RADIUS = 5" in combined
    assert "REFLECT_MODE = \"off\"" in combined
    assert "beam.publish_result" in combined
    assert "beam.save_result" in combined
    assert "graph.beam_search(" in combined


def test_puzzle_specific_cube_module_is_not_part_of_the_package():
    assert not (PACKAGE / "cube555.py").exists()
