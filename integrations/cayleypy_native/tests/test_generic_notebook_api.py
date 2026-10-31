import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parents[3] / "kaggle" / "cube555_github" / "cube555-native-2xt4-blend.ipynb"
PACKAGE = Path(__file__).resolve().parents[1] / "multigpubeamsearch"


def test_public_notebook_accepts_graph_and_predictor_objects_without_cube_helper():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    combined = "\n".join(code)
    for source in code:
        ast.parse(source)
    assert "GRAPH = None" in combined
    assert "PREDICTOR = None" in combined
    assert "multigpubeamsearch.cube555" not in combined
    assert "beam.publish_result" in combined
    assert "graph.beam_search(" in combined


def test_puzzle_specific_cube_module_is_not_part_of_the_package():
    assert not (PACKAGE / "cube555.py").exists()
