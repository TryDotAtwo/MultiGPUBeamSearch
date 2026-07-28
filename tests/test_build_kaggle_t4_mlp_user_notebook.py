from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.build_kaggle_t4_mlp_user_notebook import OUT_DIR, OUT_NOTEBOOK, build_notebook


def test_universal_notebook_contract(tmp_path: Path) -> None:
    notebook_path, metadata_path = build_notebook(tmp_path / OUT_DIR.name)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "trydotatwo/cayley-beam-2xt4-mlp-universal"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    markdown = "\n".join("".join(c.get("source", [])) for c in notebook["cells"] if c["cell_type"] == "markdown")
    assert "batchnorm-folded" in markdown
    assert "resmlp-layernorm" in markdown
    assert "arbitrary PyTorch" in markdown
    first_code = next("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    for name in ("CHECKPOINT_PATH", "CHECKPOINT_FORMAT", "MODEL_DTYPE", "PUZZLE_INFO_JSON", "TEST_CSV", "SAMPLE_SUBMISSION_CSV", "BEAM_WIDTH", "MAX_DEPTH", "PUZZLE_IDS", "RUN_MODE", "MODEL_SOURCE_MODE"):
        assert name in first_code
    source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    assert "round_half_up_log2" in source
    assert "PROFILE_REGISTRY" in source
    assert "str(BEAM_WIDTH)" in source
    assert "--nproc-per-node=2" in source
    assert "requested_beam" in source and "effective_beam" in source and "alignment_delta" in source
    assert "selected_profile.json" in source
    assert "run_summary.json" in source
    assert "beam_run_results.csv" in source
    assert "submission.csv" in source
    assert "validated_hardware" in source
    assert "validate_solution" in source


def test_generated_universal_code_cells_parse(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path / OUT_DIR.name)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{index}")