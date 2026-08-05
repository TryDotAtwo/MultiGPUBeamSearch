from __future__ import annotations

import ast
import json
from pathlib import Path

from tools.build_kaggle_t4_mlp_autoprofile_sweep import (
    OUT_DIR,
    OUT_NOTEBOOK,
    build_notebook,
)


def test_builder_emits_private_two_t4_notebook(tmp_path: Path) -> None:
    out_dir = tmp_path / OUT_DIR.name
    notebook_path, metadata_path = build_notebook(out_dir)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "trydotatwo/cayley-beam-2xt4-mlp-autoprofiles"
    assert metadata["is_private"] is True
    assert metadata["enable_gpu"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["code_file"] == OUT_NOTEBOOK.name

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "NVIDIA T4" in source
    assert "TORCHRUN_NPROC_PER_NODE = 2" in source
    assert "--nproc-per-node=" in source
    assert "OUTPUT_CLASSES = ('output1', 'output_move_count')" in source
    assert "BEAM_POWERS = tuple(range(16, 26))" in source
    assert "DEPTH_LIMIT = 9" in source
    assert "SHARD_CANDIDATES_BY_POWER" in source
    assert '"selection_metric": "depth_done=8 depth_sec"' in source
    assert 'row.get("max_depth_completed", -1) >= 8' in source
    assert "warmup" in source
    assert "autoprofile_attempts.csv" in source
    assert "selected_profiles.json" in source
    assert "run_summary.json" in source
    assert "except subprocess.TimeoutExpired" in source
    assert "continue" in source
    assert "str(requested_beam)" in source
    assert "min_capacity = max(" in source
    assert 'runtime["stream4_batch_candidates"]' in source
    assert 'runtime["stream3_batch_candidates"]' in source
    assert "BEAM_STREAM3_BATCH_CANDIDATES" in source
    assert "RANK_PREFIX_RE" in source
    assert "DEPTH_RE.finditer" in source
    assert '"output1": {19: (2, 16)}' in source
    assert '"output_move_count": {' in source
    assert "16: (2, 4)" in source
    assert '"stream3_batch_candidates": 196608' in source


def test_generated_code_cells_parse(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path / OUT_DIR.name)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"cell-{index}")
