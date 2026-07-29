from __future__ import annotations

import ast
import hashlib
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
    assert "REPLICATION_TARGETS" in source
    assert "MEASURED_REPETITIONS = 3" in source
    assert "MATERIAL_IMPROVEMENT_RATIO = 0.95" in source
    assert "INTER_ATTEMPT_COOLDOWN_SEC = 10" in source
    assert "RUN_TIMEOUT_SEC = 120" in source
    assert "PROCESS_GROUP_TERM_GRACE_SEC = 10" in source
    assert "attempt_timeout_cleanup" in source
    assert "rendezvous_port" in source
    assert '"selection_metric": "depth_done=8 depth_sec"' in source
    assert 'parsed["max_depth_completed"] >= expected_last_depth' in source
    assert "phase=\"warmup\"" in source
    assert "phase=\"replication\"" in source
    assert "autoprofile_replication_attempts.csv" in source
    assert "replication_decisions.json" in source
    assert "selected_profiles.json" in source
    assert "run_summary.json" in source
    assert "except subprocess.TimeoutExpired" in source
    assert "all_ok" in source
    assert "str(requested_beam)" in source
    assert "min_capacity = max(" in source
    assert 'runtime["stream4_batch_candidates"]' in source
    assert 'runtime["stream3_batch_candidates"]' in source
    assert "BEAM_STREAM3_BATCH_CANDIDATES" in source
    assert "RANK_PREFIX_RE" in source
    assert "DEPTH_RE.finditer" in source
    assert "CANONICAL_PROFILE_REGISTRY_JSON" in source
    assert "CANONICAL_PROFILE_REGISTRY_SHA256" in source
    assert "actual_registry_sha256" in source
    assert "expected_anchor_keys" in source
    assert "PINNED_SOLVER_COMMIT" in source
    assert '["git", "checkout"' in source
    assert "tree_sha256" in source
    assert "all runs succeeded" in source
    assert '+ "\\\\n"' not in source  # JSON artifacts must end in a real newline, not a literal escape
    assert "kernel_version" in source
    assert '"stream3_batch_candidates": 196608' in source

def test_generated_notebook_embeds_exact_twenty_anchor_registry(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path / OUT_DIR.name)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    config_source = "".join(notebook["cells"][0]["source"])
    sweep_source = "".join(notebook["cells"][3]["source"])

    namespace: dict[str, object] = {}
    exec(config_source, namespace)
    registry_text = namespace["CANONICAL_PROFILE_REGISTRY_JSON"]
    registry_sha256 = namespace["CANONICAL_PROFILE_REGISTRY_SHA256"]
    assert isinstance(registry_text, str)
    assert hashlib.sha256(registry_text.encode("utf-8")).hexdigest() == registry_sha256

    registry = json.loads(registry_text)
    assert set(registry["profiles"]) == {"output1", "output_move_count"}
    assert sum(len(rows) for rows in registry["profiles"].values()) == 20
    assert all(row["validation_status"] == "measured"
               for rows in registry["profiles"].values() for row in rows.values())
    assert "REPO_DIR / CANONICAL" not in sweep_source



def test_generated_code_cells_parse(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path / OUT_DIR.name)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell.get("source", []))
        ast.parse(source, filename=f"cell-{index}")
