from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

from tools.build_kaggle_cayleypy_public_notebook import (
    CONFIG,
    HEADER,
    build_notebook,
)


def _source(cell: dict[str, object]) -> str:
    return "".join(cell["source"])  # type: ignore[arg-type]


def test_public_notebook_is_thin_valid_and_idempotent(tmp_path: Path) -> None:
    notebook_path, metadata_path = build_notebook(tmp_path)
    initial = notebook_path.read_bytes()
    repeated_path, repeated_metadata = build_notebook(tmp_path)
    assert repeated_path == notebook_path
    assert repeated_metadata == metadata_path
    assert notebook_path.read_bytes() == initial

    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert len(notebook["cells"]) == 6
    assert [cell["id"] for cell in notebook["cells"]] == [
        "contract", "user-config", "setup", "preflight", "run", "artifacts",
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(_source(cell))
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"


def test_public_notebook_contract_and_no_hidden_model_or_secret_controls(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path)
    rendered = notebook_path.read_text(encoding="utf-8")
    required = (
        "puzzle_info.json", "test.csv", "sample_submission.csv", "Tesla/NVIDIA T4",
        "checkpoint-only", "batchnorm-folded", "resmlp-layernorm", "fp16",
        "output_dim=1", "output_dim=move_count", "1 <= state_len <= 120", "num_classes=state_len", "0..state_len-1",
        "PUZZLE_ID_START..PUZZLE_ID_END",
        "off`, `after_original`, or `only", "first", "collect", "touch-BFS",
        "best effort", "2**16..2**25", "public HTTPS host",
        "Redirects are never followed",
    )
    for phrase in required:
        assert phrase in rendered
    forbidden = ("MODEL_SOURCE", "MODEL_DTYPE", "CHECKPOINT_FORMAT", "GITHUB_TOKEN", "CF_API_TOKEN")
    for value in forbidden:
        assert value not in CONFIG
    assert "BEAM_STREAM" not in CONFIG
    assert "REPLACE_WITH_COMPETITION" in CONFIG


def test_notebook_provenance_rule_excludes_configuration_cell() -> None:
    assert "user configuration and the provenance-computing setup cell are excluded" in HEADER
    changed_config = CONFIG.replace("BEAM_WIDTH = 2**21", "BEAM_WIDTH = 2**22")
    assert sha256(changed_config.encode()).hexdigest() != sha256(CONFIG.encode()).hexdigest()


def test_notebook_pins_official_repository_and_preserves_artifacts_on_cli_failure(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    config_source = _source(notebook["cells"][1])
    setup_source = _source(notebook["cells"][2])
    run_source = _source(notebook["cells"][4])
    display_source = _source(notebook["cells"][5])

    assert "SOLVER_REPOSITORY" not in config_source
    assert "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git" in setup_source
    assert "bb505484a839d3b78819f86aa28e76b842faab09" in config_source
    assert "check=False" in run_source
    assert '["python", "-m", "tools.run_cayleypy_public"' in run_source
    assert 'str(REPO / "tools" / "run_cayleypy_public.py")' not in run_source
    assert "RUN_RETURN_CODE = run_process.returncode" in run_source
    assert "if RUN_RETURN_CODE != 0:" in display_source
    assert "artifacts above were retained" in display_source


def test_reader_facing_notebook_source_is_ascii_clean(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path)
    rendered = notebook_path.read_text(encoding="utf-8")
    rendered.encode("ascii")
    assert "2xT4" in rendered


def test_notebook_uses_explicit_or_environment_publish_endpoint_without_shipping_one() -> None:
    assert "PUBLISH_RESULTS = False" in CONFIG
    assert 'RESULTS_INGEST_URL = ""' in CONFIG
    assert "CAYLEYPY_RESULTS_INGEST_URL" in CONFIG
    assert "results.example" not in CONFIG
    assert "workers.dev" not in CONFIG
