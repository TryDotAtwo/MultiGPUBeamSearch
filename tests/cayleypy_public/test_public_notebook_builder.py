from __future__ import annotations

import ast
from hashlib import sha256
import json
from pathlib import Path

from tools.build_kaggle_cayleypy_public_notebook import (
    CONFIG,
    DEBUG_CONFIG,
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
    assert len(notebook["cells"]) == 7
    assert [cell["id"] for cell in notebook["cells"]] == [
        "contract", "user-config", "debug-config", "setup", "preflight", "run", "artifacts",
    ]
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse(_source(cell))
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["id"] == "trydotatwo/cayleypy-2xt4-checkpoint-beam-search"
    assert metadata["is_private"] is False
    assert metadata["machine_shape"] == "NvidiaTeslaT4"


def test_public_notebook_contract_and_no_hidden_model_or_secret_controls(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path)
    rendered = notebook_path.read_text(encoding="utf-8")
    required = (
        "puzzle_info.json", "test.csv", "sample_submission.csv", "Tesla/NVIDIA T4",
        "checkpoint-only", "batchnorm-folded", "resmlp-layernorm", "piece Transformer", "fp16",
        "separate 2xT4 profile registries", "cross-applying profiles fails closed",
        "output_dim=1", "output_dim=move_count", "1 <= state_len <= 256", "0..num_classes-1",
        "PUZZLE_ID_START..PUZZLE_ID_END",
        "off`, `after_original`, or `only", "first", "collect", "touch-BFS",
        "best effort", "MLP supports through `2**25`",
        "piece Transformer supports through measured `2**26`", "public HTTPS host",
        "Redirects are never followed",
        "first publishes one best solution per puzzle; collect publishes every validated solution",
    )
    for phrase in required:
        assert phrase in rendered
    forbidden = ("MODEL_SOURCE", "MODEL_DTYPE", "CHECKPOINT_FORMAT", "GITHUB_TOKEN", "CF_API_TOKEN")
    for value in forbidden:
        assert value not in CONFIG
    assert "BEAM_STREAM" not in CONFIG
    assert "REPLACE_WITH_COMPETITION" in CONFIG
    assert "TOUCH_BFS_RADIUS = 4" in CONFIG


def test_notebook_provenance_rule_excludes_configuration_cell() -> None:
    assert "user configuration and the provenance-computing setup cell are excluded" in HEADER
    changed_config = CONFIG.replace("BEAM_WIDTH = 2**21", "BEAM_WIDTH = 2**22")
    assert sha256(changed_config.encode()).hexdigest() != sha256(CONFIG.encode()).hexdigest()


def test_notebook_pins_official_repository_and_preserves_artifacts_on_cli_failure(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    config_source = _source(notebook["cells"][1])
    setup_source = _source(notebook["cells"][3])
    run_source = _source(notebook["cells"][5])
    display_source = _source(notebook["cells"][6])

    assert "SOLVER_REPOSITORY" not in config_source
    assert "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git" in setup_source
    assert 'SCRATCH = Path("/tmp/cayleypy_public")' in setup_source
    assert "SCRATCH.mkdir(parents=True, exist_ok=True)" in setup_source
    assert 'OUTPUT_DIR = Path("/kaggle/working/cayleypy_public/output")' in setup_source
    assert "f679504baddbab3765c91af526e57ec9360cf309" in config_source
    assert "check=False" in run_source
    assert '["python", "-m", "tools.run_cayleypy_public"' in run_source
    assert 'str(REPO / "tools" / "run_cayleypy_public.py")' not in run_source
    assert "RUN_RETURN_CODE = run_process.returncode" in run_source
    assert "if RUN_RETURN_CODE not in (None, 0):" in display_source
    assert "artifacts above were retained" in display_source


def test_unconfigured_public_template_finishes_with_setup_required(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path)
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    sources = {
        cell["id"]: _source(cell)
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    }
    namespace: dict[str, object] = {"display": lambda value: None}
    for cell_id in ("user-config", "debug-config", "setup", "preflight", "run", "artifacts"):
        exec(compile(sources[cell_id], f"generated-{cell_id}-cell", "exec"), namespace)
    assert namespace["SETUP_REQUIRED"] is True
    assert namespace["RUN_RETURN_CODE"] is None

def test_reader_facing_notebook_source_is_ascii_clean(tmp_path: Path) -> None:
    notebook_path, _ = build_notebook(tmp_path)
    rendered = notebook_path.read_text(encoding="utf-8")
    rendered.encode("ascii")
    assert "2xT4" in rendered


def test_notebook_ships_token_free_public_publish_endpoint() -> None:
    assert "PUBLISH_RESULTS = True" in CONFIG
    assert 'RESULTS_INGEST_URL = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v1/results"' in CONFIG
    assert "results.example" not in CONFIG
    assert "cayleypy-results-ingest-staging.tupa-expert.workers.dev" in CONFIG


def test_display_cell_renders_zero_byte_solutions_csv_without_failing_success(
    tmp_path: Path,
) -> None:
    notebook_path, _ = build_notebook(tmp_path / "notebook")
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    display_source = _source(
        next(cell for cell in notebook["cells"] if cell["id"] == "artifacts")
    )
    output_dir = tmp_path / "output"
    solutions_dir = output_dir / "solutions"
    solutions_dir.mkdir(parents=True)
    (solutions_dir / "solutions.csv").write_bytes(b"")
    displayed: list[object] = []

    exec(
        compile(display_source, "generated-display-cell", "exec"),
        {
            "OUTPUT_DIR": output_dir,
            "RUN_RETURN_CODE": 0,
            "SETUP_REQUIRED": False,
            "display": displayed.append,
        },
    )

    assert len(displayed) == 1
    assert getattr(displayed[0], "empty") is True


def test_notebook_exposes_separate_debug_controls_cell():
    required = (
        'ENABLE_DEBUG = True', 'ENABLE_DEPTH_LOGS = True', 'ENABLE_DEBUG_LOGS = False',
        'DEBUG_STREAM_TIMING = False', 'DEBUG_INFERENCE_TRACE = False',
        'DEBUG_PATH_TRACE = False', 'DEBUG_FINAL_VALIDATE = False',
        'DEBUG_FINAL_EXCHANGE_TRACE = False', 'DEBUG_FINAL_HISTOGRAM_TRACE = False',
        'DEBUG_STREAM4_HISTOGRAM_TRACE = False', 'DEBUG_DEPTH_FLOW_TRACE = False',
        'DEBUG_PIPELINE_STATS = False', 'DEPTH_LOG_EVERY = 1', 'PUZZLE_LOG_EVERY = 1',
    )
    for line in required:
        assert line in DEBUG_CONFIG
