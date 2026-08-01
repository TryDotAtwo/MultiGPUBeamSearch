#!/usr/bin/env python3
"""Build the thin public CayleyPy checkpoint-only 2xT4 Kaggle notebook.

The notebook is intentionally a handoff layer.  Search, export, validation and
publication behaviour remain in ``tools/run_cayleypy_public.py`` in the checked
out repository; no solver implementation is embedded in the notebook.
"""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPOSITORY_ROOT / "kaggle" / "cayleypy-2xt4-checkpoint-public"
OUT_NOTEBOOK = OUT_DIR / "cayleypy-2xt4-checkpoint-public.ipynb"
OUT_METADATA = OUT_DIR / "kernel-metadata.json"
OFFICIAL_SOLVER_REPOSITORY = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"


HEADER = """# CayleyPy checkpoint-only beam search -- exactly 2xT4

This is a thin, reader-facing launcher for **standard CayleyPy competitions**:
the attached input must contain `puzzle_info.json`, `test.csv`, and
`sample_submission.csv`. It runs the repository's existing two-rank beam-search
implementation; this notebook does not contain a second solver.

## Fixed model contract

- Runtime: exactly **two Tesla/NVIDIA T4 GPUs**; the run fails before export if
  the hardware differs.
- Models: checkpoint-only, with automatic checkpoint-format detection. Supported
  families are **batchnorm-folded MLP**, **resmlp-layernorm MLP**, and the
  supported **piece Transformer** bundle. Standard Transformer metadata and
  generators are discovered beside the checkpoint; explicit paths are optional.
- MLP checkpoints may use `output_dim=1` or `output_dim=move_count`; the current
  piece Transformer contract uses `output_dim=move_count` (24 for Cube4).
- MLP and Transformer use separate 2xT4 profile registries. A profile is selected
  only after backend detection, and cross-applying profiles fails closed.
- Model dtype is automatically **fp16**; users do not select dtype or format.
- The fixed public State128 runner requires `1 <= state_len <= 120`. The value
  alphabet is inferred independently from contiguous central-state labels
  `0..num_classes-1`; every selected initial state must use that same alphabet.
- The inclusive `PUZZLE_ID_START..PUZZLE_ID_END` range is checked strictly:
  a missing ID is a configuration error.

## Search contract

Your requested beam is kept, except for documented two-rank/shard alignment.
The nearest backend-specific measured/safe 2xT4 profile is selected automatically:
MLP supports through `2**25`, while piece Transformer supports through measured `2**26`.
You may choose reflection `off`, `after_original`, or `only`; choose either first
solution or collection through a chosen depth; and set the touch-BFS radius.
CUDA/C++ algorithm knobs are deliberately absent.

## Results publishing

The mode controls what is sent: `first publishes one best solution per puzzle; collect publishes every validated solution`. The local `submission.csv` always uses the best solution per puzzle.
One notebook run is packed into one deterministic gzip archive and sent with one
HTTPS request. Only when the compressed archive would exceed 32 MiB is it split
into bounded parts, which are uploaded sequentially without dropping solutions.


Publishing is best effort: valid local solutions and `submission.csv` remain even
if the ingest service is unavailable. The status is saved in `publish_status.json`.
No token or secret belongs in this notebook.
Publishing uses the bundled free HTTPS ingest endpoint by default. Set
`PUBLISH_RESULTS=False` to disable it, or override `RESULTS_INGEST_URL`. Publishing
fails closed before solve if author or competition/Kaggle provenance still uses
the supplied `replace-with-*` placeholders.
The endpoint must be a public HTTPS host: localhost and non-global IP literals are
rejected. Redirects are never followed, so a configured origin cannot redirect
the publisher to a private or plaintext destination.

### Provenance hash rule

`KAGGLE_NOTEBOOK_SHA256` is the SHA-256 of canonical immutable **cell sources**:
the header, preflight, CLI invocation, and artifact-display cells, in notebook
order. The user configuration and the provenance-computing setup cell are excluded
to avoid a self-referential hash. Each source is UTF-8, normalized to LF with one
trailing LF, and joined using one LF. The setup cell computes the value and the
CLI records it only when publishing is enabled.
"""


CONFIG = r'''from pathlib import Path

# USER CONFIG -- edit only these values. Explicit Kaggle paths prevent ambiguous
# auto-discovery across several attached competitions/models.
CHECKPOINT_PATH = Path("/kaggle/input/REPLACE_WITH_MODEL/checkpoint.pth")
# Usually leave these as None: supported Transformer bundles are discovered
# next to CHECKPOINT_PATH. Set explicit paths only for a non-standard bundle.
CHECKPOINT_METADATA_JSON = None
CHECKPOINT_GENERATOR_JSON = None
CHECKPOINT_SOURCE_ROOT = None
PUZZLE_INFO_JSON = Path("/kaggle/input/REPLACE_WITH_COMPETITION/puzzle_info.json")
TEST_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/test.csv")
SAMPLE_SUBMISSION_CSV = Path("/kaggle/input/REPLACE_WITH_COMPETITION/sample_submission.csv")

PUZZLE_ID_START = 0                 # inclusive
PUZZLE_ID_END = 0                   # inclusive
BEAM_WIDTH = 2**21                  # requested value; only alignment may increase it
MAX_DEPTH = 100
REFLECT_MODE = "off"                # off | after_original | only
REFLECT_SOURCE_CSV = None            # required only for REFLECT_MODE == "only"
SOLUTION_MODE = "first"             # first | collect
COLLECT_UNTIL_DEPTH = MAX_DEPTH      # meaningful when SOLUTION_MODE == "collect"
MAX_COLLECTED_SOLUTIONS = 100
# Radius 4 is the measured whole-notebook balance for the supplied <=24-move
# examples; radius 5 reduces solve depth further but costs more setup time.
TOUCH_BFS_RADIUS = 4

AUTHOR_NAME = "replace-with-author"
PUBLISH_RESULTS = True               # best effort; set False to disable publishing
# Public token-free endpoint; valid local results survive every publishing failure.
RESULTS_INGEST_URL = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v1/results"
COMPETITION = "replace-with-competition"
KAGGLE_OWNER = "replace-with-kaggle-owner"
KAGGLE_SLUG = "replace-with-kaggle-notebook-slug"
KAGGLE_VERSION = 1
KAGGLE_USERNAME = None

# Official repository revision. Keep this 40-character commit pinned for a
# reproducible solve; SOLVER_COMMIT always refers to TryDotAtwo/MultiGPUBeamSearch.
SOLVER_COMMIT = "78565a7cf0b89c394e957dc8ca59ae55b1280f27"
'''


DEBUG_CONFIG = r'''# OPTIONAL DEBUG / LOGGING -- safe defaults; enable only what you need.
ENABLE_DEBUG = True
ENABLE_DEPTH_LOGS = True
ENABLE_DEBUG_LOGS = False
DEBUG_STREAM_TIMING = False
DEBUG_INFERENCE_TRACE = False
DEBUG_PATH_TRACE = False
DEBUG_FINAL_VALIDATE = False
DEBUG_FINAL_EXCHANGE_TRACE = False
DEBUG_FINAL_HISTOGRAM_TRACE = False
DEBUG_STREAM4_HISTOGRAM_TRACE = False
DEBUG_DEPTH_FLOW_TRACE = False
DEBUG_PIPELINE_STATS = False

DEPTH_LOG_EVERY = 1
PUZZLE_LOG_EVERY = 1
'''


SETUP = r'''# Checkout only the pinned public solver revision and derive notebook provenance.
from hashlib import sha256
import json
from pathlib import Path
import shutil
import subprocess

SCRATCH = Path("/tmp/cayleypy_public")
WORK = Path("/kaggle/working/cayleypy_public")
REPO = SCRATCH / "MultiGPUBeamSearch"
CONFIG_PATH = SCRATCH / "run_config.json"
OUTPUT_DIR = Path("/kaggle/working/cayleypy_public/output")
OFFICIAL_SOLVER_REPOSITORY = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"

if SOLVER_COMMIT == "0" * 40:
    raise ValueError("set SOLVER_COMMIT to a real 40-character public commit")
required_paths = (CHECKPOINT_PATH, PUZZLE_INFO_JSON, TEST_CSV, SAMPLE_SUBMISSION_CSV)
SETUP_REQUIRED = any("REPLACE_WITH_" in str(path) for path in required_paths)
if SETUP_REQUIRED:
    print("SETUP_REQUIRED: Copy & Edit this notebook, attach competition data and a supported checkpoint, then replace every REPLACE_WITH_* path in USER CONFIG.")
    checked_commit = SOLVER_COMMIT
else:
    for required_path in required_paths:
        if not required_path.is_file():
            raise FileNotFoundError(required_path)
    for optional_path in (CHECKPOINT_METADATA_JSON, CHECKPOINT_GENERATOR_JSON):
        if optional_path is not None and not Path(optional_path).is_file():
            raise FileNotFoundError(optional_path)
    if CHECKPOINT_SOURCE_ROOT is not None and not Path(CHECKPOINT_SOURCE_ROOT).is_dir():
        raise NotADirectoryError(CHECKPOINT_SOURCE_ROOT)

    SCRATCH.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", OFFICIAL_SOLVER_REPOSITORY, str(REPO)], check=True)
    subprocess.run(["git", "fetch", "--depth", "1", "origin", SOLVER_COMMIT], cwd=REPO, check=True)
    subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=REPO, check=True)
    checked_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if checked_commit != SOLVER_COMMIT:
        raise RuntimeError(f"pinned commit mismatch: expected={SOLVER_COMMIT} got={checked_commit}")

# Canonical notebook-source hash: immutable cells only (not user config/setup).
NOTEBOOK_CELL_SOURCES = __CANONICAL_SOURCES__
def canonical_cell_source(source):
    return source.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
KAGGLE_NOTEBOOK_SHA256 = sha256(
    "\n".join(canonical_cell_source(source) for source in NOTEBOOK_CELL_SOURCES).encode("utf-8")
).hexdigest()
print({"solver_commit": checked_commit, "notebook_source_sha256": KAGGLE_NOTEBOOK_SHA256})
'''


PREFLIGHT = r'''# Validate the fixed public runtime before compiling or solving.
import subprocess

if SETUP_REQUIRED:
    print("preflight_skipped", {"reason": "SETUP_REQUIRED"})
else:
    gpu_names = [line.strip() for line in subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], text=True
    ).splitlines() if line.strip()]
    if len(gpu_names) != 2 or any(name not in {"Tesla T4", "NVIDIA T4"} for name in gpu_names):
        raise RuntimeError(f"this launcher requires exactly two T4 GPUs; observed={gpu_names!r}")
    print("preflight_ok", {"gpu_names": gpu_names, "requested_beam": BEAM_WIDTH,
                           "puzzle_ids": [PUZZLE_ID_START, PUZZLE_ID_END]})
'''

RUN = r'''# The repository CLI owns model detection/export, profile selection, build, solve and validation.
run_config = {
    "author_name": AUTHOR_NAME,
    "checkpoint_path": str(CHECKPOINT_PATH),
    "checkpoint_metadata_json": None if CHECKPOINT_METADATA_JSON is None else str(CHECKPOINT_METADATA_JSON),
    "checkpoint_generator_json": None if CHECKPOINT_GENERATOR_JSON is None else str(CHECKPOINT_GENERATOR_JSON),
    "checkpoint_source_root": None if CHECKPOINT_SOURCE_ROOT is None else str(CHECKPOINT_SOURCE_ROOT),
    "puzzle_info_json": str(PUZZLE_INFO_JSON),
    "test_csv": str(TEST_CSV),
    "sample_submission_csv": str(SAMPLE_SUBMISSION_CSV),
    "puzzle_id_start": PUZZLE_ID_START,
    "puzzle_id_end": PUZZLE_ID_END,
    "beam_width": BEAM_WIDTH,
    "max_depth": MAX_DEPTH,
    "reflect_mode": REFLECT_MODE,
    "reflect_source_csv": None if REFLECT_SOURCE_CSV is None else str(REFLECT_SOURCE_CSV),
    "solution_mode": SOLUTION_MODE,
    "collect_until_depth": COLLECT_UNTIL_DEPTH,
    "max_collected_solutions": MAX_COLLECTED_SOLUTIONS,
    "touch_bfs_radius": TOUCH_BFS_RADIUS,
    "enable_debug": ENABLE_DEBUG,
    "enable_depth_logs": ENABLE_DEPTH_LOGS,
    "enable_debug_logs": ENABLE_DEBUG_LOGS,
    "debug_stream_timing": DEBUG_STREAM_TIMING,
    "debug_inference_trace": DEBUG_INFERENCE_TRACE,
    "debug_path_trace": DEBUG_PATH_TRACE,
    "debug_final_validate": DEBUG_FINAL_VALIDATE,
    "debug_final_exchange_trace": DEBUG_FINAL_EXCHANGE_TRACE,
    "debug_final_histogram_trace": DEBUG_FINAL_HISTOGRAM_TRACE,
    "debug_stream4_histogram_trace": DEBUG_STREAM4_HISTOGRAM_TRACE,
    "debug_depth_flow_trace": DEBUG_DEPTH_FLOW_TRACE,
    "debug_pipeline_stats": DEBUG_PIPELINE_STATS,
    "depth_log_every": DEPTH_LOG_EVERY,
    "puzzle_log_every": PUZZLE_LOG_EVERY,
    "publish_results": PUBLISH_RESULTS,
    "results_ingest_url": RESULTS_INGEST_URL,
    "competition": COMPETITION,
    "kaggle_owner": KAGGLE_OWNER,
    "kaggle_slug": KAGGLE_SLUG,
    "kaggle_version": KAGGLE_VERSION,
    "kaggle_username": KAGGLE_USERNAME,
    "solver_commit": SOLVER_COMMIT,
    "kaggle_notebook_sha256": KAGGLE_NOTEBOOK_SHA256,
}
if SETUP_REQUIRED:
    RUN_RETURN_CODE = None
    print("run_skipped", {"reason": "SETUP_REQUIRED"})
else:
    CONFIG_PATH.write_text(json.dumps(run_config, sort_keys=True) + "\n", encoding="utf-8")
    run_process = subprocess.run(
        ["python", "-m", "tools.run_cayleypy_public",
         "--config-json", str(CONFIG_PATH), "--output-dir", str(OUTPUT_DIR)],
        cwd=REPO,
        check=False,
    )
    RUN_RETURN_CODE = run_process.returncode
print({"cli_return_code": RUN_RETURN_CODE})
'''


DISPLAY = r'''# Bounded handoff: show machine-readable statuses and a small solution preview.
import json
import pandas as pd

if SETUP_REQUIRED:
    print("SETUP_REQUIRED: no solve was attempted. Use Copy & Edit, attach inputs, edit USER CONFIG, and Run All.")

for name in ("selected_profile.json", "preflight.json", "publish_status.json", "run_summary.json"):
    path = OUTPUT_DIR / name
    print(f"\n== {name} ==")
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2)[:6000] if path.is_file() else "missing")

for name in ("beam_run_results.csv", "solutions/solutions.csv", "submission.csv"):
    path = OUTPUT_DIR / name
    print(f"\n== {name} ==")
    if path.is_file():
        try:
            preview = pd.read_csv(path).head(20)
        except pd.errors.EmptyDataError:
            preview = pd.DataFrame()
        display(preview)
    else:
        print("missing")

if RUN_RETURN_CODE not in (None, 0):
    raise RuntimeError(f"public CLI failed with return code {RUN_RETURN_CODE}; artifacts above were retained")
'''


def _cell(cell_type: str, source: str, cell_id: str) -> dict[str, Any]:
    cell: dict[str, Any] = {"cell_type": cell_type, "id": cell_id, "metadata": {}}
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    cell["source"] = source.strip().splitlines(keepends=True)
    return cell


def _setup_source() -> str:
    sources = [HEADER, PREFLIGHT, RUN, DISPLAY]
    return SETUP.replace("__CANONICAL_SOURCES__", repr(sources))


def _notebook() -> dict[str, Any]:
    sources = [HEADER, CONFIG, DEBUG_CONFIG, _setup_source(), PREFLIGHT, RUN, DISPLAY]
    notebook = {
        "cells": [
            _cell("markdown", sources[0], "contract"),
            _cell("code", sources[1], "user-config"),
            _cell("code", sources[2], "debug-config"),
            _cell("code", sources[3], "setup"),
            _cell("code", sources[4], "preflight"),
            _cell("code", sources[5], "run"),
            _cell("code", sources[6], "artifacts"),
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return notebook


def build_notebook(out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = out_dir / OUT_NOTEBOOK.name
    metadata_path = out_dir / OUT_METADATA.name
    notebook_path.write_text(json.dumps(_notebook(), ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    metadata = {
        "id": "trydotatwo/cayleypy-2xt4-checkpoint-beam-search",
        "title": "CayleyPy 2xT4 Checkpoint Beam Search",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return notebook_path, metadata_path


def main() -> None:
    print(*build_notebook())


if __name__ == "__main__":
    main()
