#!/usr/bin/env python3
"""Generate one universal and four ready-to-run Molab marimo notebooks."""
from __future__ import annotations

from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "molab" / "cayleypy-notebooks"
REPOSITORY = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
REPOSITORY_BRANCH = "molab/notebooks"
SOLVER_COMMIT = "d156c4a1c62e5c49d9db78c0b37a216aeb1db460"
INGEST_URL = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v1/results"

EXAMPLES: dict[str, dict[str, Any]] = {
    "444": {
        "title": "CayleyPy Molab — Cube 4x4x4 example",
        "competition": "cayley-py-444-cube",
        "puzzle_id": 0,
        "asset_kind": "kaggle_dataset",
        "asset_ref": "trydotatwo/cube4-full-transformer-inference",
        "checkpoint_glob": "model.pth",
        "metadata_glob": "model/model.json",
        "generator_glob": "generators/p002.json",
        "output_dim": 24,
    },
    "megaminx": {
        "title": "CayleyPy Molab — Megaminx example",
        "competition": "cayley-py-megaminx",
        "puzzle_id": 10,
        "asset_kind": "kaggle_model",
        "asset_ref": "trydotatwo/megaminx-output24-p900-t000-q-sym/PyTorch/default/1",
        "checkpoint_glob": "*.pth",
        "metadata_glob": None,
        "generator_glob": None,
        "output_dim": 24,
    },
    "ihes": {
        "title": "CayleyPy Molab — IHES Cube example",
        "competition": "cayleypy-ihes-cube",
        "puzzle_id": 1,
        "asset_kind": "kaggle_model",
        "asset_ref": "arabidopsisthalian/ihes-e08192/PyTorch/default/1",
        "checkpoint_glob": "*.pth",
        "metadata_glob": None,
        "generator_glob": None,
        "output_dim": 1,
    },
    "tetraminx": {
        "title": "CayleyPy Molab — Professor Tetraminx example",
        "competition": "cayley-py-professor-tetraminx-solve-optimally",
        "puzzle_id": 0,
        "asset_kind": "kaggle_notebook_output",
        "asset_ref": "rokham/cayleypy-cube-train-and-solve",
        "checkpoint_glob": "p888-t000_1765097793_e01024.pth",
        "metadata_glob": "logs/model_p888-t000_1765097793.json",
        "generator_glob": "generators/p888.json",
        "output_dim": 1,
        "max_depth": 60,
    },
}


def _literal(value: object) -> str:
    return repr(value)


def _source(name: str, spec: dict[str, Any] | None) -> str:
    title = "CayleyPy checkpoint-only beam search for Molab" if spec is None else spec["title"]
    competition = "replace-with-competition" if spec is None else spec["competition"]
    puzzle_id = 0 if spec is None else int(spec["puzzle_id"])
    max_depth = 100 if spec is None else int(spec.get("max_depth", 100))
    asset_kind = "local_checkpoint" if spec is None else spec["asset_kind"]
    asset_ref = "/tmp/uploaded-model/checkpoint.pth" if spec is None else spec["asset_ref"]
    checkpoint_glob = "*.pth" if spec is None else spec["checkpoint_glob"]
    metadata_glob = None if spec is None else spec.get("metadata_glob")
    generator_glob = None if spec is None else spec.get("generator_glob")
    author = "replace-with-author" if spec is None else f"molab-example-{name}"
    competition_root = "/tmp/uploaded-competition" if spec is None else "auto"
    example_note = (
        "Edit the USER CONFIG cell, upload/expose the competition and checkpoint, then run."
        if spec is None else
        "Ready-to-run public example. Kaggle assets are downloaded by kagglehub inside Molab."
    )
    return f'''from __future__ import annotations

import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # {title}

    {example_note}

    Supported checkpoints: batchnorm-folded MLP, resmlp-layernorm MLP, and the
    supported piece Transformer bundle. Output head must be `1` or the puzzle's
    move count. Dtype and checkpoint format are detected automatically.

    The notebook uses the attached Molab GPU count, keeps all local artifacts,
    and publishes validated results to the token-free Cloudflare ingest service
    on a best-effort basis.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    # USER CONFIG
    PLATFORM = "molab"
    REPOSITORY_URL = "{REPOSITORY}"
    REPOSITORY_BRANCH = "{REPOSITORY_BRANCH}"
    SOLVER_COMMIT = "{SOLVER_COMMIT}"
    MODEL_SOURCE_KIND = "checkpoint"
    MODEL_DTYPE = "auto"
    CHECKPOINT_FORMAT = "auto"

    COMPETITION = "{competition}"
    COMPETITION_SOURCE_KIND = "kaggle_competition" if {str(spec is not None)} else "local_directory"
    COMPETITION_SOURCE = "{competition}" if {str(spec is not None)} else "{competition_root}"
    MODEL_ASSET_KIND = "{asset_kind}"
    MODEL_ASSET_REF = "{asset_ref}"
    CHECKPOINT_GLOB = "{checkpoint_glob}"
    CHECKPOINT_METADATA_GLOB = {_literal(metadata_glob)}
    CHECKPOINT_GENERATOR_GLOB = {_literal(generator_glob)}

    PUZZLE_ID_START = {puzzle_id}
    PUZZLE_ID_END = {puzzle_id}
    BEAM_WIDTH = 2**16
    MAX_DEPTH = {max_depth}
    REFLECT_MODE = "off"
    REFLECT_SOURCE_CSV = None
    SOLUTION_MODE = "first"
    COLLECT_UNTIL_DEPTH = MAX_DEPTH
    MAX_COLLECTED_SOLUTIONS = 100
    TOUCH_BFS_RADIUS = 4

    AUTHOR_NAME = "{author}"
    PUBLISH_RESULTS = True
    RESULTS_INGEST_URL = "{INGEST_URL}"

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

    OUTPUT_DIR = Path("/tmp/cayleypy_molab/output")
    LOG_PATH = Path("/tmp/cayleypy_molab/molab-run.log")
    config_names = (
        "PLATFORM", "REPOSITORY_URL", "REPOSITORY_BRANCH", "SOLVER_COMMIT",
        "MODEL_SOURCE_KIND", "MODEL_DTYPE", "CHECKPOINT_FORMAT",
        "COMPETITION", "COMPETITION_SOURCE_KIND", "COMPETITION_SOURCE",
        "MODEL_ASSET_KIND", "MODEL_ASSET_REF", "CHECKPOINT_GLOB",
        "CHECKPOINT_METADATA_GLOB", "CHECKPOINT_GENERATOR_GLOB",
        "PUZZLE_ID_START", "PUZZLE_ID_END", "BEAM_WIDTH", "MAX_DEPTH",
        "REFLECT_MODE", "REFLECT_SOURCE_CSV", "SOLUTION_MODE",
        "COLLECT_UNTIL_DEPTH", "MAX_COLLECTED_SOLUTIONS", "TOUCH_BFS_RADIUS",
        "AUTHOR_NAME", "PUBLISH_RESULTS", "RESULTS_INGEST_URL",
        "ENABLE_DEBUG", "ENABLE_DEPTH_LOGS", "ENABLE_DEBUG_LOGS",
        "DEBUG_STREAM_TIMING", "DEBUG_INFERENCE_TRACE", "DEBUG_PATH_TRACE",
        "DEBUG_FINAL_VALIDATE", "DEBUG_FINAL_EXCHANGE_TRACE",
        "DEBUG_FINAL_HISTOGRAM_TRACE", "DEBUG_STREAM4_HISTOGRAM_TRACE",
        "DEBUG_DEPTH_FLOW_TRACE", "DEBUG_PIPELINE_STATS", "DEPTH_LOG_EVERY",
        "PUZZLE_LOG_EVERY", "OUTPUT_DIR", "LOG_PATH",
    )
    config_scope = locals()
    molab_user_config = {{name: config_scope[name] for name in config_names}}
    return (molab_user_config,)


@app.cell
def _(molab_user_config):
    def run_molab_cayleypy_v1(config):
        import json
        import pathlib
        import subprocess
        import sys

        workspace = pathlib.Path("/tmp/cayleypy_molab")
        repo = workspace / "repo"
        workspace.mkdir(parents=True, exist_ok=True)
        log_path = pathlib.Path(config["LOG_PATH"])
        output_dir = pathlib.Path(config["OUTPUT_DIR"])
        output_dir.mkdir(parents=True, exist_ok=True)

        if not repo.joinpath(".git").exists():
            subprocess.run([
                "git", "clone", "--filter=blob:none", "--branch",
                config["REPOSITORY_BRANCH"], config["REPOSITORY_URL"], str(repo),
            ], check=True)
        subprocess.run(["git", "fetch", "--depth", "1", "origin", config["SOLVER_COMMIT"]], cwd=repo, check=True)
        subprocess.run(["git", "checkout", "--detach", "FETCH_HEAD"], cwd=repo, check=True)

        config_path = workspace / "notebook_config.json"
        serializable = {{key: str(value) if isinstance(value, pathlib.Path) else value
                        for key, value in config.items() if not key.startswith("_")}}
        config_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

        command = [sys.executable, "-m", "tools.run_cayleypy_molab",
                   "--notebook-config", str(config_path), "--output-dir", str(output_dir)]
        print({{"command": command, "output_dir": str(output_dir), "log_path": str(log_path)}})
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                if line.startswith(("molab_preflight=", "depth_start=", "depth_done=",
                                    "puzzle_solved=", "publish_status=")):
                    print(line, end="", flush=True)
            return_code = process.wait()
        summary_path = output_dir / "run_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {{}}
        print({{"return_code": return_code, "summary": summary}})
        if return_code != 0:
            raise RuntimeError(f"Molab solve failed with code {{return_code}}; see {{log_path}}")
        return summary

    molab_run_summary = run_molab_cayleypy_v1(molab_user_config)
    return (molab_run_summary,)


@app.cell
def _(mo, molab_run_summary):
    mo.md(f"""
    ## Run complete

    ```json
    {{molab_run_summary}}
    ```

    Artifacts: `/tmp/cayleypy_molab/output/run_summary.json`,
    `publish_status.json`, `submission.csv`, `solutions/`, and `logs/`.
    """)
    return


if __name__ == "__main__":
    app.run()
'''


def build_all(out_root: Path = OUT_ROOT) -> dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}
    main_path = out_root / "cayleypy_molab.py"
    main_path.write_text(_source("main", None), encoding="utf-8")
    outputs["main"] = main_path
    for name, spec in EXAMPLES.items():
        path = out_root / f"cayleypy_molab_{name}.py"
        path.write_text(_source(name, spec), encoding="utf-8")
        outputs[name] = path
    return outputs


if __name__ == "__main__":
    for generated in build_all().values():
        print(generated)
