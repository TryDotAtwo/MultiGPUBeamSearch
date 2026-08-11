from __future__ import annotations

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
    # CayleyPy checkpoint-only beam search for Molab

    Edit the USER CONFIG cell, upload/expose the competition and checkpoint, then run.

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
    REPOSITORY_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
    REPOSITORY_BRANCH = "molab/notebooks"
    SOLVER_COMMIT = "141311a55fe5fbcf23a8b60fbea85dace95e15cb"
    MODEL_SOURCE_KIND = "checkpoint"
    MODEL_DTYPE = "auto"
    CHECKPOINT_FORMAT = "auto"

    COMPETITION = "replace-with-competition"
    COMPETITION_SOURCE_KIND = "kaggle_competition" if False else "local_directory"
    COMPETITION_SOURCE = "replace-with-competition" if False else "/tmp/uploaded-competition"
    MODEL_ASSET_KIND = "local_checkpoint"
    MODEL_ASSET_REF = "/tmp/uploaded-model/checkpoint.pth"
    CHECKPOINT_GLOB = "*.pth"
    CHECKPOINT_METADATA_GLOB = None
    CHECKPOINT_GENERATOR_GLOB = None

    PUZZLE_ID_START = 0
    PUZZLE_ID_END = 0
    BEAM_WIDTH = 2**16
    MAX_DEPTH = 100
    REFLECT_MODE = "off"
    REFLECT_SOURCE_CSV = None
    SOLUTION_MODE = "first"
    COLLECT_UNTIL_DEPTH = MAX_DEPTH
    MAX_COLLECTED_SOLUTIONS = 100
    TOUCH_BFS_RADIUS = 4

    AUTHOR_NAME = "replace-with-author"
    PUBLISH_RESULTS = True
    RESULTS_INGEST_URL = "https://cayleypy-results-ingest-staging.tupa-expert.workers.dev/v1/results"

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
    molab_user_config = {name: config_scope[name] for name in config_names}
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
        serializable = {key: str(value) if isinstance(value, pathlib.Path) else value
                        for key, value in config.items() if not key.startswith("_")}
        config_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")

        command = [sys.executable, "-m", "tools.run_cayleypy_molab",
                   "--notebook-config", str(config_path), "--output-dir", str(output_dir)]
        print({"command": command, "output_dir": str(output_dir), "log_path": str(log_path)})
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
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
        print({"return_code": return_code, "summary": summary})
        if return_code != 0:
            raise RuntimeError(f"Molab solve failed with code {return_code}; see {log_path}")
        return summary

    molab_run_summary = run_molab_cayleypy_v1(molab_user_config)
    return (molab_run_summary,)


@app.cell
def _(mo, molab_run_summary):
    mo.md(f"""
    ## Run complete

    ```json
    {molab_run_summary}
    ```

    Artifacts: `/tmp/cayleypy_molab/output/run_summary.json`,
    `publish_status.json`, `submission.csv`, `solutions/`, and `logs/`.
    """)
    return


if __name__ == "__main__":
    app.run()
