# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.18", "torch>=2.8", "numpy", "pandas", "kaggle", "kagglehub", "cmake", "ninja"]
# ///
"""Interactive Cube555 notebook for one physical Molab GPU."""
import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    # USER CONFIG: edit these values.
    MODEL_ROOT = Path("cube555_inputs/model")
    CHECKPOINT_PATH = MODEL_ROOT / "q555_f1_bell2k.pt"
    RESMLP_CHECKPOINT_PATH = MODEL_ROOT / "q555_2k_BEST.pt"
    COMPETITION_ROOT = Path("cube555_inputs/competition")
    PUZZLE_ID_START = 1020
    PUZZLE_ID_END = 1020
    BEAM_WIDTH = 3_100_000
    MAX_DEPTH = 140
    TRANSFORMER_WEIGHT = 0.8
    REFLECT_MODE = "off"  # off | after_original | only
    REFLECT_SOURCE_CSV = None
    SOLUTION_MODE = "collect"  # first | collect
    COLLECT_EXTRA_DEPTHS = 1
    MAX_COLLECTED_SOLUTIONS = 2_000
    return (BEAM_WIDTH, CHECKPOINT_PATH, COLLECT_EXTRA_DEPTHS, COMPETITION_ROOT,
            MAX_COLLECTED_SOLUTIONS, MAX_DEPTH, MODEL_ROOT, PUZZLE_ID_END,
            PUZZLE_ID_START, REFLECT_MODE, REFLECT_SOURCE_CSV,
            RESMLP_CHECKPOINT_PATH, SOLUTION_MODE, TRANSFORMER_WEIGHT)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # Cube555 — native beam search on Molab

    1. Attach a GPU in Molab. This notebook uses **one real GPU, one native rank**.
    2. Put your Kaggle API credential in Molab Secrets (`KAGGLE_API_TOKEN`),
       accept the Cube555 competition rules on Kaggle, then press **Prepare**.
       Alternatively upload model files and competition CSVs to the paths above.
    3. Press **Check GPU**, choose a short replay smoke or your configured search,
       then press **Run search**. Work stays in the foreground cell with live logs.

    Same Transformer/ResMLP blend and native search algorithm as the Kaggle notebook.
    BFS radius 5, outer batch 8192, model microbatch 128, one inference lane.
    These settings were measured on two T4s; **Molab speed is not yet measured**.
    The Kaggle 12-hour forecast does not transfer to this GPU.
    Maximum beam request 4,000,000; memory preflight can cap depth, never beam.
    Collect saves at most 2000 solutions, then continues the requested extra depths;
    a staging overflow logs dropped hits and continues. Every saved solution is replayed.

    Automatic Cloudflare publishing is disabled here: the current ingestion contract
    requires Kaggle execution provenance. Molab results are saved locally without
    labelling this run as a Kaggle run. Download the results ZIP after each run;
    do not assume sandbox-generated files survive session termination.
    """)
    return


@app.cell
def _(mo):
    prepare_button = mo.ui.run_button(label="Prepare model, data and pinned solver")
    check_button = mo.ui.run_button(label="Check GPU and build prerequisites")
    run_button = mo.ui.run_button(label="Run search")
    run_kind = mo.ui.dropdown(options=["Configured search", "Short replay smoke"],
                              value="Configured search", label="Run mode")
    mo.vstack([prepare_button, check_button, run_kind, run_button])
    return check_button, prepare_button, run_button, run_kind


@app.cell
def _(COMPETITION_ROOT, MODEL_ROOT, mo, prepare_button):
    mo.stop(not prepare_button.value)
    from pathlib import Path as _Path
    import json as _json
    import shutil as _shutil
    import subprocess as _subprocess
    import zipfile as _zipfile
    import hashlib as _hashlib

    _commit = "0157b6f461eff797bfa481b5593fe42d2238a2d6"
    _repo = _Path("cube555_solver").resolve()
    if not _repo.exists():
        _subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git", str(_repo)], check=True)
    _subprocess.run(["git", "fetch", "--depth", "1", "origin", _commit], cwd=_repo, check=True)
    _subprocess.run(["git", "checkout", "--detach", _commit], cwd=_repo, check=True)
    assert _subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo, text=True).strip() == _commit
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    COMPETITION_ROOT.mkdir(parents=True, exist_ok=True)
    _required_model = ["q555_f1_bell2k.pt", "q555_2k_BEST.pt", "piece_layout_555.json", "puzzle_info.json"]
    if not all((MODEL_ROOT / _name).is_file() for _name in _required_model):
        import kagglehub as _kh
        _download = _Path(_kh.dataset_download("trydotatwo/cube555-transformer-resmlp-artifacts"))
        for _name in _required_model:
            _matches = list(_download.rglob(_name))
            if len(_matches) != 1:
                raise RuntimeError(f"Expected one model bundle file: {_name}")
            _shutil.copy2(_matches[0], MODEL_ROOT / _name)
    if not all((COMPETITION_ROOT / _name).is_file() for _name in ["test.csv", "sample_submission.csv"]):
        from kaggle.api.kaggle_api_extended import KaggleApi as _KaggleApi
        _api = _KaggleApi()
        _api.authenticate()
        _api.competition_download_files("cayley-py-555-cube", path=str(COMPETITION_ROOT), quiet=True)
        with _zipfile.ZipFile(COMPETITION_ROOT / "cayley-py-555-cube.zip") as _archive:
            for _member in _archive.infolist():
                _target = (COMPETITION_ROOT / _member.filename).resolve()
                if not _target.is_relative_to(COMPETITION_ROOT.resolve()):
                    raise RuntimeError("Unsafe competition ZIP member")
            _archive.extractall(COMPETITION_ROOT)
    _manifest = {}
    for _path in [*(MODEL_ROOT / _name for _name in _required_model),
                  COMPETITION_ROOT / "test.csv", COMPETITION_ROOT / "sample_submission.csv"]:
        _digest = _hashlib.sha256()
        with _path.open("rb") as _file:
            for _block in iter(lambda: _file.read(1024 * 1024), b""):
                _digest.update(_block)
        _manifest[str(_path)] = _digest.hexdigest()
    _Path("cube555_input_manifest.json").write_text(_json.dumps(_manifest, indent=2))
    prepared_repo = _repo
    mo.md("Prepared pinned solver and hashed model/data files. Now check the GPU.")
    return (prepared_repo,)


@app.cell
def _(check_button, mo):
    mo.stop(not check_button.value)
    import torch as _torch
    import shutil as _shutil
    import subprocess as _subprocess
    if _torch.cuda.device_count() != 1:
        raise RuntimeError("Attach one GPU in Molab before running this notebook")
    for _tool in ["nvcc", "cmake", "ninja", "git", "nvidia-smi"]:
        if _shutil.which(_tool) is None:
            raise RuntimeError(f"Missing build prerequisite: {_tool}")
    _props = _torch.cuda.get_device_properties(0)
    print(dict(gpu=_props.name, vram_gib=round(_props.total_memory / 2**30, 2),
               cuda_arch=f"{_props.major}{_props.minor}", torch=_torch.__version__,
               torch_cuda=_torch.version.cuda), flush=True)
    _subprocess.run(["nvcc", "--version"], check=True)
    gpu_checked = True
    return (gpu_checked,)


@app.cell
def _(BEAM_WIDTH, CHECKPOINT_PATH, COLLECT_EXTRA_DEPTHS, COMPETITION_ROOT,
      MAX_COLLECTED_SOLUTIONS, MAX_DEPTH, MODEL_ROOT, PUZZLE_ID_END, PUZZLE_ID_START,
      REFLECT_MODE, REFLECT_SOURCE_CSV, RESMLP_CHECKPOINT_PATH, SOLUTION_MODE,
      TRANSFORMER_WEIGHT, gpu_checked, mo, prepared_repo, run_button, run_kind):
    mo.stop(not run_button.value or not gpu_checked)
    import subprocess as _subprocess
    import sys as _sys
    import time as _time
    import json as _json
    import shutil as _shutil
    from pathlib import Path as _Path
    import os as _os
    import signal as _signal
    _output = _Path("cube555_results") / _time.strftime("%Y%m%d_%H%M%S")
    _output.mkdir(parents=True, exist_ok=False)
    _sys.path.insert(0, str(prepared_repo))
    _pids = list(range(PUZZLE_ID_START, PUZZLE_ID_END + 1))
    if not _pids:
        raise ValueError("PUZZLE_ID_END must be >= PUZZLE_ID_START")
    _competition = COMPETITION_ROOT.resolve()
    _beam, _depth, _radius = BEAM_WIDTH, MAX_DEPTH, 5
    if run_kind.value == "Short replay smoke":
        from tools.cube555.smoke import make_fixture as _make_fixture
        _competition = (_output / "smoke_fixture").resolve()
        _make_fixture(MODEL_ROOT.resolve(), _competition)
        _pids, _beam, _depth, _radius = [0, 1, 2, 3], 4096, 6, 0
    _command = [_sys.executable, "-u", "-m", "tools.cube555.run",
        "--runtime-target", "molab-single-gpu", "--assets", str(MODEL_ROOT.resolve()),
        "--checkpoint", str(CHECKPOINT_PATH.resolve()),
        "--mlp-checkpoint", str(RESMLP_CHECKPOINT_PATH.resolve()),
        "--competition", str(_competition), "--output", str((_output / "run").resolve()),
        "--pids", *map(str, _pids), "--beam", str(_beam), "--depth", str(_depth),
        "--touch-radius", str(_radius), "--transformer-weight", str(TRANSFORMER_WEIGHT),
        "--b-micro", "8192", "--model-micro", "128", "--inference-concurrency", "1",
        "--reflect-mode", REFLECT_MODE, "--solution-mode", SOLUTION_MODE,
        "--collect-extra-depths", str(COLLECT_EXTRA_DEPTHS),
        "--max-collected-solutions", str(MAX_COLLECTED_SOLUTIONS)]
    (_output / "provenance.json").write_text(_json.dumps(dict(
        platform="molab", world_size=1,
        solver_commit=_subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=prepared_repo, text=True).strip(),
        beam=_beam, max_depth=_depth, touch_bfs_radius=_radius,
        transformer_weight=TRANSFORMER_WEIGHT, model_micro=128,
        inference_concurrency=1, solution_mode=SOLUTION_MODE,
        collect_extra_depths=COLLECT_EXTRA_DEPTHS,
        max_collected_solutions=MAX_COLLECTED_SOLUTIONS), indent=2))
    _shutil.copy2("cube555_input_manifest.json", _output / "input_manifest.json")
    if REFLECT_SOURCE_CSV is not None:
        _command += ["--reflect-source-csv", str(_Path(REFLECT_SOURCE_CSV).resolve())]
    with (_output / "launcher.log").open("w") as _log:
        _process = _subprocess.Popen(_command, cwd=prepared_repo, stdout=_subprocess.PIPE,
                                    stderr=_subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=True)
        try:
            for _line in _process.stdout:
                print(_line, end="", flush=True)
                _log.write(_line)
                _log.flush()
            _code = _process.wait()
        except BaseException:
            _os.killpg(_process.pid, _signal.SIGTERM)
            try:
                _process.wait(timeout=15)
            except _subprocess.TimeoutExpired:
                _os.killpg(_process.pid, _signal.SIGKILL)
                _process.wait()
            raise
        finally:
            # Preserve partial logs/solutions on ordinary search failure too.
            _archive = _shutil.make_archive(str(_output), "zip", root_dir=_output)
            print(f"Results archive: {_archive}", flush=True)
    if _code:
        raise RuntimeError(f"Search exited {_code}; inspect {_output}/launcher.log")
    _summary = _json.loads((_output / "run/run_summary.json").read_text())
    print(_json.dumps(_summary, indent=2))
    if run_kind.value == "Short replay smoke":
        import pandas as _pd
        for _pid in _pids:
            _solutions = _pd.read_csv(_output / f"run/puzzle-{_pid}/solutions/solutions.csv")
            assert len(_solutions) and _solutions["valid"].all()
        print("PASS: four Cube555 scrambles solved and replayed on one native rank")
    mo.md(f"Completed. Download **{_archive}** using the Molab file browser.")
    return


if __name__ == "__main__":
    app.run()
