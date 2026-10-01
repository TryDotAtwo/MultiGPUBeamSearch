# /// script
# requires-python = ">=3.11"
# dependencies = ["marimo>=0.18", "torch==2.11.0", "numpy", "pandas", "cmake", "ninja", "cuda-toolkit[nvcc,crt,nvvm,cudart,cccl,culibos]==13.0.2"]
# ///
"""Interactive Cube555 notebook for one physical Molab GPU."""
import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    # USER CONFIG: edit these values.
    INPUT_BUNDLE_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch/releases/download/cube555-inputs-20261001/cube555-inputs.zip"
    CHECKPOINT_FILENAME = "q555_f1_bell2k.pt"  # Transformer
    RESMLP_CHECKPOINT_FILENAME = "q555_2k_BEST.pt"  # ResMLP
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
    AUTHOR_NAME = "Molab Cube555 participant"  # your public name
    return (AUTHOR_NAME, BEAM_WIDTH, CHECKPOINT_FILENAME, COLLECT_EXTRA_DEPTHS,
            MAX_COLLECTED_SOLUTIONS, MAX_DEPTH, INPUT_BUNDLE_URL, PUZZLE_ID_END,
            PUZZLE_ID_START, REFLECT_MODE, REFLECT_SOURCE_CSV,
            RESMLP_CHECKPOINT_FILENAME, SOLUTION_MODE, TRANSFORMER_WEIGHT)


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(CHECKPOINT_FILENAME, INPUT_BUNDLE_URL, RESMLP_CHECKPOINT_FILENAME):
    from pathlib import Path as _cell2_Path
    # Cache paths are internal; Run All downloads the public bundle automatically.
    MODEL_ROOT = _cell2_Path("cube555_inputs/model")
    CHECKPOINT_PATH = MODEL_ROOT / CHECKPOINT_FILENAME
    RESMLP_CHECKPOINT_PATH = MODEL_ROOT / RESMLP_CHECKPOINT_FILENAME
    COMPETITION_ROOT = _cell2_Path("cube555_inputs/competition")
    return CHECKPOINT_PATH, COMPETITION_ROOT, MODEL_ROOT, RESMLP_CHECKPOINT_PATH


@app.cell
def _(mo):
    mo.md("""
    # Cube555 — Run All on Molab

    Select one GPU in Molab, edit the first cell if needed, then **Run All**.
    The following cells download public inputs from GitHub without credentials,
    verify hashes, build the pinned native solver and run the search in the foreground.
    Model weights are the Transformer/ResMLP blend restored from Artgor's notebook.
    BFS radius 5; collect keeps at most 2000 solutions and searches the requested
    number of additional depths after the first solution. Every saved path is replayed.
    Download the results ZIP before ending the sandbox session.
    Molab speed has not yet been measured; the Kaggle 12-hour estimate does not apply.
    """)
    return


@app.cell
def _(CHECKPOINT_FILENAME, COMPETITION_ROOT, INPUT_BUNDLE_URL, MODEL_ROOT,
      RESMLP_CHECKPOINT_FILENAME, mo):
    from pathlib import Path as _cell4_Path
    import json as _cell4_json
    import shutil as _cell4_shutil
    import subprocess as _cell4_subprocess
    import zipfile as _cell4_zipfile
    import hashlib as _cell4_hashlib

    _commit = "856ab0aa0222900547c81fd2f42b072747af77a1"
    _repo = _cell4_Path("cube555_solver").resolve()
    if _repo.exists() and not (_repo / ".git").is_dir():
        # Molab persistence may restore source files without hidden Git metadata.
        # Keep that snapshot and recreate the verifiable pinned checkout.
        import time as _cell4_time
        _repo.rename(_repo.with_name(_repo.name + f".snapshot-{_cell4_time.time_ns()}"))
    if not _repo.exists():
        _cell4_subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
                        "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git", str(_repo)], check=True)
    _cell4_subprocess.run(["git", "fetch", "--depth", "1", "origin", _commit], cwd=_repo, check=True)
    _cell4_subprocess.run(["git", "checkout", "--detach", _commit], cwd=_repo, check=True)
    assert _cell4_subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_repo, text=True).strip() == _commit
    import urllib.request as _cell4_urlrequest
    _bundle = _cell4_Path("cube555_inputs.zip")
    _expected = "49d61a127540cd197db9bf75aacad5712a7ab54d3fbcb39bf2da9ae9bb477286"
    def _file_hash(_file_path):
        _digest = _cell4_hashlib.sha256()
        with _file_path.open("rb") as _file:
            for _block in iter(lambda: _file.read(1024 * 1024), b""):
                _digest.update(_block)
        return _digest.hexdigest()
    if not _bundle.is_file() or _file_hash(_bundle) != _expected:
        print("Downloading public Cube555 inputs from GitHub", flush=True)
        _partial = _bundle.with_suffix(".partial")
        with _cell4_urlrequest.urlopen(INPUT_BUNDLE_URL, timeout=120) as _response, _partial.open("wb") as _file:
            _cell4_shutil.copyfileobj(_response, _file)
        if _file_hash(_partial) != _expected:
            raise RuntimeError("Input bundle SHA-256 mismatch")
        _partial.replace(_bundle)
    _cache = _cell4_Path("cube555_inputs").resolve()
    _cache.mkdir(exist_ok=True)
    with _cell4_zipfile.ZipFile(_bundle) as _archive:
        for _member in _archive.infolist():
            if not (_cache / _member.filename).resolve().is_relative_to(_cache):
                raise RuntimeError("Unsafe input ZIP member")
        _archive.extractall(_cache)
    _bundle_manifest = _cell4_json.loads((_cache / "manifest.json").read_text())
    for _name, _entry in _bundle_manifest.items():
        _file_path = _cache / _name
        if _file_path.stat().st_size != _entry["bytes"] or _file_hash(_file_path) != _entry["sha256"]:
            raise RuntimeError(f"Input verification failed: {_name}")
    _required_model = [CHECKPOINT_FILENAME, RESMLP_CHECKPOINT_FILENAME,
                       "piece_layout_555.json", "puzzle_info.json"]
    _manifest = {}
    for _path in [*(MODEL_ROOT / _name for _name in _required_model),
                  COMPETITION_ROOT / "test.csv", COMPETITION_ROOT / "sample_submission.csv"]:
        _digest = _cell4_hashlib.sha256()
        with _path.open("rb") as _file:
            for _block in iter(lambda: _file.read(1024 * 1024), b""):
                _digest.update(_block)
        _manifest[str(_path)] = _digest.hexdigest()
    _cell4_Path("cube555_input_manifest.json").write_text(_cell4_json.dumps(_manifest, indent=2))
    prepared_repo = _repo
    mo.md("Prepared pinned solver and hashed model/data files. Inputs ready; checking the GPU automatically.")
    return (prepared_repo,)


@app.cell
def _(prepared_repo):
    import torch as _cell5_torch
    import shutil as _cell5_shutil
    import subprocess as _cell5_subprocess
    import os as _cell5_os
    import sys as _cell5_sys
    from pathlib import Path as _cell5_Path
    # Molab installs PEP 723 dependencies into its kernel virtual environment.
    # Native tools and torchrun must use the same environment as this kernel.
    _cell5_os.environ["PATH"] = str(_cell5_Path(_cell5_sys.executable).parent) + _cell5_os.pathsep + _cell5_os.environ.get("PATH", "")
    import importlib.metadata as _cell5_metadata
    _compiler = _cell5_metadata.distribution("nvidia-cuda-nvcc")
    if _compiler.version != "13.0.88":
        raise RuntimeError("Expected the declared CUDA Toolkit 13.0.2 compiler")
    _cuda_roots = list(dict.fromkeys([
        _cell5_Path(_compiler.locate_file("nvidia/cu13")),
        _cell5_Path(_cell5_torch.__file__).resolve().parents[1] / "nvidia/cu13",
    ]))
    # Compose the wheel SDK in an owned temporary directory: pinned compiler
    # first, then PyTorch's CUDA libraries. Never modify package installations.
    _sdk = _cell5_Path("/tmp/cube555_cuda13_sdk")
    _sdk.mkdir(parents=True, exist_ok=True)
    for _root in _cuda_roots:
        for _file in _root.rglob("*"):
            if not _file.is_file():
                continue
            _target = _sdk / _file.relative_to(_root)
            if not _target.exists() and not _target.is_symlink():
                _target.parent.mkdir(parents=True, exist_ok=True)
                _target.symlink_to(_file.resolve())
    if not (_sdk / "lib64").exists():
        (_sdk / "lib64").symlink_to(_sdk / "lib", target_is_directory=True)
    for _component in ["cub", "thrust", "cuda"]:
        _nested = _sdk / "include/cccl" / _component
        _flat = _sdk / "include" / _component
        if _nested.exists() and not _flat.exists():
            _flat.symlink_to(_nested, target_is_directory=True)
    for _library in (_sdk / "lib").glob("lib*.so.*"):
        _alias = _library.with_name(_library.name.split(".so.")[0] + ".so")
        if not _alias.exists() and not _alias.is_symlink():
            _alias.symlink_to(_library)
    _cell5_os.environ["CUDA_HOME"] = _cell5_os.environ["CUDAToolkit_ROOT"] = str(_sdk)
    _cell5_os.environ["CUDACXX"] = str(_sdk / "bin/nvcc")
    _cell5_os.environ["PATH"] = str(_sdk / "bin") + _cell5_os.pathsep + _cell5_os.environ["PATH"]
    _cell5_os.environ["LD_LIBRARY_PATH"] = str(_sdk / "lib") + _cell5_os.pathsep + _cell5_os.environ.get("LD_LIBRARY_PATH", "")
    if _cell5_torch.cuda.device_count() != 1:
        raise RuntimeError("Attach one GPU in Molab before running this notebook")
    for _tool in ["nvcc", "cmake", "ninja", "git", "nvidia-smi"]:
        if _cell5_shutil.which(_tool) is None:
            raise RuntimeError(f"Missing build prerequisite: {_tool}")
    _props = _cell5_torch.cuda.get_device_properties(0)
    print(dict(gpu=_props.name, vram_gib=round(_props.total_memory / 2**30, 2),
               cuda_arch=f"{_props.major}{_props.minor}", torch=_cell5_torch.__version__,
               torch_cuda=_cell5_torch.version.cuda), flush=True)
    _cell5_subprocess.run(["nvcc", "--version"], check=True)
    gpu_checked = True
    return (gpu_checked,)


@app.cell
def _(AUTHOR_NAME, BEAM_WIDTH, CHECKPOINT_PATH, COLLECT_EXTRA_DEPTHS, COMPETITION_ROOT,
      MAX_COLLECTED_SOLUTIONS, MAX_DEPTH, MODEL_ROOT, PUZZLE_ID_END, PUZZLE_ID_START,
      REFLECT_MODE, REFLECT_SOURCE_CSV, RESMLP_CHECKPOINT_PATH, SOLUTION_MODE,
      TRANSFORMER_WEIGHT, gpu_checked, mo, prepared_repo):
    import subprocess as _cell6_subprocess
    import sys as _cell6_sys
    import time as _cell6_time
    import json as _cell6_json
    import shutil as _cell6_shutil
    from pathlib import Path as _cell6_Path
    import os as _cell6_os
    import signal as _cell6_signal
    import queue as _cell6_queue
    import threading as _cell6_threading
    import hashlib as _cell6_hashlib
    import torch as _cell6_torch
    if not gpu_checked or _cell6_torch.cuda.device_count() != 1:
        raise RuntimeError("Select one GPU in Molab, then Run All")
    _output = _cell6_Path("cube555_results") / _cell6_time.strftime("%Y%m%d_%H%M%S")
    _output.mkdir(parents=True, exist_ok=False)
    _cell6_sys.path.insert(0, str(prepared_repo))
    _pids = list(range(PUZZLE_ID_START, PUZZLE_ID_END + 1))
    if not _pids:
        raise ValueError("PUZZLE_ID_END must be >= PUZZLE_ID_START")
    _competition = COMPETITION_ROOT.resolve()
    _beam, _depth, _radius = BEAM_WIDTH, MAX_DEPTH, 5
    _command = [_cell6_sys.executable, "-u", "-m", "tools.cube555.run",
        "--runtime-target", "molab-single-gpu", "--assets", str(MODEL_ROOT.resolve()),
        "--checkpoint", str(CHECKPOINT_PATH.resolve()),
        "--mlp-checkpoint", str(RESMLP_CHECKPOINT_PATH.resolve()),
        "--competition", str(_competition), "--output", str((_output / "run").resolve()),
        "--pids", *map(str, _pids), "--beam", str(_beam), "--depth", str(_depth),
        "--touch-radius", str(_radius), "--transformer-weight", str(TRANSFORMER_WEIGHT),
        "--b-micro", "8192",
        "--reflect-mode", REFLECT_MODE, "--solution-mode", SOLUTION_MODE,
        "--collect-extra-depths", str(COLLECT_EXTRA_DEPTHS),
        "--max-collected-solutions", str(MAX_COLLECTED_SOLUTIONS)]
    _commit = _cell6_subprocess.check_output(["git", "rev-parse", "HEAD"],
                                    cwd=prepared_repo, text=True).strip()
    _publication = _output / "publication.json"
    _publication.write_text(_cell6_json.dumps(dict(
        author_name=AUTHOR_NAME, competition="cayley-py-555-cube",
        solver_commit=_commit,
        molab_notebook_url="https://molab.marimo.io/notebooks/nb_TYNXg2wyehhgBDcTVRRKzQ",
        molab_notebook_sha256=_cell6_hashlib.sha256(_cell6_Path(__file__).read_bytes()).hexdigest()), indent=2))
    _command += ["--publish", "--publication-json", str(_publication.resolve())]
    from tools.cube555.run import runtime_plan as _cell6_runtime_plan
    _selected_plan = _cell6_runtime_plan(_beam, world_size=1)
    (_output / "provenance.json").write_text(_cell6_json.dumps(dict(
        platform="molab", world_size=1,
        solver_commit=_cell6_subprocess.check_output(["git", "rev-parse", "HEAD"],
            cwd=prepared_repo, text=True).strip(),
        beam=_beam, max_depth=_depth, touch_bfs_radius=_radius,
        transformer_weight=TRANSFORMER_WEIGHT,
        model_micro=_selected_plan.runtime['model_micro'],
        inference_concurrency=_selected_plan.runtime['stream1_concurrency'],
        profile_note=_selected_plan.cross_puzzle_profile_note, solution_mode=SOLUTION_MODE,
        collect_extra_depths=COLLECT_EXTRA_DEPTHS,
        max_collected_solutions=MAX_COLLECTED_SOLUTIONS), indent=2))
    _cell6_shutil.copy2("cube555_input_manifest.json", _output / "input_manifest.json")
    if REFLECT_SOURCE_CSV is not None:
        _command += ["--reflect-source-csv", str(_cell6_Path(REFLECT_SOURCE_CSV).resolve())]
    with (_output / "launcher.log").open("w") as _log:
        _process = _cell6_subprocess.Popen(_command, cwd=prepared_repo, stdout=_cell6_subprocess.PIPE,
                                    stderr=_cell6_subprocess.STDOUT, text=True, bufsize=1,
                                    start_new_session=True)
        # A process group lets interruption stop torchrun children too. The parent
        # still reads and waits here; this is not a detached/background launch.
        _lines = _cell6_queue.Queue()
        def _read_child_output():
            try:
                for _child_line in _process.stdout:
                    _lines.put(_child_line)
            finally:
                _lines.put(None)
        _reader = _cell6_threading.Thread(target=_read_child_output, daemon=True)
        _reader.start()
        _started = _cell6_time.monotonic()
        _pipe_closed = False
        try:
            while True:
                try:
                    _line = _lines.get(timeout=15)
                except _cell6_queue.Empty:
                    _heartbeat = f"[cell-progress] pid={_process.pid} elapsed={_cell6_time.monotonic()-_started:.0f}s returncode={_process.poll()}\n"
                    print(_heartbeat, end="", flush=True)
                    _log.write(_heartbeat)
                    _log.flush()
                    if _process.poll() is not None:
                        # A surviving descendant can retain stdout after launcher
                        # exit. Never wait indefinitely for its inherited pipe.
                        try:
                            _cell6_os.killpg(_process.pid, _cell6_signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        _reader.join(timeout=15)
                        if _reader.is_alive():
                            try:
                                _cell6_os.killpg(_process.pid, _cell6_signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                        break
                    continue
                if _line is None:
                    _pipe_closed = True
                    break
                print(_line, end="", flush=True)
                _log.write(_line)
                _log.flush()
            _code = _process.wait()
            _reader.join(timeout=1)
            while not _lines.empty():
                _line = _lines.get_nowait()
                if _line is None:
                    _pipe_closed = True
                else:
                    print(_line, end="", flush=True)
                    _log.write(_line)
            _log.flush()
            _terminal = dict(returncode=_code, elapsed_seconds=_cell6_time.monotonic()-_started,
                             stdout_pipe_closed=_pipe_closed)
            (_output / "launcher_terminal.json").write_text(_cell6_json.dumps(_terminal, indent=2))
            print("[cell-terminal] "+_cell6_json.dumps(_terminal), flush=True)
            if not _pipe_closed:
                raise RuntimeError("Launcher exited but descendant stdout did not close; own process group cleanup requested")
        except BaseException as _error:
            try:
                _cell6_os.killpg(_process.pid, _cell6_signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                _process.wait(timeout=15)
            except _cell6_subprocess.TimeoutExpired:
                _cell6_os.killpg(_process.pid, _cell6_signal.SIGKILL)
                _process.wait()
            (_output / "launcher_terminal.json").write_text(_cell6_json.dumps(dict(
                returncode=_process.returncode, exception_type=type(_error).__name__,
                elapsed_seconds=_cell6_time.monotonic()-_started,
                stdout_pipe_closed=_pipe_closed), indent=2))
            raise
        finally:
            # Preserve partial logs/solutions on ordinary search failure too.
            _archive = _cell6_shutil.make_archive(str(_output), "zip", root_dir=_output)
            print(f"Results archive: {_archive}", flush=True)
    if _code:
        raise RuntimeError(f"Search exited {_code}; inspect {_output}/launcher.log")
    _summary = _cell6_json.loads((_output / "run/run_summary.json").read_text())
    print(_cell6_json.dumps(_summary, indent=2))
    mo.md(f"Completed. Download **{_archive}** using the Molab file browser.")
    return


if __name__ == "__main__":
    app.run()
