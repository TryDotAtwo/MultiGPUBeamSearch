"""Build an auditable Kaggle notebook pinned to an immutable solver commit."""
import argparse
import ast
import json
from pathlib import Path
import re


def build(commit: str, output: Path, smoke: bool):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('solver commit must be a full Git SHA')
    cells = []
    def cell(source, kind='code'):
        result = dict(id=f'cube555-{len(cells)}', cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
        if kind == 'code':
            ast.parse(source)
            result.update(execution_count=None, outputs=[])
        cells.append(result)
    cell('''# CayleyPy Cube555 - native beam on 2xT4

Artgor PieceTransformerQ555 + ResMLPQ, parent-Q blend **0.8 / 0.2**.
The two ranks share one native beam. This is the existing MultiGPUBeamSearch
pipeline with a Cube555 LibTorch scoring module, not the TPU/JAX search engine.
FP16 on T4 can choose different paths from BF16 on TPU. The reported 102/100
results have not been reproduced by this notebook.

Attach `artgor/cube555-tpu-artifacts` and `cayley-py-555-cube`.
Enter `BEAM_WIDTH` directly. Existing p22-p26 pipeline settings are selected
automatically; the width is never silently reduced. Default: `2**25`.
Only compatible Cube555 PieceTransformerQ555 / ResMLPQ weights are supported.
Keep layout and puzzle_info.json beside the model bundle.
Cube555 capacity is checked by the native
150/160-byte memory planner; Cube4's measured ceiling is not a Cube555 claim.
`B_MICRO=8192` is the outer parent transaction. `MODEL_MICRO` independently limits
one LibTorch forward. The original Transformer registry selects shards, Stream4
buffers and final exchange chunks from the requested width.
History uses at most 24 GB RAM, preserves at least 6 GB of available RAM
for runtime overhead, and uses 50 GiB scratch disk. Default depth is 140.
Collect staging is bounded by MAX_COLLECTED_SOLUTIONS * effective MAX_DEPTH
records (40 bytes each per GPU), capped by the layer candidate count.
A layer exceeding this budget logs a warning and dropped-hit counts, saves
the stored hits, and continues search. performance.json records truncation.
The native total-memory preflight still applies to the beam and all other buffers. Preflight explicitly reports
requested/effective MAX_DEPTH; beam is never silently reduced to fit history.
All returned solutions are replayed. Both rank logs and provenance are retained.
`gpu_samples.csv` samples both GPUs every second, and `performance.json` records
memory high-water samples and native per-depth timings. Sampling can miss brief peaks.
Result publication is on by default and applies only to locally replayed solutions.
Each puzzle retains `publish_status.json` and the exact `results-*.json.gz` request.
HTTP acceptance is not a claim that the GitHub promotion has completed.
When forking, update the Kaggle owner, slug and saved version in the config.
The notebook-source hash is derived from actual running cells, not a placeholder.
''', 'markdown')
    cell(f'''from pathlib import Path

# USER CONFIG: change these values.
MODEL_ROOT = Path("/kaggle/input/datasets/artgor/cube555-tpu-artifacts")
CHECKPOINT_PATH = MODEL_ROOT / "q555_f1_bell2k.pt"  # Transformer
RESMLP_CHECKPOINT_PATH = MODEL_ROOT / "q555_2k_BEST.pt"  # ResMLP

PUZZLE_ID_START = 1020  # inclusive
PUZZLE_ID_END = 1020  # inclusive
BEAM_WIDTH = 2**25  # any requested width; pipeline profile is selected automatically
MAX_DEPTH = 140
TRANSFORMER_WEIGHT = 0.8  # 0 = ResMLP, 1 = Transformer

REFLECT_MODE = "off"  # off | after_original | only
REFLECT_SOURCE_CSV = None  # path to solutions CSV; required for "only"
SOLUTION_MODE = "collect"  # first | collect
COLLECT_UNTIL_DEPTH = MAX_DEPTH  # used by "collect"; capped to history budget
MAX_COLLECTED_SOLUTIONS = 100_000

KAGGLE_OWNER = "trydotatwo"
KAGGLE_SLUG = "cube555-native-2xt4-blend"
''')
    cell(f'''# Runtime settings: normally leave unchanged.
SOLVER_COMMIT = {commit!r}
SMOKE_TEST = {smoke!r}
COMPETITION_ROOT = Path("/kaggle/input/competitions/cayley-py-555-cube")
LAYOUT_PATH = MODEL_ROOT / "piece_layout_555.json"
TOUCH_BFS_RADIUS = 4
B_MICRO = 8192
MODEL_MICRO = 512
PUBLISH_RESULTS = True
KAGGLE_VERSION = 9
PUZZLE_IDS = list(range(PUZZLE_ID_START, PUZZLE_ID_END + 1))
if not PUZZLE_IDS:
    raise ValueError("PUZZLE_ID_END must be >= PUZZLE_ID_START")
for path in (CHECKPOINT_PATH, RESMLP_CHECKPOINT_PATH, LAYOUT_PATH,
             MODEL_ROOT / "puzzle_info.json", COMPETITION_ROOT / "test.csv",
             COMPETITION_ROOT / "sample_submission.csv"):
    if not path.is_file():
        raise FileNotFoundError(f"Missing input: {{path}}. Attach the dataset or edit its path.")
''')
    cell('''import json, subprocess, sys, time
import torch
names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
assert len(names) == 2 and all("T4" in name for name in names), names
print("GPUs:", names, "torch:", torch.__version__, flush=True)
repo = Path("/tmp/cube555_native_repo")
if not repo.exists():
    subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout",
        "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git", str(repo)], check=True)
subprocess.run(["git", "fetch", "--depth", "1", "origin", SOLVER_COMMIT], cwd=repo, check=True)
subprocess.run(["git", "checkout", "--detach", SOLVER_COMMIT], cwd=repo, check=True)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
assert actual == SOLVER_COMMIT, (actual, SOLVER_COMMIT)
sys.path.insert(0, str(repo))
assets = MODEL_ROOT
competition = COMPETITION_ROOT
output = Path("/kaggle/working") / ("cube555_" + time.strftime("%Y%m%d_%H%M%S"))
output.mkdir()
(output / "provenance.json").write_text(json.dumps(dict(solver_commit=actual,
    torch=torch.__version__, gpus=names, smoke=SMOKE_TEST), indent=2))
''')
    cell('''from tools.cube555.smoke import make_fixture, validate_cuda
if SMOKE_TEST:
    validate_cuda(assets, output / "cuda_parity.json")
    competition = Path("/tmp/cube555_smoke_fixture")
    make_fixture(assets, competition)
    pids, beam, depth, radius = [0, 1, 2, 3], 4096, 6, 0
else:
    pids, beam, depth, radius = PUZZLE_IDS, BEAM_WIDTH, MAX_DEPTH, TOUCH_BFS_RADIUS
command = [sys.executable, "-u", "-m", "tools.cube555.run",
    "--assets", str(assets), "--competition", str(competition),
    "--checkpoint", str(CHECKPOINT_PATH), "--mlp-checkpoint", str(RESMLP_CHECKPOINT_PATH),
    "--layout", str(LAYOUT_PATH),
    "--output", str(output / "run"), "--pids", *map(str, pids),
    "--beam", str(beam), "--depth", str(depth), "--touch-radius", str(radius),
    "--transformer-weight", str(TRANSFORMER_WEIGHT)]
command += ["--b-micro", str(B_MICRO), "--model-micro", str(MODEL_MICRO)]
if not SMOKE_TEST:
    command += ["--reflect-mode", REFLECT_MODE, "--solution-mode", SOLUTION_MODE,
        "--collect-until-depth", str(COLLECT_UNTIL_DEPTH),
        "--max-collected-solutions", str(MAX_COLLECTED_SOLUTIONS)]
    if REFLECT_SOURCE_CSV is not None:
        command += ["--reflect-source-csv", str(REFLECT_SOURCE_CSV)]
if PUBLISH_RESULTS and not SMOKE_TEST:
    import hashlib
    notebook_path = Path("/kaggle/working/__notebook__.ipynb")
    if not notebook_path.exists():
        notebook_path = Path("__notebook__.ipynb")
    if not notebook_path.exists():
        raise FileNotFoundError("Cannot publish without the running notebook source")
    source = json.loads(notebook_path.read_text())
    source_cells = [{"cell_type": c["cell_type"], "source": c["source"]} for c in source["cells"]]
    source_hash = hashlib.sha256(json.dumps(source_cells, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    publication = dict(competition="cayley-py-555-cube", kaggle_owner=KAGGLE_OWNER,
        kaggle_slug=KAGGLE_SLUG, kaggle_version=KAGGLE_VERSION,
        kaggle_username=KAGGLE_OWNER, solver_commit=SOLVER_COMMIT,
        kaggle_notebook_sha256=source_hash)
    # Hash identifies actual running cell sources, including the edited configuration.
    (output / "publication.json").write_text(json.dumps(publication, indent=2))
    command += ["--publish", "--publication-json", str(output / "publication.json")]
print("Run:", command, flush=True)
with (output / "launcher.log").open("w") as log:
    process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in process.stdout:
        print(line, end="", flush=True)
        log.write(line)
        log.flush()
    code = process.wait()
assert code == 0, f"native launcher exited {code}; inspect {output}"
''')
    cell('''import pandas as pd
summary = json.loads((output / "run/run_summary.json").read_text())
print(json.dumps(summary, indent=2))
submission = output / "run/submission.csv"
display(pd.read_csv(submission).head(20))
if SMOKE_TEST:
    for pid in range(4):
        solutions = pd.read_csv(output / f"run/puzzle-{pid}/solutions/solutions.csv")
        assert len(solutions) > 0, f"smoke puzzle {pid} not solved"
        assert solutions["valid"].all(), f"smoke puzzle {pid} failed replay"
    print("PASS: four legal Cube555 scrambles solved and replayed by two native ranks")
''')
    cells[0], cells[1] = cells[1], cells[0]
    output.mkdir(parents=True, exist_ok=True)
    name = 'cube555-2xt4-blend.ipynb'
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3')), nbformat=4, nbformat_minor=5)
    (output / name).write_text(json.dumps(notebook, indent=1, ensure_ascii=True) + '\n', encoding='utf-8')
    slug = 'cube555-native-2xt4-blend-smoke' if smoke else 'cube555-native-2xt4-blend'
    metadata = dict(id='trydotatwo/' + slug, title='Cube555 native 2xT4 blend' + (' smoke' if smoke else ''),
        code_file=name, language='python', kernel_type='notebook', is_private=smoke,
        enable_gpu=True, enable_internet=True, dataset_sources=['artgor/cube555-tpu-artifacts'],
        competition_sources=[] if smoke else ['cayley-py-555-cube'], kernel_sources=[], model_sources=[])
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--solver-commit', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    build(args.solver_commit, args.output, args.smoke)
