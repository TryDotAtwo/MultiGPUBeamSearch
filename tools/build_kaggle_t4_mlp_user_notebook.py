#!/usr/bin/env python3
"""Build the private universal Kaggle 2xT4 MLP beam-search notebook."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

OUT_DIR = Path("kaggle_t4_mlp_universal")
OUT_NOTEBOOK = OUT_DIR / "cayley-beam-2xt4-mlp.ipynb"
REGISTRY_PATH = Path("configs/kaggle_t4_mlp_profiles.json")

HEADER = """# Universal Cayley Beam Search — Kaggle 2×T4

## SUPPORTED MODELS

This notebook does not support an arbitrary PyTorch architecture. It accepts only:

1. PilgrimAttnRes-style BatchNorm MLP via `batchnorm-folded` export.
2. ResMLPDistance-style LayerNorm MLP via `resmlp-layernorm` export.

The exported head must have `output_dim=1` or `output_dim=move_count` (24 for Megaminx).
The user supplies the exact beam. The notebook selects the nearest measured `2**16..2**25`
profile and changes the effective beam only by the solver's distributed alignment.
"""

CONFIG = r'''
from pathlib import Path

# USER SETTINGS — normally this is the only cell to edit.
MODEL_SOURCE_MODE = "repository_exported"  # checkpoint | repository_exported
CHECKPOINT_PATH = Path("/kaggle/input/models/arabidopsisthalian/megaminx2048-512-8-e4000/pytorch/default/1/weights_megaminx2048_512_8_e4000.pth")
CHECKPOINT_FORMAT = "batchnorm-folded"  # batchnorm-folded | resmlp-layernorm
MODEL_DTYPE = "fp16"  # fp16 on T4
PUZZLE_INFO_JSON = Path("/tmp/cayley_beam_repo/data/puzzle_info.json")
TEST_CSV = Path("/tmp/cayley_beam_repo/data/test.csv")
SAMPLE_SUBMISSION_CSV = Path("/tmp/cayley_beam_repo/data/sample_submission.csv")
BEAM_WIDTH = 2**21
MAX_DEPTH = 100
PUZZLE_IDS = [0]
RUN_MODE = "solve"  # solve | depth_only
SOLVED_NEIGHBORHOOD_RADIUS = 0
HISTORY_RAM_BYTES = 28 * 1024**3
HISTORY_DISK_BYTES = 32 * 1024**3
GPU_HEADROOM_BYTES = 224 * 1024**2
'''

SELECTOR = r'''
import json, math
PROFILE_REGISTRY = json.loads(PROFILE_REGISTRY_JSON)

def round_half_up_log2(beam_width):
    if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
        raise ValueError("BEAM_WIDTH must be a positive integer")
    return min(25, max(16, int(math.floor(math.log2(beam_width) + 0.5))))

def align_beam(beam_width, world_size, shard_count, alignment=1024):
    quantum = world_size * shard_count * alignment
    return ((beam_width + quantum - 1) // quantum) * quantum

def select_profile(output_dim, move_count):
    if output_dim == 1:
        model_class = "output1"
    elif output_dim == move_count:
        model_class = "output_move_count"
    else:
        raise ValueError(f"only output_dim=1 or output_dim=move_count is supported; got {output_dim}")
    power = round_half_up_log2(BEAM_WIDTH)
    profile = PROFILE_REGISTRY["profiles"][model_class][str(power)]
    if profile["validation_status"] != "measured":
        raise ValueError(f"profile {model_class}/{power} is not measured")
    runtime = dict(profile["runtime"])
    effective = align_beam(BEAM_WIDTH, 2, runtime["shard_count"])
    return {"requested_beam": BEAM_WIDTH, "effective_beam": effective,
            "alignment_delta": effective - BEAM_WIDTH, "profile_power": power,
            "model_class": model_class, **profile}
'''

SETUP = r'''
import json, os, re, shutil, subprocess, sys, time
from pathlib import Path
import pandas as pd

WORK = Path("/kaggle/working")
REPO = Path("/tmp/cayley_beam_repo")
CUTLASS = Path("/tmp/cutlass")
BUILD = Path("/tmp/cayley_beam_build")
WEIGHTS = Path("/tmp/user_stream1_weights")
LOGS = WORK / "beam_logs"
LOGS.mkdir(parents=True, exist_ok=True)

def checked(cmd, cwd=None, env=None):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(list(map(str, cmd)), cwd=cwd, env=env, check=True)

gpu_rows = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], text=True).strip().splitlines()
if len(gpu_rows) != 2 or any(row.split(",", 1)[0].strip() not in {"Tesla T4", "NVIDIA T4"} for row in gpu_rows):
    raise RuntimeError(f"expected exactly two T4 GPUs, observed={gpu_rows!r}")
print("validated_hardware=", gpu_rows)
for path in (REPO, BUILD, WEIGHTS):
    if path.exists(): shutil.rmtree(path)
checked(["git", "clone", "--depth", "1", "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git", REPO])
if not (CUTLASS / "include").exists():
    if CUTLASS.exists(): shutil.rmtree(CUTLASS)
    checked(["git", "clone", "--depth", "1", "https://github.com/NVIDIA/cutlass.git", CUTLASS])

puzzle_info = json.loads(PUZZLE_INFO_JSON.read_text(encoding="utf-8"))
move_count = len(puzzle_info["generators"])
state_len = len(puzzle_info["central_state"])
num_classes = max(max(v) for v in puzzle_info["generators"].values()) + 1
if MODEL_SOURCE_MODE == "checkpoint":
    if CHECKPOINT_FORMAT not in {"batchnorm-folded", "resmlp-layernorm"}:
        raise ValueError("CHECKPOINT_FORMAT must be batchnorm-folded or resmlp-layernorm")
    if not CHECKPOINT_PATH.is_file(): raise FileNotFoundError(CHECKPOINT_PATH)
    checked(["python3", REPO / "tools/export_stream1_mlp.py", "--weights", CHECKPOINT_PATH,
             "--out", WEIGHTS, "--format", CHECKPOINT_FORMAT, "--dtype", MODEL_DTYPE,
             "--num-classes", str(num_classes)], cwd=REPO)
elif MODEL_SOURCE_MODE == "repository_exported":
    shutil.copytree(REPO / "stream1_weights", WEIGHTS)
else:
    raise ValueError("MODEL_SOURCE_MODE must be checkpoint or repository_exported")
manifest = json.loads((WEIGHTS / "manifest.json").read_text(encoding="utf-8"))
if manifest.get("state_len") != state_len or manifest.get("num_classes") != num_classes:
    raise ValueError(f"manifest puzzle mismatch: {manifest}")
output_dim = int(manifest["output_dim"])
selected = select_profile(output_dim, move_count)
print(json.dumps({k: selected[k] for k in ("requested_beam", "effective_beam", "alignment_delta", "profile_power", "model_class")}, indent=2))
(WORK / "selected_profile.json").write_text(json.dumps(selected, indent=2) + "\n", encoding="utf-8")
checked(["cmake", "-S", REPO, "-B", BUILD, "-GNinja", "-DCMAKE_BUILD_TYPE=Release",
         "-DBEAM_CUDA_ARCHITECTURES=75", f"-DCUTLASS_DIR={CUTLASS}",
         f"-DBEAM_PUZZLE_INFO_JSON={PUZZLE_INFO_JSON}", "-DBEAM_ENABLE_DEBUG=ON", "-DBEAM_ENABLE_DEPTH_LOGS=ON"])
checked(["cmake", "--build", BUILD, "--target", "production_runner", "-j", "2"])
'''

RUN = r'''
runtime = selected["runtime"]
world_size = 2
shards = int(runtime["shard_count"])
local_beam = selected["effective_beam"] // world_size
logical_shard = (local_beam + shards - 1) // shards
capacity = ((logical_shard * int(runtime["shard_capacity_scale_ppm"]) + 999999) // 1000000 + 1023) // 1024 * 1024
if output_dim == 1:
    parent_batch = int(runtime["b_micro"]) // move_count
else:
    parent_batch = int(runtime["b_micro"])
stream3_batch = parent_batch * move_count * int(runtime["stream3_ring_slots"])
capacity = max(capacity, stream3_batch, int(runtime["stream4_batch_candidates"]), int(runtime["stream4_trigger_candidates"]))
capacity = (capacity + 1023) // 1024 * 1024
print("preflight", {"requested_beam": BEAM_WIDTH, "effective_beam": selected["effective_beam"], "local_beam": local_beam,
                    "shard_capacity": capacity, "stream3_batch": stream3_batch, "history_ram_bytes": HISTORY_RAM_BYTES})

def make_env(history_path):
    env = os.environ.copy()
    env.update({
        "BEAM_WEIGHT_DIR": str(WEIGHTS), "BEAM_PUZZLE_INFO_JSON": str(PUZZLE_INFO_JSON), "BEAM_TEST_CSV": str(TEST_CSV),
        "BEAM_RUNTIME_CONFIG_MODE": "manual", "BEAM_B_MICRO": str(runtime["b_micro"]),
        "BEAM_STREAM1_CONCURRENCY": str(runtime["stream1_concurrency"]), "BEAM_STREAM3_RING_SLOTS": str(runtime["stream3_ring_slots"]),
        "BEAM_SHARD_COUNT": str(shards), "BEAM_SHARD_BUFFER_COUNT": "2", "BEAM_SHARD_CAPACITY_CANDIDATES": str(capacity),
        "BEAM_STREAM4_BATCH_CANDIDATES": str(runtime["stream4_batch_candidates"]),
        "BEAM_STREAM4_TRIGGER_CANDIDATES": str(runtime["stream4_trigger_candidates"]),
        "BEAM_STREAM4_ACTIVE_SORT_SLOTS": str(runtime["stream4_active_sort_slots"]), "BEAM_GLOBAL_SPILL_CAPACITY": "0",
        "BEAM_STREAM5_RECV_CAPACITY_SCALE_PPM": "1000000", "BEAM_GPU_HEADROOM_BYTES": str(GPU_HEADROOM_BYTES),
        "BEAM_HISTORY_MODE": "ram", "BEAM_HISTORY_SLOT_COUNT": "2", "BEAM_HISTORY_WORKERS": "1",
        "BEAM_HISTORY_RAM_BYTES": str(HISTORY_RAM_BYTES), "BEAM_HISTORY_DISK_BYTES": str(HISTORY_DISK_BYTES),
        "BEAM_HISTORY_DISK_PATH": str(history_path), "BEAM_SOLVED_NEIGHBORHOOD_RADIUS": str(SOLVED_NEIGHBORHOOD_RADIUS),
        "BEAM_STREAM2_SUFFIX_RADIUS": "0", "BEAM_DEPTH_LOG_EVERY": "1",
    })
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK"): env.pop(key, None)
    return env

solved_re = re.compile(r"puzzle_solved=(\d+) puzzle_id=(\d+) seconds=([0-9.eE+-]+) solution_length=(-?\d+) solution=(.*)$")
test_df = pd.read_csv(TEST_CSV, index_col="initial_state_id")
sample_df = pd.read_csv(SAMPLE_SUBMISSION_CSV, index_col="initial_state_id") if SAMPLE_SUBMISSION_CSV.is_file() else None
gens = {k: list(v) for k, v in puzzle_info["generators"].items()}
central = list(puzzle_info["central_state"])
def validate_solution(puzzle_id, solution):
    state = [int(x) for x in str(test_df.loc[puzzle_id, "initial_state"]).split(",")]
    for move in ([] if not solution else solution.split(".")):
        state = [state[i] for i in gens[move]]
    return state == central

rows = []
for puzzle_id in PUZZLE_IDS:
    history = Path("/tmp/beam_history_universal") / str(puzzle_id)
    if history.exists(): shutil.rmtree(history)
    rank_dir = LOGS / f"puzzle_{puzzle_id}_ranks"
    cmd = [sys.executable, "-m", "torch.distributed.run", "--no-python", "--nnodes=1", "--nproc-per-node=2", "--node-rank=0",
           "--rdzv-backend=c10d", "--rdzv-endpoint=127.0.0.1:29500", f"--rdzv-id=user_{puzzle_id}",
           f"--log-dir={rank_dir}", "--redirects=3", "--tee=0:3", BUILD / "production_runner",
           str(puzzle_id), str(MAX_DEPTH), str(BEAM_WIDTH)]
    started = time.perf_counter()
    proc = subprocess.run(list(map(str, cmd)), cwd=REPO, env=make_env(history), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    elapsed = time.perf_counter() - started
    log_path = LOGS / f"puzzle_{puzzle_id}.log"
    log_path.write_text(proc.stdout, encoding="utf-8")
    matches = [solved_re.search(line) for line in proc.stdout.splitlines()]
    matches = [m for m in matches if m and int(m.group(1))]
    solution = matches[0].group(5) if matches else ""
    valid = bool(solution) and validate_solution(puzzle_id, solution)
    rows.append({"puzzle_id": puzzle_id, "return_code": proc.returncode, "solved": int(bool(matches)),
                 "valid_solution": int(valid), "solution_length": len(solution.split(".")) if solution else None,
                 "solution": solution, "seconds": elapsed, "log_path": str(log_path)})
    if proc.returncode != 0: raise RuntimeError(f"runner failed puzzle={puzzle_id}; see {log_path}")
    if RUN_MODE == "solve" and not valid: raise RuntimeError(f"no valid solution for puzzle={puzzle_id}")
results = pd.DataFrame(rows)
results.to_csv(WORK / "beam_run_results.csv", index=False)
if RUN_MODE == "solve" and sample_df is not None:
    submission = sample_df.copy()
    for row in rows: submission.loc[row["puzzle_id"], "path"] = row["solution"]
    submission.to_csv(WORK / "submission.csv")
summary = {"hardware": gpu_rows, "requested_beam": BEAM_WIDTH, "effective_beam": selected["effective_beam"],
           "profile_power": selected["profile_power"], "model_class": selected["model_class"], "output_dim": output_dim,
           "move_count": move_count, "max_depth": MAX_DEPTH, "run_mode": RUN_MODE, "solved": int(results["valid_solution"].sum()),
           "result_count": len(results), "artifacts": ["selected_profile.json", "beam_run_results.csv", "submission.csv"]}
(WORK / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
'''

def code(source: str, cell_id: str) -> dict[str, Any]:
    return {"cell_type":"code","id":cell_id,"execution_count":None,"metadata":{},"outputs":[],"source":source.strip().splitlines(keepends=True)}
def markdown(source: str, cell_id: str) -> dict[str, Any]:
    return {"cell_type":"markdown","id":cell_id,"metadata":{},"source":source.strip().splitlines(keepends=True)}

def build_notebook(out_dir: Path = OUT_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = REGISTRY_PATH.read_text(encoding="utf-8")
    selector = "PROFILE_REGISTRY_JSON = " + repr(registry) + "\n" + SELECTOR
    nb={"cells":[markdown(HEADER,"header"),code(CONFIG,"config"),code(selector,"selector"),code(SETUP,"setup"),code(RUN,"run")],
        "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3"}},
        "nbformat":4,"nbformat_minor":5}
    notebook_path=out_dir/OUT_NOTEBOOK.name
    notebook_path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n",encoding="utf-8")
    metadata={"id":"trydotatwo/cayley-beam-2xt4-mlp-universal","title":"Cayley Beam 2xT4 MLP Universal","code_file":notebook_path.name,
              "language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"machine_shape":"NvidiaTeslaT4","enable_internet":True,
              "dataset_sources":[],"competition_sources":[],"kernel_sources":[],
              "model_sources":["arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1"]}
    metadata_path=out_dir/"kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8")
    return notebook_path,metadata_path

def main():
    print(*build_notebook())
if __name__=="__main__": main()