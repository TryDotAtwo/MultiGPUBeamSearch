import json
from pathlib import Path


SOURCE = Path("kaggle_t4_torchrun/t4-torchrun-p0-depth10.ipynb")
OUT_DIR = Path("kaggle_t4_mlp_sweep")
OUT_NOTEBOOK = OUT_DIR / "t4-mlp-output1-output24-sweep.ipynb"

OUTPUT1_PATH = (
    "/kaggle/input/models/arabidopsisthalian/"
    "megaminx2048-512-8-e4000/pytorch/default/1/"
    "weights_megaminx2048_512_8_e4000.pth"
)

CONFIG = f'''STREAM1_MODEL_TESTS = [
    {{"name": "repo_output24", "weights": ""}},
    {{"name": "kaggle_output1", "weights": "{OUTPUT1_PATH}"}},
]

BEAM_WIDTH = 2**22
START_PUZZLE_ID = 0
PUZZLE_COUNT = 1
DEPTH_LIMIT = 7
RUN_TIMEOUT_SEC = 0

GITHUB_REPO_URL = "https://github.com/TryDotAtwo/MultiGPUBeamSearch.git"
GITHUB_BRANCH = "main"
CUDA_ARCHITECTURES = "75"

TORCHRUN_NNODES = 1
TORCHRUN_NPROC_PER_NODE = 2
TORCHRUN_NODE_RANK = 0
TORCHRUN_RDZV_BACKEND = "c10d"
TORCHRUN_RDZV_ENDPOINT = "127.0.0.1:29500"
TORCHRUN_RDZV_ID = "beam_t4_mlp_sweep"

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
HISTORY_MODE = "ram"
HISTORY_SLOT_COUNT = 2
HISTORY_WORKERS = 1
HISTORY_RAM_BYTES = 28 * 1024**3
HISTORY_DISK_BYTES = 54 * 1024**3
HISTORY_DISK_PATH = "/tmp/beam_history_arena"
SOLVED_NEIGHBORHOOD_RADIUS = 4
SOLVED_NEIGHBORHOOD_MAX_ENTRIES = 0
STREAM2_SUFFIX_RADIUS = 0
STREAM2_SUFFIX_BACKEND = "composed_permutations"
STREAM2_SUFFIX_MAX_COUNT = 0
STOP_ON_FAILURE = False
LIVE_LOG_RANKS = [0]
RUN_STREAM_BENCHMARK = False

RUNTIME_CONFIG_MODE = "manual"
SHARD_BUFFER_COUNT = 2
STREAM4_BATCH_ALIGNMENT = 1024
SHARD_CAPACITY_SCALE_PPM = 1050000
GLOBAL_SPILL_CAPACITY = 0
STREAM5_RECV_CAPACITY_SCALE_PPM = 1000000
GPU_HEADROOM_BYTES = 224 * 1024**2
B_MICRO = 2048
STREAM1_CONCURRENCY = 4
STREAM3_RING_SLOTS = 4
SHARD_COUNT = 8
STREAM4_ACTIVE_SORT_SLOTS = 4
STREAM4_BATCH_CANDIDATES = 98304
STREAM4_TRIGGER_CANDIDATES = 98304

SWEEP_CONFIGS = [
    {{
        "name": "out24_b98_t98",
        "MODEL_TEST": "repo_output24",
        "B_MICRO": 2048,
        "STREAM1_CONCURRENCY": 4,
        "STREAM3_RING_SLOTS": 4,
        "SHARD_COUNT": 8,
        "STREAM4_BATCH_CANDIDATES": 98304,
        "STREAM4_TRIGGER_CANDIDATES": 98304,
        "STREAM4_ACTIVE_SORT_SLOTS": 4,
    }},
    {{
        "name": "out24_b131_t131",
        "MODEL_TEST": "repo_output24",
        "B_MICRO": 2048,
        "STREAM1_CONCURRENCY": 4,
        "STREAM3_RING_SLOTS": 4,
        "SHARD_COUNT": 8,
        "STREAM4_BATCH_CANDIDATES": 131072,
        "STREAM4_TRIGGER_CANDIDATES": 131072,
        "STREAM4_ACTIVE_SORT_SLOTS": 4,
    }},
    {{
        "name": "out24_b196_t196",
        "MODEL_TEST": "repo_output24",
        "B_MICRO": 2048,
        "STREAM1_CONCURRENCY": 4,
        "STREAM3_RING_SLOTS": 4,
        "SHARD_COUNT": 8,
        "STREAM4_BATCH_CANDIDATES": 196608,
        "STREAM4_TRIGGER_CANDIDATES": 196608,
        "STREAM4_ACTIVE_SORT_SLOTS": 4,
    }},
    {{
        "name": "out1_rows24576_sh8",
        "MODEL_TEST": "kaggle_output1",
        "B_MICRO": 24576,
        "STREAM1_CONCURRENCY": 2,
        "STREAM3_RING_SLOTS": 2,
        "SHARD_COUNT": 8,
        "STREAM4_BATCH_CANDIDATES": 98304,
        "STREAM4_TRIGGER_CANDIDATES": 98304,
        "STREAM4_ACTIVE_SORT_SLOTS": 4,
    }},
    {{
        "name": "out1_rows49152_sh8",
        "MODEL_TEST": "kaggle_output1",
        "B_MICRO": 49152,
        "STREAM1_CONCURRENCY": 4,
        "STREAM3_RING_SLOTS": 4,
        "SHARD_COUNT": 8,
        "STREAM4_BATCH_CANDIDATES": 98304,
        "STREAM4_TRIGGER_CANDIDATES": 98304,
        "STREAM4_ACTIVE_SORT_SLOTS": 4,
    }},
    {{
        "name": "out1_rows98304_sh8",
        "MODEL_TEST": "kaggle_output1",
        "B_MICRO": 98304,
        "STREAM1_CONCURRENCY": 2,
        "STREAM3_RING_SLOTS": 2,
        "SHARD_COUNT": 8,
        "STREAM4_BATCH_CANDIDATES": 98304,
        "STREAM4_TRIGGER_CANDIDATES": 98304,
        "STREAM4_ACTIVE_SORT_SLOTS": 4,
    }},
]
'''


def main() -> None:
    notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
    notebook["cells"][0]["source"] = (CONFIG + "\nSWEEP_CONFIGS = [cfg for cfg in SWEEP_CONFIGS if cfg['MODEL_TEST'] == 'kaggle_output1']\n").splitlines(keepends=True)
    for cell in notebook["cells"]:
        cell["outputs"] = []
        cell["execution_count"] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NOTEBOOK.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "id": "trydotatwo/cayley-beam-t4-mlp-output-sweep",
        "title": "Cayley Beam T4 MLP Output Sweep",
        "code_file": OUT_NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "machine_shape": "NvidiaTeslaT4",
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [
            "arabidopsisthalian/megaminx2048-512-8-e4000/PyTorch/default/1"
        ],
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()






