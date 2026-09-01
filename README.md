# MultiGPUBeamSearch

CUDA/C++ multi-GPU beam search runner for Cayley graph puzzle searches.

The production executable is `production_runner`. It can run on one GPU or on
multiple GPUs through `torchrun`; the same binary is used in both cases.

## Use from CayleyPy

The optional [`cayleypy-native` Python package](integrations/cayleypy_native/README.md)
keeps `graph.beam_search(...)` and CayleyPy result objects while running compatible
searches through the native CUDA/NCCL backend. Install it from this repository:

```bash
python -m pip install "git+https://github.com/TryDotAtwo/MultiGPUBeamSearch.git#subdirectory=integrations/cayleypy_native"
```

See the adapter guide for explicit source setup, single/multi-GPU configuration,
supported model/graph contracts, CPU fallback and reproducible GPU validation.
Importing or installing the package does not compile CUDA or download models.
The project and adapter are provided under the [MIT license](LICENSE); external
dependencies and model weights retain their own licenses.

## What Is Included

- CUDA beam-search runtime with Stream1 neural scoring, Stream2 move/hash work,
  Stream3/4 filtering and dedup, Stream5 exchange, and final materialization.
- Dynamic puzzle sizing from `puzzle_info.json` and `test.csv`.
- Stream1 weight export from PyTorch checkpoints to runtime weights.
- Single-output and multi-output Stream1 models.
- FP16 and BF16 exported weights, depending on target GPU.
- SLURM examples for large multi-GPU runs.
- Kaggle notebooks for smaller validation runs.

## Requirements

- Linux with NVIDIA driver and CUDA toolkit.
- CMake, Ninja, C++ compiler, CUDA compiler.
- Python with PyTorch.
- CUTLASS checkout.
- For multi-GPU: `torchrun` and NCCL-visible GPUs.

Typical Python setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch ninja numpy
```

Get CUTLASS:

```bash
git clone --depth 1 https://github.com/NVIDIA/cutlass.git external/cutlass
```

## Build

From the repository root:

```bash
cmake -S . -B build \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBEAM_CUDA_ARCHITECTURES=80 \
  -DCUTLASS_DIR="$PWD/external/cutlass"

cmake --build build --target production_runner -j
```

Use the CUDA architecture for your GPU:

| GPU | CMake arch |
|---|---:|
| T4 | `75` |
| A100 | `80` |
| H100 | `90` |

## Data And Weights

By default the runner reads:

```text
data/puzzle_info.json
data/test.csv
stream1_weights/
```

Override these paths with environment variables:

```bash
export BEAM_PUZZLE_INFO_JSON=/path/to/puzzle_info.json
export BEAM_TEST_CSV=/path/to/test.csv
export BEAM_WEIGHT_DIR=/path/to/stream1_weights
```

Export a PyTorch checkpoint:

```bash
python tools/export_stream1_mlp.py \
  --input model.pth \
  --output stream1_weights_custom \
  --dtype bf16
```

For older GPUs such as T4, use `--dtype fp16`. For A100/H100, BF16 is usually
the intended path when the model/export supports it.

## Run On 1 GPU

Build `production_runner`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 \
BEAM_WEIGHT_DIR=stream1_weights \
build/production_runner 0 30 30000000 1 0
```

Arguments:

```text
production_runner <puzzle_id> <depth_limit> <beam_width> [world_size] [local_rank]
```

For a tiny smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 build/production_runner 0 2 512 1 0
```

Useful runtime knobs:

```bash
export SHARD_COUNT=16
export SHARD_CAPACITY_SCALE_PPM=1500000
export BEAM_B_MICRO=8192
export BEAM_STREAM1_CONCURRENCY=8
export BEAM_STREAM3_RING_SLOTS=8
export STREAM4_BATCH_CANDIDATES=262144
export STREAM4_TRIGGER_CANDIDATES=524288
export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=65536
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=2000000
export BEAM_SOLVED_NEIGHBORHOOD_RADIUS=4
export BEAM_HISTORY_RAM_BYTES=68719476736
export BEAM_HISTORY_DISK_BYTES=274877906944
export BEAM_HISTORY_DIR="$PWD/history"
```

## Run Multi-GPU With Torchrun

`production_runner` is launched once per rank. `torchrun` provides
`WORLD_SIZE`, `RANK`, and `LOCAL_RANK`; the runner uses those to split the beam
across GPUs.

Example for one 8-GPU node:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BEAM_WEIGHT_DIR="$PWD/stream1_weights"
export BEAM_HISTORY_DIR="$PWD/history"

export SHARD_COUNT=32
export BEAM_B_MICRO=8192
export BEAM_STREAM1_CONCURRENCY=8
export BEAM_STREAM3_RING_SLOTS=8
export STREAM4_BATCH_CANDIDATES=262144
export STREAM4_TRIGGER_CANDIDATES=1048576
export BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=88064
export BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=8000000

torchrun \
  --nnodes=1 \
  --nproc-per-node=8 \
  --standalone \
  build/production_runner 991 80 700000000
```

For multi-node, replace `--standalone` with an explicit rendezvous:

```bash
torchrun \
  --nnodes=2 \
  --nproc-per-node=8 \
  --node-rank="$NODE_RANK" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="$MASTER_ADDR:29500" \
  build/production_runner 991 80 700000000
```

## SLURM Cluster Example

The portable 8xA100 80GB wrapper runs the two-pass workflow:

1. solve original puzzle;
2. create reflected synthetic puzzle;
3. solve reflected puzzle;
4. print the inverted reflected candidate for the original.

```bash
source /path/to/venv/bin/activate

CUTLASS_DIR=/path/to/cutlass \
NINJA_VENV_DIR=/path/to/venv \
PUZZLE_ID=992 \
BEAM_WIDTH=1400000000 \
DEPTH_LIMIT=80 \
sbatch -p YOUR_GPU_PARTITION hpc/portable_8xa100_80gb/run_1p4b_8xa100_80gb.sh
```

The wrapper defaults are tuned for 8xA100 80GB. Override them with environment
variables when using smaller GPUs.

For direct custom SLURM scripts, the core pattern is:

```bash
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

set -euo pipefail
source /path/to/venv/bin/activate
cd /path/to/MultiGPUBeamSearch

export CUTLASS_DIR=/path/to/cutlass
export BEAM_WEIGHT_DIR="$PWD/stream1_weights"
export BEAM_HISTORY_DIR="$PWD/history-${SLURM_JOB_ID}"

cmake -S . -B "build-${SLURM_JOB_ID}" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBEAM_CUDA_ARCHITECTURES=80 \
  -DCUTLASS_DIR="$CUTLASS_DIR"
cmake --build "build-${SLURM_JOB_ID}" --target production_runner -j "${SLURM_CPUS_PER_TASK:-8}"

torchrun --nnodes=1 --nproc-per-node=8 --standalone \
  "build-${SLURM_JOB_ID}/production_runner" 991 80 700000000
```

## IHES Cube

IHES uses its own puzzle data and model. Prepare:

```bash
export BEAM_PUZZLE_INFO_JSON=/path/to/ihes/puzzle_info.json
export BEAM_TEST_CSV=/path/to/ihes/test.csv

python tools/export_stream1_mlp.py \
  --input p888-t000_1778521793_e32692.pth \
  --output stream1_weights_ihes_bf16 \
  --format batchnorm-folded \
  --dtype bf16 \
  --num-classes 72

export BEAM_WEIGHT_DIR="$PWD/stream1_weights_ihes_bf16"
```

Then run the same 1-GPU or `torchrun` commands.

The ready wrapper is:

```bash
PUZZLE_ID=166 DEPTH_LIMIT=30 BEAM_WIDTH=900000000 \
sbatch -p YOUR_GPU_PARTITION hpc/ihes_cube_model/ihes_solve_then_reflect.sh
```

## Kaggle

Notebook entrypoints live in:

```text
kaggle/
kaggle_torchrun/
kaggle_t4_torchrun/
```

Use them when you want a Kaggle-managed GPU run instead of a local build. The
notebooks clone the repository, build the CUDA target, configure the same
environment variables, and launch the runner. The T4 notebooks should use FP16
weights and smaller beam/history budgets than A100/H100 runs.

Typical Kaggle parameters to edit near the top of the notebook:

```python
PUZZLE_ID = 0
DEPTH_LIMIT = 10
BEAM_WIDTH = 2**24
TORCHRUN_NNODES = 1
TORCHRUN_NPROC_PER_NODE = 2
STREAM1_MODEL_WEIGHTS_URL = ""
```

If `STREAM1_MODEL_WEIGHTS_URL` is set to a `.pth` URL/path, the notebook exports
that checkpoint into runtime Stream1 weights before building.

## Logs And Results

The runner prints solved lines like:

```text
puzzle_solved=1 puzzle_id=991 seconds=... solution_length=... solution=...
```

Useful grep:

```bash
grep -hRE "puzzle_solved=1|solution_length=|candidate_solution_for_original=|candidate_solution_solves_original=" logs/*.log slurm-*.out 2>/dev/null
```

For GPU monitoring while a job runs:

```bash
nvidia-smi
nvidia-smi dmon -s pucm
```

## Cleanup

Large runs create heavy build and history directories. Keep logs/results, but
remove per-job build/history after successful runs:

```bash
rm -rf build-<jobid> history-<jobid>
```

Do not delete `data/`, `stream1_weights*/`, or logs unless you have copied the
solutions you need.

## Development Tests

Local build and tests:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Docker GPU tests:

```bash
docker compose build beam-tests
docker compose run --rm beam-tests
```

Test artifacts and verification notes belong under `test_results/`.
