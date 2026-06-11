# Portable 8xA100 80GB 1.4B Beam Launch

This folder is for running the current production beam search on one node with
8 NVIDIA A100 80GB GPUs. The launcher runs the same two-pass workflow used on
MEPhI:

1. solve the original puzzle;
2. build the reflected synthetic puzzle from the original solution;
3. solve the reflected puzzle;
4. print and verify `candidate_solution_for_original`.

## Requirements

- SLURM cluster with one 8-GPU node.
- CUDA compiler and driver visible on the compute node.
- CUTLASS checkout with `include/` available.
- Python venv with `python`, `torch`, `ninja`, and NVIDIA NCCL package files.
- Repository cloned with the exported Stream1 weights present:
  `stream1_weights_artgor_bf16/`.

The script does not install dependencies on the cluster.

## Quick Start

From the cloned repository root:

```bash
chmod +x hpc/portable_8xa100_80gb/run_1p4b_8xa100_80gb.sh

CUTLASS_DIR=/path/to/cutlass \
NINJA_VENV_DIR=/path/to/ninja-venv \
PUZZLE_ID=992 \
sbatch -p YOUR_PARTITION hpc/portable_8xa100_80gb/run_1p4b_8xa100_80gb.sh
```

If the venv is already active, `NINJA_VENV_DIR` can be omitted:

```bash
source /path/to/ninja-venv/bin/activate
CUTLASS_DIR=/path/to/cutlass PUZZLE_ID=992 \
sbatch -p YOUR_PARTITION hpc/portable_8xa100_80gb/run_1p4b_8xa100_80gb.sh
```

## Default Parameters

The launcher defaults are tuned for 8xA100 80GB:

```bash
BEAM_WIDTH=1400000000
DEPTH_LIMIT=80
SHARD_COUNT=64
SHARD_CAPACITY_SCALE_PPM=1150000
BEAM_B_MICRO=8192
BEAM_STREAM1_CONCURRENCY=8
BEAM_STREAM3_RING_SLOTS=8
STREAM4_BATCH_CANDIDATES=262144
STREAM4_TRIGGER_CANDIDATES=524288
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=88064
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=8000000
BEAM_HISTORY_DISK_BYTES=2199023255552
```

`SHARD_COUNT=64` keeps the per-shard size close to the tested 700M/32-shard
configuration while doubling global beam width for 80GB GPUs.

## Logs

SLURM writes the main output to the normal `slurm-<jobid>.out` file. The runner
also writes:

- `logs/production_runner_*_original.log`
- `logs/production_runner_*_reflected.log`
- `logs/ranks_<jobid>_original/rank*.log`
- `logs/ranks_<jobid>_reflected/rank*.log`
- `logs/tuning_<jobid>/nvidia_smi_original.log`
- `logs/tuning_<jobid>/nvidia_smi_reflected.log`

To summarize a finished run:

```bash
grep -hRE "solution_line_original=|solution_line_reflected=|original_solution_length=|reflected_solution_length=|candidate_solution_length=|candidate_solution_solves_original=|candidate_solution_for_original=" \
  slurm-<jobid>.out logs/ranks_<jobid>_* 2>/dev/null
```

## Cleanup

The job uses per-job directories:

- `build-a100-<jobid>`
- `history-<jobid>`

On success, the script removes those heavy directories. On failure, history is
kept for debugging. Core dumps are disabled with `ulimit -c 0`.
