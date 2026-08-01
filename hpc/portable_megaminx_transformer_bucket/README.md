# Portable Megaminx Transformer Bucket Runner

This folder is a generic SLURM entrypoint for Megaminx `piece_transformer`
solve-bucket runs. It is intended for a person with any one-node multi-GPU
server/cluster to clone the repository, point it at their CUDA/CUTLASS/Python
environment, and launch one prebuilt binary plus many one-puzzle array jobs.

The run mode:

1. reads a CSV with `puzzle_id,path`;
2. selects rows by `KNOWN_LENGTHS`;
3. solves the original puzzle in bucket mode;
4. if original is not shorter than the known path, solves reflected states
   generated from found original solutions;
5. keeps all found solutions in TSV logs.

## Requirements

- SLURM with a GPU partition.
- One node with NVIDIA GPUs.
- CUDA compiler and runtime visible on the compute node.
- CUTLASS checkout available through `CUTLASS_DIR`.
- Python venv with `python`, `torch`, `ninja`, and NCCL package files.
- Repository clone with `weights/megaminx_vlad_transformer_fp16/manifest.json`.
- Input CSV with header:

```csv
puzzle_id,path
108,move.move.-move
```

## Quick Start

From repository root:

```bash
chmod +x hpc/portable_megaminx_transformer_bucket/run_len_bucket.sh

PARTITION=YOUR_GPU_PARTITION \
RUN_ROOT=/scratch/$USER/megaminx_bucket \
CUTLASS_DIR=/path/to/cutlass \
NINJA_VENV_DIR=/path/to/ninja-venv \
SOLUTIONS_CSV=/path/to/target_solutions.csv \
KNOWN_LENGTHS=75 \
ARRAY_SPEC=0-99%1 \
hpc/portable_megaminx_transformer_bucket/run_len_bucket.sh
```

The script submits:

- one prebuild job;
- one dependent bucket array job.

It prints both job ids and the exact log paths.

## Common Knobs

Defaults target 8xA100 40GB with a small 30M bucket beam:

```bash
BEAM_WIDTH=30000000
DEPTH_LIMIT=auto
SHARD_COUNT=4
BEAM_B_MICRO=512
BEAM_STREAM1_CONCURRENCY=2
BEAM_STREAM3_RING_SLOTS=8
STREAM4_BATCH_CANDIDATES=65536
STREAM4_TRIGGER_CANDIDATES=262144
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=65536
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=WORLD_SIZE_EFFECTIVE*1000000
BEAM_SOLVED_NEIGHBORHOOD_RADIUS=5
BEAM_SOLVE_BUCKET_EXTRA_DEPTHS=2
BEAM_SOLVED_RESULT_CAPACITY=1048576
BEAM_SOLVE_BUCKET_GATHER_SCRATCH_BYTES=2147483648
```

Override these environment variables before calling the script when using a
different GPU count or memory size.

For one puzzle only:

```bash
ARRAY_SPEC=0-0%1 PUZZLE_LIMIT=1 hpc/portable_megaminx_transformer_bucket/run_len_bucket.sh
```

For many puzzles, keep `PUZZLE_LIMIT=1` and use an array:

```bash
ARRAY_SPEC=0-141%1 PUZZLE_LIMIT=1 hpc/portable_megaminx_transformer_bucket/run_len_bucket.sh
```

Increase `%1` only if the cluster has enough free 8-GPU nodes.

## Logs

After submission:

```bash
squeue -u "$USER" -o "%.18i %.9P %.40j %.8T %.10M %.10L %.20R"
tail -f "$RUN_ROOT/slurm-<array_job>_0.out"
```

Per-run logs are under:

```text
$RUN_ROOT/logs/
$RUN_ROOT/logs/tuning_<jobid>/
$RUN_ROOT/logs/ranks_<jobid>_*/
```

Main result TSV:

```text
$RUN_ROOT/logs/solve_bucket_fresh_<FRESH_RUN_TAG>.tsv
```

## Notes

- The script does not install dependencies.
- The script does not download from Kaggle.
- The script reuses one prebuilt `production_runner` for the whole array.
- Heavy `build-*` and `history-*` directories are job-local; successful jobs
  clean them automatically.
