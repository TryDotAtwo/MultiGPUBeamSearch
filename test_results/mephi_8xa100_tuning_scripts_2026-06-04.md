# MEPhI 8xA100 tuning scripts

Date: 2026-06-04

Prepared a three-step SLURM tuning workflow for `basis/kaf12`:

1. `hpc/tune_8xa100_stream1.sh`
   - builds `stream_benchmark`;
   - tests the compiled Stream1 sweep for `B_MICRO` and inference concurrency;
   - writes `logs/best_stream1.env`.

2. `hpc/tune_8xa100_pipeline.sh`
   - builds `production_runner`;
   - sources `logs/best_stream1.env` when present;
   - sweeps shard count, Stream3 ring slots, Stream4 batch, and Stream4 trigger;
   - runs every valid config through torchrun on 8 ranks to `DEPTH_LIMIT=12`;
   - writes `logs/tuning_<job_id>/pipeline_sweep_<job_id>.tsv`;
   - writes the fastest passing config to `logs/best_pipeline.env`.

3. `hpc/start_8xa100_best.sh`
   - sources `logs/best_pipeline.env` when present;
   - derives per-rank shard capacity from local beam and shard count;
   - launches the selected production config through torchrun.

Shared logic:

- `hpc/mephi_8xa100_common.sh` owns MEPhI paths, venv/NCCL/CUTLASS preflight,
  CMake/Ninja build flags, Kaggle-style local-beam shard-capacity derivation,
  manual config guards, NCCL id-file setup, and rank-separated torchrun logging.

Safety notes:

- No compute payload runs outside SLURM.
- The scripts do not clone/fetch/reset inside compute jobs.
- Rank 0 writes the main log; other ranks write under `logs/ranks_<job>_<tag>/`.
- Configs are rejected before launch if Stream4 batch/trigger or Stream3 batch
  cannot fit in a physical shard buffer, or if Stream1 concurrency exceeds
  Stream3 ring slots.

Local verification:

- Windows WSL `bash -n` was unavailable in this Codex environment due
  `Bash/Service/CreateInstance/E_ACCESSDENIED`, so verification was rerouted
  through Docker.
- Docker image used: existing local `gpu-dev:latest`; no image build or pull.
- Docker `bash -n` passed for:
  - `hpc/mephi_8xa100_common.sh`
  - `hpc/tune_8xa100_stream1.sh`
  - `hpc/tune_8xa100_pipeline.sh`
  - `hpc/start_8xa100_best.sh`
- Docker dry-run of the 8-rank, `BEAM_WIDTH=260000000`, `SHARD_COUNT=16`,
  `STREAM4_BATCH_CANDIDATES=524288`, `STREAM4_TRIGGER_CANDIDATES=1048576`,
  `SHARD_CAPACITY_SCALE_PPM=1250000`, `B_MICRO=4096`,
  `STREAM3_RING_SLOTS=4` sizing produced:
  - `world=8`
  - `local=32505856`
  - `logical=2031616`
  - `cap=2539520`
  - `stream3_batch=393216`

Cluster follow-up:

- Job `31079` failed because SLURM copied the submitted script to
  `/var/lib/slurm/slurmd/job31079/slurm_script`, making `BASH_SOURCE[0]`
  resolve beside the spool script rather than the submit directory. The tuning
  launchers now use `${SLURM_SUBMIT_DIR:-$(pwd)}` for `SCRIPT_DIR`.
- Job `31080` completed the Stream1 benchmark and exposed a parser bug:
  `stream_benchmark` logs `b_micro=...`, while the first launcher parsed
  `B_MICRO=...`. The parser now reads lowercase `b_micro=`.
- Job `31081` completed the first pipeline config through depth 12, then the
  sweep script failed while summarizing because the average-depth `awk` used
  `printf "%.6f"` without passing `sum / n`. The pipeline summary parser now
  uses `printf "%.6f", sum / n`.
- Added automatic per-config GPU monitoring around torchrun. Each config writes
  `logs/tuning_<job>/nvidia_smi_<run_tag>.log` with a timestamped `nvidia-smi`
  CSV sample every 5 seconds.
