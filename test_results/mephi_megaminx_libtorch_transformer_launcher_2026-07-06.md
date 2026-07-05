# MEPhI Megaminx LibTorch Transformer Launcher

Date: 2026-07-06

Scope:

- Prepare a MEPhI `basis:kaf12` launcher for trying Vlad Kuznetsov's Megaminx `piece_transformer` model with the explicit C++ LibTorch Stream1 executor.
- Target first large run: 8xA100 40GB, `BEAM_WIDTH=900000000`, puzzle `991` by default, depth `120` by default.
- Preserve the existing MLP/native CUDA path and avoid fallback behavior.

## Files Changed

- `hpc/mephi_8xa100_common.sh`
  - `beam_configure_build` now forwards `CMAKE_PREFIX_PATH` when set.
  - `beam_configure_build` now enables `-DBEAM_ENABLE_LIBTORCH_STREAM1=ON` only when `BEAM_ENABLE_LIBTORCH_STREAM1=ON` is exported.
  - `beam_torchrun_production` and `beam_native_production` now use `BEAM_PRODUCTION_RUNNER_PATH` when set, defaulting to `${BUILD_DIR}/production_runner` otherwise.

- `hpc/start_8xa100_libtorch_megaminx.sh`
  - New SLURM script for `production_runner_libtorch_stream1`.
  - Exports `BEAM_STREAM1_EXECUTOR=libtorch_eager`.
  - Uses `torch.utils.cmake_prefix_path` from the prepared cluster venv for CMake.
  - Adds Torch library directory to `LD_LIBRARY_PATH` for runtime.
  - Uses existing guarded cleanup and 5-second `nvidia-smi` monitor path.

- `hpc/README_8xa100.md`
  - Adds exact cluster commands and weight placement expectations.

## Default Large-Run Config

```text
PUZZLE_ID=991
DEPTH_LIMIT=120
BEAM_WIDTH=900000000
SHARD_COUNT=32
SHARD_CAPACITY_SCALE_PPM=1000000
STREAM4_BATCH_CANDIDATES=262144
STREAM4_TRIGGER_CANDIDATES=1048576
BEAM_B_MICRO=8192
BEAM_STREAM1_CONCURRENCY=8
BEAM_STREAM3_RING_SLOTS=8
BEAM_FINAL_MATERIALIZE_CHUNK_CANDIDATES=98304
BEAM_FINAL_MATERIALIZE_EXCHANGE_SCALE_PPM=2000000
```

For 8 ranks and 32 shards, the 900M aligned local shard size is about 3.52M candidates. With `B_MICRO=8192`, `STREAM3_RING_SLOTS=8`, and 24 Megaminx moves, Stream3 batch is 1,572,864 candidates, so the manual guard has shard-capacity headroom before the job starts.

## Verification

Local checks:

```text
python tests\test_stream1_transformer_backends.py
5 tests passed

python tests\test_stream1_transformer_parity.py
7 tests passed

git diff --check
pass
```

Shell syntax:

```text
docker run --rm -v ${PWD}:/work -w /work cmz-native-dev:2026-05-26 bash -lc "bash -n hpc/mephi_8xa100_common.sh hpc/start_8xa100_libtorch_megaminx.sh"
pass
```

The first direct `bash -n` attempt through Windows `bash.exe` failed because that binary routes to a blocked WSL instance (`E_ACCESSDENIED`), not because of shell syntax.

## Runtime Notes

Actual cluster submission has not been performed in this Codex turn because no app terminal session is attached to the thread. MEPhI rules require compute to run through `sbatch` on `basis:kaf12`, and the README now contains the exact commands to copy/update scripts and submit.

The launcher requires either an already-exported `BEAM_WEIGHT_DIR` containing `manifest.json`, or Vlad's full Kaggle model dump with `.pth`, metadata/generators, and `pilgrim/model.py` available under `MODEL_DIR` so `tools/export_stream1.py` can export fp16 weights before the run.