# Cube4 portable native-FP16 bring-up

This entry point is for a single node with 1..N homogeneous NVIDIA GPUs. It does not imply that every GPU/CUDA/CUTLASS combination compiles or that a profile measured on one card transfers to another. Unsupported combinations fail before the search. No rental or automatic GPU selection is performed.

Required assets, available **before** paid hot-path testing:

- A matching Cube4 `puzzle_info.json` and `test.csv` (the repository's default `data/puzzle_info.json` is a different 120-state puzzle and is rejected).
- The original Cube4 piece-Transformer FP16 export with `manifest.json`; keep all tensor files beside it. The archived export under `test_results/paper_benchmarks/cube4_ours_t4_b4m_depth10/stream1_transformer_weights_fp16` matched the saved H200 57-file SHA-256 list locally, but files are not part of this Git checkout. Transfer from an authorized existing archive, then repeat checksums on the target.
- A CUDA toolkit whose `nvcc --list-gpu-arch` contains the detected compute capability, a CUTLASS checkout, NCCL headers/library, Python 3, CMake and Ninja. CUDA 13+ is needed for SM103/B300; the script asks the installed compiler rather than guessing other version support.

Set `PORTABLE_REPO_DIR`, `PORTABLE_DATA_DIR`, `PORTABLE_WEIGHT_DIR`, `PORTABLE_BUILD_DIR`, `PORTABLE_RUN_ROOT`, `PORTABLE_CUTLASS_DIR`, `PORTABLE_NCCL_INCLUDE_DIR`, and `PORTABLE_NCCL_LIBRARY` to the actual paths. The defaults use `/workspace`. Then:

```bash
bash hpc/cube4_portable.sh probe
bash hpc/cube4_portable.sh build
bash /workspace/MGBFS/hpc/cube4_portable.sh s1
```

`probe` reports the visible GPU architecture, names, VRAM and free disk. `build` validates the model/generator ordering and puzzle row, then builds for the detected SM with Cube4 state length 96 and 24 moves. Run `stream1_transformer_cuda_tests` and exact score-parity checks before accepting a new GPU backend. `s1` runs one isolated Stream1 benchmark on GPU 0; `PORTABLE_GPU_INDEX` selects another GPU. Sweep `PORTABLE_S1_MICRO` (model inference microbatch) and `PORTABLE_S1_LANES` on the fixed build/puzzle, recording the full reports. The launcher's 128/2 defaults are only a conservative first probe, not a speed profile.

The full search requires an explicit measured profile and memory budget:

```bash
BEAM_WIDTH=<requested-global-beam> BEAM_B_MICRO=<outer-candidate-batch> \
BEAM_STREAM1_CONCURRENCY=<lanes> BEAM_STREAM3_RING_SLOTS=<slots-greater-than-lanes> \
BEAM_SHARD_COUNT=<shards-per-rank> BEAM_HISTORY_RAM_BYTES=<global-bytes> \
BEAM_HISTORY_DISK_BYTES=<global-bytes> PORTABLE_WORLD_SIZE=<visible-gpu-count> \
DEPTH_LIMIT=8 bash hpc/cube4_portable.sh run
```

Set `BEAM_STREAM1_TRANSFORMER_MICRO` separately if the model inference microbatch should be smaller than the outer batch. The launcher aligns shard capacity and reports requested/effective beam; it does **not** silently shrink the beam. It rejects absent profile fields, ring slots not exceeding inference lanes, insufficient free host RAM/disk, a mismatched CMake cache, and mixed GPU architectures. The production runner's exact static GPU memory plan remains the final allocation gate. Start with a bounded depth-8 profile; collect Stream1 throughput and steady-state `depth_done=8` before increasing the beam. A successful mock test is not a CUDA, numerical, or throughput result.

For H200/B300-specific optimization, keep `h200_*` and `b300_*` experiments separate. The portable path intentionally disables architecture-specific Hopper/FP8 kernels and uses the native FP16 CUDA-graph baseline until target-hardware validation justifies changing it.
