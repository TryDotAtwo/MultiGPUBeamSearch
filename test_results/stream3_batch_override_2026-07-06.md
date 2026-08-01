# Stream3 Batch Override for Transformer Full Runs

Date: 2026-07-06

Context:

- Full 700M Megaminx transformer run with `B_MICRO=512` used `STREAM3_BATCH_CANDIDATES=98304`.
- At depth 7 this produced `stream3_jobs=10029` and `depth_sec=285.868`.
- Counting effective work by `final_request_count`, throughput was `874544039 / 285.868 = 3059276` requests/s.
- The Stream1+2+3 A100 pipeline smoke stayed healthy at about `4.08M` candidates/s, so full-run loss likely came from too many downstream Stream3/Stream4 scheduling units rather than Stream1 graph replay.

Change:

- Added opt-in `BEAM_STREAM3_BATCH_CANDIDATES` for manual runtime configs.
- Without the env var, behavior is unchanged: `STREAM3_BATCH_CANDIDATES = BEAM_STREAM3_RING_SLOTS * effective_parent_batch * MOVE_COUNT`.
- With the env var, shell preflight and C++ runtime require divisibility by `effective_parent_batch * MOVE_COUNT`, then derive the effective Stream3 ring-slot count from the batch.
- `stream_pipeline_benchmark` accepts the same env var so Stream1+2+3 smoke can test larger Stream3 batches before full solve.
- `bench_8xa100_megaminx_transformer.sh` now writes `BEAM_STREAM3_BATCH_CANDIDATES` into the generated best-env file when a target sweep uses the override.

Expected test point:

```text
BEAM_B_MICRO=512
BEAM_STREAM1_CONCURRENCY=2
BEAM_STREAM3_BATCH_CANDIDATES=1572864
```

For a 24-output Megaminx transformer, this is `512 * 24 * 128`, so Stream3 batches the same candidate volume as the old `8192 * 24 * 8` MLP-style setting while Stream1 still runs small transformer microbatches.

Local verification:

```text
bash -n hpc/mephi_8xa100_common.sh hpc/bench_8xa100_megaminx_transformer.sh
cmake -S . -B build-stream3-override-check -DCUTLASS_DIR=/opt/cutlass -DBEAM_CUDA_ARCHITECTURES=75
cmake --build build-stream3-override-check --target stream_pipeline_benchmark production_runner -j2
cmake --build build-stream3-override-check --target contract_tests -j2
./build-stream3-override-check/contract_tests
```

Result:

```text
[100%] Built target stream_pipeline_benchmark
[100%] Built target production_runner
contract_tests=pass
```

Formula check:

```text
stream3_batch=1572864
stream3_effective_ring_slots=128
shard_capacity=2735104
logical_shard=2735104
```

Invalid guard check:

```text
BEAM_STREAM3_BATCH_CANDIDATES=1572865
invalid_stream3_batch=1572865 slot_candidates=12288
```
Follow-up ring-count fix:

The first override preserved `STREAM3_BATCH_CANDIDATES <= shard_capacity`, but left manual `ring_count` autodetection based on one small `B_MICRO * MOVE_COUNT` slot. For `B_MICRO=512`, `BEAM_STREAM3_BATCH_CANDIDATES=1572864`, and requested staging slots `8`, that would keep `ring_count=223` and inflate physical ring staging from about `21.9M` candidates to about `350.7M` candidates.

Manual runtime autodetection now preserves the original staging-window target:

```text
ring_count = ceil(logical_shard_size * requested_BEAM_STREAM3_RING_SLOTS / STREAM3_BATCH_CANDIDATES)
```

For the 700M A100 transformer run this gives:

```text
logical_shard_size=2735104
requested_BEAM_STREAM3_RING_SLOTS=8
STREAM3_BATCH_CANDIDATES=1572864
ring_count=ceil(2735104 * 8 / 1572864)=14
physical_ring_candidates=14 * 1572864 = 22020096
```

This keeps ring staging near the old `223 * 98304 = 21921792` candidates while reducing Stream3 job count by roughly `16x`.

Follow-up verification:

```text
cmake --build build-stream3-ringcount-check --target production_runner stream_pipeline_benchmark contract_tests -j2
./build-stream3-ringcount-check/contract_tests
contract_tests=pass
```
