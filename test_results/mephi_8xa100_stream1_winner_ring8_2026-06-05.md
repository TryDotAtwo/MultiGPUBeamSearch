# MEPhI 8xA100 Stream1 Winner Ring8 Update 2026-06-05

## Context

The visible MEPhI terminal showed the saved Stream1 microbenchmark result:

```text
stream1_micro b_micro=8192 concurrent=8 candidates_per_sec=87111861.4
```

This was the fastest observed Stream1 point in the cluster log excerpt.

## Changes

- `hpc/tune_8xa100_staged.sh` now defaults to `BEAM_B_MICRO=8192`,
  `BEAM_STREAM1_CONCURRENCY=8`, `BEAM_STREAM3_RING_SLOTS=8`, and
  `STREAM3_RING_SLOTS_SWEEP=8`.
- `hpc/tune_8xa100_pipeline.sh` uses the same Stream1/ring defaults.
- `hpc/start_8xa100_best.sh` and `hpc/start_8xa100_beamsearch.sh` now fall back
  to the same A100 Stream1 winner profile.
- The tuner scripts keep `BEAM_STREAM3_RING_SLOTS >= BEAM_STREAM1_CONCURRENCY`
  after sourcing `best_stream1.env`, preventing the prior
  `invalid_stream1_concurrency=8 stream3_ring_slots=4` failure.
- The standalone `start_8xa100_beamsearch.sh` now checks the same shard-fit
  invariant as the common MEPhI helpers:
  `STREAM3_BATCH_CANDIDATES <= SHARD_CAPACITY_CANDIDATES`.
- Tuner SLURM time limits were set to `02:00:00`.

## Example Shard-Fit Check

With `BEAM_WIDTH=260000000`, `WORLD_SIZE=8`, `B_MICRO=8192`, and
`STREAM3_RING_SLOTS=8`, Stream3 batch is `1572864` candidates.

```text
SHARD_COUNT=4  shard_capacity=10157056  stream3_batch=1572864  ok
SHARD_COUNT=8  shard_capacity=5079040   stream3_batch=1572864  ok
SHARD_COUNT=16 shard_capacity=2539520   stream3_batch=1572864  ok
SHARD_COUNT=24 shard_capacity=1693696   stream3_batch=1572864  ok
```

This is not a hard `SHARD_COUNT=24` limit. The launchers compute
`SHARD_CAPACITY_CANDIDATES` from the current `BEAM_WIDTH`, `WORLD_SIZE`,
`SHARD_COUNT`, and capacity scale, then reject only configs where the computed
`STREAM3_BATCH_CANDIDATES` does not fit.

## Verification

```text
git diff --check
docker run --rm -v ${PWD}:/work -w /work bash:5.2 bash -n \
  hpc/tune_8xa100_staged.sh hpc/tune_8xa100_pipeline.sh \
  hpc/start_8xa100_best.sh hpc/start_8xa100_beamsearch.sh
```

Both checks passed. Docker required access to the local Docker API.
