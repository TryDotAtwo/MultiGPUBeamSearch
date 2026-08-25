# Molab Cube4 ReLU boundary diagnosis — 2026-08-25

## Acceptance contract

- GPU execution: Molab RTX PRO 6000 Blackwell Server Edition, SM120.
- Model: Cube4 piece Transformer, ReLU, `output_dim=24`.
- Acceptance workload: beam `2**25`, exactly `depth_done=8`.
- Accepted control: 80.2952 seconds and 391 Stream 3 jobs.
- Outer microbatch 3,584; inner Transformer microbatch 896; concurrency 4.
- Final materialization chunk 88,064.

## Observations

The initial automatic profile allocated 25 logical rings and 600 physical
jobs and the sandbox disappeared after depth 5. A manual profile with two
physical rings progressed through saturated depth 6, demonstrating that a
small physical pipeline is viable. That run accidentally reused 88,064 as the
Stream 4 batch and trigger. It therefore emitted 1,659 Stream 4 jobs and depth
6 took 107.929 seconds; it is a diagnostic run, not a performance candidate.

A subsequent `2**25` attempt retained two rings and increased the Stream 4
batch, but the remote sandbox disappeared before a terminal status or CUDA
error could be recovered. The evidence does not distinguish a platform
watchdog, host termination, or device-memory failure, so this is not labelled
an OOM.

## Prepared boundary sweep

The next live Molab session will run one anchor at a time and advance only
after a successful `depth_done=8`:

| Beam | Physical rings | Stream 3 slots | Stream 3 batch | Shards | Per-shard capacity |
|---:|---:|---:|---:|---:|---:|
| `2**22` | 2 | 4 | 344,064 | 4 | 1,048,576 |
| `2**24` | 2 | 12 | 1,032,192 | 8 | 2,097,152 |
| `2**25` | 2 | 24 | 2,064,384 | 16 | 2,097,152 |

All anchors use Stream 4 batch 196,608, trigger 393,216, four active sort
slots, final materialization 88,064, disk history, and 8 GiB explicit device
headroom. The launch and poll scripts are parameterized by
`MOLAB_BOUNDARY_BEAM_EXP` and live under the workspace `test_results` folder.

## Current blocker

The replacement Molab session successfully ran the first two anchors with the
saved SM120 runner (SHA-256
`51a790190f66783d809042933f78d0b35ab04003607febbb1f128f40e441c71e`).
The persistent filesystem is `noexec`, so the launcher now copies the runner
to `/tmp` and marks that copy executable before starting it.

| Beam | Depth | Seconds | Stream 3 jobs | Stream 4 jobs | Result |
|---:|---:|---:|---:|---:|---|
| `2**22` | 8 | 14.2836 | 293 | 345 | rc=0 |
| `2**24` | 6 | 56.3178 | 391 | 362 | complete |
| `2**24` | 7 | 56.4880 | 391 | 378 | complete |
| `2**24` | 8 | 56.4112 | 391 | 372 | rc=0 |

The apparent `2**24` failure was a detached/idle-lifetime problem. Re-running
the identical profile as one foreground marimo execution with a heartbeat
completed in 190.86 s wall time and produced the rc=0 depth-8 row above.

The subsequent foreground `2**25` run reached the full 33,554,432 frontier at
depth 5 in 11.5007 s, using 12,819 MiB device memory and 100% reported GPU.
The whole sandbox then disappeared during saturated depth 6 before a terminal
status was written. No CUDA fatal line was observed before endpoint loss. The
next run will move disk history from the persistent/FUSE mount to `/tmp` and
record host available memory and `/tmp` free space in every heartbeat so the
remaining host/storage boundary can be classified.
