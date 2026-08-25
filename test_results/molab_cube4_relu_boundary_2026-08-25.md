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

The supplied Molab endpoint no longer accepts `marimo-pair` connections. No
local GPU run was substituted. A fresh live Molab URL/token is required to
execute the prepared sweep.
