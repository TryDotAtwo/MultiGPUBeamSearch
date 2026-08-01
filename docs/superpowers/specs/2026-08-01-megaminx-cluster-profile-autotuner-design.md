# Adaptive Megaminx Cluster Profile Autotuner Design

**Date:** 2026-08-01  
**Status:** approved design, pending implementation plan

## Goal

Add a standalone cluster-side autotuner to each native Megaminx archive. A user
allocates one homogeneous GPU set and runs one command; the tuner discovers the
maximum stable beam, tunes beam-specific runtime settings down to 30,000,000,
selects the touch-BFS boundary radius, and emits an evidence-backed registry
fragment for that exact hardware tuple.

The first validation target is 8x NVIDIA A100 40 GB, native sm80. The default
wall-clock budget is six hours.

## Public interface

```bash
./autotune.sh \
  --gpus 0,1,2,3,4,5,6,7 \
  --min-beam 30000000 \
  --time-budget 6h
```

Optional `--puzzles 900,901,902` replaces the packaged three-puzzle calibration
set. The tuner submits one SLURM job; that job uses torchrun with one rank per
scheduler-visible GPU. It preserves the same archive, native-SM, payload-hash,
and homogeneous-allocation preflight as normal solving.

The default output directory is a new run-owned path under `autotune-runs/`.
Every session is resumable and refuses to merge observations from a different
GPU name, VRAM class, SM, world size, driver, solver commit, model digest, or
release manifest.

## Search strategy

The tuner uses deterministic adaptive successive halving rather than an
exhaustive Cartesian grid or a probabilistic optimizer.

### Phase 1: maximum stable beam

Starting from 30,000,000, the tuner grows the beam exponentially until it sees
OOM, timeout, capacity failure, or insufficient memory margin. It then performs
a bounded binary refinement. A beam is stable only when:

- all ranks complete;
- NCCL and CUDA report no errors;
- CPU replay and the correctness digest match;
- observed peak VRAM leaves at least 10 percent free on every GPU;
- scratch-disk headroom passes the derived requirement;
- the probe finishes inside its per-candidate time slice.

The highest stable value becomes the upper beam anchor. Failed probes remain in
the evidence log.

### Phase 2: beam anchors

The tuner creates half-up log2 anchors from the discovered maximum down to
30,000,000, plus both exact endpoints. Requested beam is never replaced by the
anchor; the runner receives the requested value and records its actual aligned
effective beam.

### Phase 3: successive halving per anchor

Candidate families cover:

- `b_micro`;
- Stream 1 concurrency;
- Stream 3 ring slots;
- shard count and capacity scale;
- Stream 4 batch, trigger, and active sort slots;
- touch-BFS boundary radius.

Candidates receive one warm-up and one short screening run. Each round retains
the fastest stable fraction while increasing puzzle coverage and measurement
duration. Finalists run three measured repetitions on all three calibration
puzzles. Selection uses robust median end-to-end time, then lower peak VRAM,
then deterministic configuration id as tie breakers.

## Touch-BFS boundary tuning

Touch-BFS radius is a first-class, beam-specific profile field. Radius search is
bounded before execution using the branching upper bound
`move_count ** radius`; the calculation uses checked integers and stops before
overflow or configured RAM/disk limits. This bound is deliberately pessimistic:
the tuner also records the actual deduplicated frontier size at every radius.

For each beam anchor, the tuner evaluates radii from zero upward. A next radius
is admitted only if all of the following hold:

- its `move_count ** radius` upper bound fits the candidate budget;
- actual frontier construction fits host RAM and scratch limits;
- the boundary artifact passes its digest and replay checks;
- its construction time can be amortized within the remaining six-hour budget;
- the preceding radius was stable.

Search stops at the first inadmissible radius; it does not skip over a failed
radius. The winning radius minimizes total end-to-end time including boundary
construction and lookup. Precomputation time is never excluded. Different beam
anchors may select different radii.

## Calibration and correctness

The packaged default calibration set contains one short, one medium, and one
hard Megaminx puzzle. Puzzle order rotates between repetitions. Setup-heavy
paths and CUDA graphs are warmed before measured work.

Every accepted row records workload ids, requested/effective beam, complete
runtime configuration, BFS radius and frontier counts, exactness digest, replay
result, per-rank peak VRAM, host RAM, scratch bytes, setup time, solve time,
wall time, throughput, exit status, driver, CUDA runtime, solver commit, model
digest, and release-manifest digest.

A mismatch, OOM, timeout, crash, or missing metric is a failed row. There is no
fallback to another GPU, world size, model class, beam range, or BFS radius.

## Budgeting and resume

The six-hour controller maintains a hard deadline. It reserves enough time to
run three repetitions of the current finalists and stops launching screening
candidates when that reserve would be violated. A partial session remains
resumable but cannot emit a runnable `measured` profile.

Atomic checkpoints are written after every candidate. Resume validates the
session identity fields before reusing observations. SLURM timeout, process
termination, and individual candidate failure do not erase completed evidence.

## Outputs

- `session.json`: immutable hardware/workload/release identity and budget;
- `autotune_results.jsonl`: append-only row for every success and failure;
- `leaderboard.tsv`: deterministic human-readable ranking;
- `profile_candidate.json`: selected anchors and resource evidence;
- `registry.fragment.json`: exact hardware/world-size registry fragment;
- `resume.json`: atomic controller checkpoint;
- per-candidate stdout, stderr, rank logs, and metric files.

The registry fragment uses `measured` only after every required final repetition
passes. Otherwise it uses `unverified` and remains rejected by the normal
profile selector. Installation is an explicit local command; the tuner does not
modify a signed archive or publish results automatically.

## Components

- `autotune.sh`: thin public wrapper and SLURM submit command.
- `autotune_submit.py`: argument validation, run directory, and allocation.
- `scripts/autotune_job.sh`: compute-node preflight and torchrun controller.
- `autotune/controller.py`: deadline, resume, phases, and successive halving.
- `autotune/search_space.py`: deterministic runtime and BFS candidates.
- `autotune/evidence.py`: identity, JSONL rows, ranking, and registry emission.
- `autotune/probe.py`: isolated solver invocation and metrics parsing.

The existing solver, Stream 1-5 architecture, and production result publisher
remain unchanged.

## Error handling and tests

All invalid inputs fail before `sbatch`. Compute-node checks fail closed on
mixed hardware, wrong SM, wrong GPU count, payload drift, insufficient disk, or
unsupported profiles. Candidate subprocesses have hard timeouts and run-owned
cleanup paths.

Deterministic tests cover argument validation, deadline reservation, beam-bound
search, half-up anchors, successive-halving selection, tie breaking, checked
`move_count ** radius`, BFS stop rules, resume identity, OOM/timeout rows,
exactness rejection, registry schema, and mock-SLURM end-to-end execution. The
8xA100 trial additionally records real nvidia-smi, peak-memory, correctness,
and repeated timing evidence before the profile becomes runnable.
