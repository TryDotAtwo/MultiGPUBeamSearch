# Kaggle 2xT4 MLP Beam Autoprofiles

## Goal

Provide one reusable private Kaggle notebook for real 2xT4
`MultiGPUBeamSearch` runs. A user supplies a supported MLP checkpoint, puzzle
inputs, an exact beam width, depth, and puzzle ids. The notebook validates the
inputs, exports the model, selects the nearest measured runtime profile, builds
the solver, runs two ranks, and preserves logs and result artifacts.

The notebook must make `2**25` practical enough to benchmark and use. It must
also provide measured profiles anchored at every power from `2**16` through
`2**25`, separately for one-output and per-move-output MLPs.

## Supported Model Contract

The notebook header must state that arbitrary PyTorch models are not supported.
It accepts only checkpoint layouts already implemented by
`tools/export_stream1_mlp.py`:

1. `PilgrimAttnRes`-style BatchNorm MLP checkpoints exported as
   `batchnorm-folded`.
2. `ResMLPDistance`-style LayerNorm MLP checkpoints exported as
   `resmlp-layernorm`.

The exporter infers `output_dim` from the checkpoint. The solver accepts
`output_dim == 1` or `output_dim == move_count`, which is `24` for Megaminx.
Every other model layout or output dimension fails before build or solver
launch with an actionable message.

## User Interface

The first notebook section contains:

- a prominent `SUPPORTED MODELS` block with the exact contract above;
- one config cell containing checkpoint path, exporter format, export dtype,
  puzzle-info path, test CSV path, exact beam width, maximum depth, puzzle ids,
  and benchmark/solve mode;
- optional advanced overrides in a separate section.

The ordinary path requires no editing below the first config cell.

## Profile Selection

Profiles are keyed by Kaggle `2xT4`, model class (`output1` or
`output_move_count`), and every anchor beam power from `16` through `25`.

For a positive user beam width `B`, the notebook selects:

```python
profile_power = clamp(round_half_up(log2(B)), 16, 25)
```

Half-up rounding makes midpoint behavior deterministic. The selected profile
supplies runtime parameters only and never replaces the requested beam.

The runner receives the exact user beam width. The existing solver alignment
rule may increase the effective beam slightly to satisfy
`world_size * shard_count * alignment`. The notebook prints requested beam,
profile anchor, effective beam, local beam, and alignment delta before launch.

## Profile Contents

Each profile records Stream1 row budget and concurrency, Stream3 ring slots,
shard count and capacity scale, Stream4 batch/trigger/sort slots, GPU headroom,
history settings, steady-state depth time, candidate throughput, static or peak
VRAM, status, and evidence kernel version.

Profiles for `output_dim=1` and `output_dim=move_count` are independent. For
`output_dim=1`, row budgeting is converted using the generator move count; no
Megaminx-specific constant is hidden in the selector.

## Tuning Method

The sweep runs on real Kaggle 2xT4 hardware with fixed code revision, build
flags, model checkpoints, and puzzle workload.

1. Establish a correctness and two-rank baseline.
2. Measure anchors `2**16..2**25` for both output classes.
3. Tune one family at a time: row budget/concurrency, shards/capacity, Stream3
   ring, then Stream4 batch/trigger/slots.
4. Warm up setup-heavy paths before steady-state measurement.
5. Record end-to-end and per-phase/depth time.
6. Preserve OOM, overflow, timeout, and invalid-config rows.
7. Select the fastest stable point with memory and capacity margin.

The primary metric is steady-state time at a full frontier. Puzzle solve time
is reported but is not the selection metric because it depends on model quality
and puzzle difficulty. Adjacent small-beam anchors may share a configuration,
but each anchor still receives an explicit measured evidence row.

## Notebook Execution Flow

1. Validate exactly two T4 GPUs and explicit torchrun topology.
2. Identify and export the checkpoint.
3. Validate manifest state dimensions, class count, move count, and output mode.
4. Select and display the nearest profile.
5. Compute aligned/local beam, capacities, history budget, disk need, and
   expected GPU allocation.
6. Fail closed on invalid or unsafe budgets.
7. Clone and build transient dependencies under `/tmp`.
8. Launch the existing C++ runner through explicit two-rank torchrun.
9. Stream concise rank-zero progress while preserving full rank logs.
10. Write machine-readable results under `/kaggle/working`.

Notebook tuning does not change CUDA/C++ search algorithms.

## Errors and Diagnostics

The notebook distinguishes unsupported checkpoints, unsupported output
dimensions, model/generator mismatches, wrong hardware, failed preflight,
invalid runtime configuration, stream overflow, CUDA graph allocation failure,
rank failure, and unsolved-within-depth.

Failures preserve logs, selected profile, derived configuration, and a summary
JSON. They do not silently clamp capacities, change the user's beam, or fall
back to another model mode.

## Artifacts

The tuning kernel produces raw logs, a CSV of every attempted configuration, a
JSON file with selected profiles, and a Markdown verification report. The user
notebook produces the exported manifest, requested/effective beam metadata,
selected profile snapshot, per-puzzle results, rank logs, final summary JSON,
and a submission CSV in solve mode.

## Verification Gates

- Notebook JSON parses and ordinary Python cells AST-parse.
- Profile selection tests cover boundaries, midpoints, clamps, and arbitrary
  non-power-of-two beams.
- The requested beam is preserved and only altered by documented alignment.
- Both exporter formats are locally exercised.
- Both output modes complete a real two-rank Kaggle smoke.
- Representative small, middle, and `2**25` anchors complete on 2xT4.
- Every profile has evidence or is explicitly marked unvalidated.
- Downloaded logs, results, paths, and submission are inspected.
- The notebook remains private until publication is separately approved.
