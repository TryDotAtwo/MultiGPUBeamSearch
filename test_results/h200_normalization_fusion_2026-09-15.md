# H200 normalization fusion experiments

Worktree: `codex/hopper-stream1-fusion`, parent `b7df353f`. Physical target:
one H200 SXM, CUDA 12.8.93, SM90a; Cube4 FP16 D256, compact57,
Hopper QKV/FF1, FF2 m128n128, final CLS reduction, micro384/concurrency8.
Units are **parents/s**, not 24 candidate scores per parent.

## LN -> QKV experiment: rejected

`cuda/stream1_transformer_ln_qkv_fusion.cuh` is an explicitly experimental,
standalone test-only implementation; production inference does not call it.
It normalizes input in registers/shared memory and consumes it with SM90 WGMMA,
removing the normalized global intermediate. Packed weights retain one layout.

At 21,888 token rows, three unprofiled graph measurements (200 replays):

| Path | Microseconds |
|---|---|
| Existing LN + Hopper QKV | 44.5355, 44.4482, 44.4149 |
| Experimental fused LN-QKV | 631.862, 632.468, 631.852 |

Approximately 14.2x slower. Do not promote. Small shapes and tails 1/63/64/65/131
were checked; max absolute half-output difference was 7.62939e-6, not bit exact.
At 21,888 rows, 588/16,809,984 half outputs differed. Memcheck: zero errors.
Resource metadata: 90 registers/thread, 1,024 static shared bytes, 98,304 dynamic
shared bytes. Hardware-counter collection was denied (`ERR_NVGPUCTRPERM`), so
there is no measured occupancy or DRAM-traffic attribution. The simple schedule
is not evidence that all possible LN-GEMM fusion designs are unprofitable.

## Input LayerNorm -> block-0 LayerNorm

Opt-in `BEAM_STREAM1_TRANSFORMER_DUAL_INPUT_LN=1`, default off. Keeps the
persistent FP16 input-LN output and feeds that same rounded value to block-0 LN
from registers. Removes one kernel and one token-buffer read, not every
intermediate write. Other transformer blocks and GEMM implementations unchanged.
Requires compact FP16 D256 with at least one block; freeze before graph capture.

Initial synthetic tests: both output buffers bit exact at batches 2/5/17,
including graph-job selection and inactive parents; memcheck and synccheck clean.
Initial 5-pair isolated sweep showed approximately 1% gain, but p3 real-input
score parity failed. A varied 97-state real-weight test localized this to 148
normalized half outputs, with persistent tokens exact. This initial version is
not eligible for promotion. Final follow-up results are recorded below.

### Corrected scalar path and final physical-H200 gate

Using the incumbent dtype-aware scalar load/store helpers for the second LN
removed the mismatch; direct FP16 helper substitution had changed generated
arithmetic. Exact compiler-level cause was not isolated. Do not simplify this
conversion path without rerunning the real-weight test.

Five process pairs, 300 graph replays each:

| Path | Median parents/s | Individual rates |
|---|---:|---|
| Existing input LN + block-0 LN | 593973.9 | 594377.8, 591767.3, 594771.2, 593973.9, 593000.9 |
| Dual LN | 598913.8 | 598625.4, 600343.8, 599373.7, 598913.8, 597236.9 |

Median throughput +0.8317%; all five pairs positive (0.7143–1.4493%).
The process order was control then candidate in each pair, not counterbalanced.
Both intermediate buffers match on the varied 97-state real-weight fixture;
full score dumps match for CSV p1–10 and p1000. Final real-weight memcheck and
synccheck both report zero errors. These are inference gates, not solved-puzzle
or whole-pipeline performance claims. No complete project regression was run.

User requested a deeper RTX6000 campaign review before further experiments.
Therefore the 100M full-pipeline pair and promotion are **deferred**, not passed.
Default and `h200_single_optimized.sh` remain unchanged. Final source formatting
and CTest registration were performed locally after the measured build;
the arithmetic implementation above was tested on H200.

Raw logs, dumps, rejected-v1 evidence, compiler output and sanitizer records:
`h200_normalization_fusion_evidence.tar.gz`. GPU instance 51111383 was stopped
and the Vast UI confirmed Inactive, $0.112/hr retained disk. Credit changed
$836.52 -> $835.18 during this restart window (~$1.34; not an itemized invoice).
Archive SHA256: `210518245b48ee456e8d16fd1c39d624d16f96ffe8f36616eb546e9e68261ac7`.
