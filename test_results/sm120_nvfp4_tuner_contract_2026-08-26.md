# SM120 NVFP4 autotuner contract preparation (2026-08-26)

## Scope

This checkpoint prepares the quality/autotuning side for the user-approved
fixed Cube4 fused FFN path. It is not a production acceptance result.

## Contract added

- precision identifier: `sm120_nvfp4`;
- value encoding: signed E2M1, maximum magnitude 6;
- scale encoding: unsigned UE4M3;
- activation scale ownership: one scale per row and contiguous K16 vector;
- immutable HxO weight scale ownership: one scale per output column and K16;
- accumulation: FP32;
- output/fallback: FP16;
- quality candidates: all 16 core operators and each single operator;
- truth comparison remains original FP32 versus accepted FP16 versus candidate.

The native fixed-shape policy additionally requires SM120, FP16 source
weights, ReLU, `d_model=256`, `ff_dim=1024`, `output_dim=24`, a padded sequence
length no larger than 64, immutable offline weights, and a next layer. It fixes
two N128 consumer groups and starts the Molab register-pressure sweep at M64.

Positive E4M3FN values share the finite UE4M3 encoding used for scale-factor
QDQ. The logical evaluator clamps the scale to the minimum positive UE4M3
subnormal before conversion. E2M1 QDQ uses the finite hardware value set
`0, 0.5, 1, 1.5, 2, 3, 4, 6` with sign.

## Fail-closed boundary

The native selector/export path still rejects NVFP4. It will be enabled only
after the following evidence exists for the same immutable candidate:

1. native SM120 fused `256 -> 1024 -> 256` kernel with no global hidden tensor;
2. exact residual/bias/LayerNorm semantics and next-layer NVFP4 emission;
3. Molab CUDA correctness against FP16 and FP32 references;
4. output-24 ranking and real reconstructed-frontier gates;
5. native Stream1 benchmark and Cube4 `2**25`, `depth_done=8` below 80.2952 s.

## Verification state

The supplied sandbox `sb-0ca79391930349fd.sb.molab.run` returned HTTP 410.
Accordingly, no CUDA build or test was run locally and no performance or
correctness claim is made for this checkpoint. The new Python contract tests
are prepared for the next foreground Molab session.

## Additional fail-closed gates prepared after the expired session

- Every candidate benchmark must contain at least three depth-8 wall-clock
  samples and use their median.
- Candidate and FP16 control must share one explicit Cube4/output24/depth0..8
  workload fingerprint and beam width.
- The native kernel contract and compiled-kernel SHA-256 are mandatory.
- A candidate is rejected unless it beats both its contemporaneous FP16
  control and the accepted 80.2952 s depth-8 baseline.
- ReLU FFN equalization is serialized as an immutable paired transform on FF1
  and FF2; missing partners or non-identical 1024-channel scales are invalid.
- The fused-FFN API exposes no global hidden pointer and fixes
  `global_hidden_bytes=0`. Its epilogue forms the residual in FP32, stores an
  FP16 copy, computes LayerNorm from the unrounded FP32 sum, and emits packed
  E2M1 with K16 UE4M3 scales.

These are prepared contracts, not executed results. The selector intentionally
continues rejecting NVFP4 until the native CUDA body and matching evidence are
present.

## Molab session `sb-188a...` preflight

The replacement sandbox initially accepted native marimo-pair execution and
displayed the requested toast. Read-only probes established:

- GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0;
- VRAM: 97,887 MiB total, 0 MiB used, 0% utilization;
- no active `nvcc`, CMake, Ninja, production runner, or torchrun process;
- CUDA compiler wheel: `nvidia-cuda-nvcc 13.3.73`;
- compiler path: `/tmp/uv-venv/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc`;
- CUDA Runtime/CCCL/NVVM wheels and CUTLASS DSL 4.4.2 were present;
- persistent storage was empty in this new sandbox.

The sandbox then stopped accepting pair connections before the public Git
clone/configure command began. A second harmless `print('ALIVE')` probe failed
the same way. No detached process, compilation, or GPU allocation was started,
so this is not evidence of a project or CUDA failure.

A 22,645-byte patch bundle was prepared locally to reproduce the exact three
committed benchmark revisions plus current uncommitted tuner contracts. Its
upload was not performed: the execution approval layer requires explicit user
authorization to send that source payload to the private Molab sandbox.

## Primary implementation references

- NVIDIA CUTLASS SM120 example 79a documents dense block-scaled NVFP4 MMA and
  its 2x MXFP8 throughput relationship.
- CUTLASS Blackwell documentation defines NVFP4 as E2M1 values with UE4M3
  scales and dense scale-factor vector size 16.
- CUTLASS `float_subbyte.h` defines the E2M1 finite value set and exponent bias.
- CUTLASS `float8.h` defines UE4M3 range `[0, 448]`, exponent bias 7, and
  round-to-nearest conversion.

## Live Molab verification update

The user authorized the 22,645-byte source bundle and it was uploaded to the
private `sb-188a...` sandbox. The exact WIP was reconstructed over public base
`8159d254`, configured with CUDA 13.3/CUTLASS `7107b055` for `sm_120a`, and
built foreground. The policy contract passes.

Fresh three-run controls:

| rows | two-GEMM transport median | dense-equivalent | N=256 FF2 median | dense-equivalent |
|---:|---:|---:|---:|---:|
| 51,072 | 0.102514 ms | 522.394 TFLOP/s | 0.039516 ms | 677.613 TFLOP/s |
| 204,288 | 0.409694 ms | 522.857 TFLOP/s | 0.154841 ms | 691.716 TFLOP/s |

The N=256 rerun supersedes the older 117.566 TFLOP/s observation: the full-row
FF2 shape is efficient with the current fixed tile/build.

Two failed compile probes narrowed the actual fused topology:

1. A stock block-scaled `N=1024` CTA tile is invalid: the SM120 builder's MMA
   atom remains N128 and TMA layout equivalence fails (the initial auto-stage
   attempt also fell to one stage, below the required two).
2. A `cluster_N=8` producer is invalid through the stock SM120 block-scaled
   builder: it fails closed with `no programmatic multicast on this arch`.

Therefore the next native kernel remains cluster-size one. Each M64 CTA loops
over eight N128 FF1 producer tiles, writes 32 KiB packed E2M1 values plus 4 KiB
K16 UE4M3 scales to CTA-local shared memory, synchronizes, and executes two
N128 FF2 consumer passes. This removes global hidden traffic without relying
on unsupported giant tiles or cluster multicast.

The native C++ policy executable passes in the same Molab build. The first
selected Python regression run exposed two missing imports; after fixing them,
all 34 calibration/evaluation/selection/tuner tests pass on Molab in 1.69 s.

## CTA-local shared-pipeline probe

The fixed producer/consumer ownership model was compiled and executed on the
same Molab SM120 GPU. The probe allocates exactly 36,864 bytes per CTA: 32 KiB
for packed E2M1 FF1 values and 4 KiB for K16 UE4M3 scales. Eight logical FF1
producer tiles populate that storage, a CTA barrier separates the phases, and
two logical FF2 consumer groups read it. No hidden byte is materialized in
global memory.

CUDA reports two active 288-thread CTAs per SM with this allocation. All output
checksums were non-zero and every run returned zero:

| blocks | repetitions | wall time | effective shared traffic |
|---:|---:|---:|---:|
| 1,024 | 100 | 1.864512 ms | 3,771.094 GiB/s |
| 4,096 | 100 | 6.676000 ms | 4,212.852 GiB/s |
| 1,024 | 1,000 | 18.107103 ms | 3,883.145 GiB/s |

This closes the occupancy/capacity risk: the intended 36 KiB local hidden
buffer fits with useful two-CTA occupancy. It does not yet prove the complete
two-GEMM fusion; tensor-core producer/consumer wiring and numerical gates are
still required.

After recovering the marimo kernel through the normal notebook UI, the probe
was also built through its committed CMake target rather than direct `nvcc`.
CMake regenerated the existing SM120 build successfully and the target linked
cleanly. Repeat runs returned zero with 3,768.830 and 4,216.955 GiB/s for the
1,024- and 4,096-block cases respectively. This verifies both the standalone
kernel and its repository build integration on Molab.

An exploratory CuTe DSL import then identified a tooling-version mismatch:
the checked-out CUTLASS `7107b055` SM120 Python example imports
`MXF8F6F4_SUPPORTED_PAIRS`, while Molab's installed `nvidia-cutlass-dsl`
package does not export that symbol. The repository's Python CuTeDSL sources
cannot simply shadow the installed wheel because their matching compiled
`cutlass._mlir` extension is absent. This is a DSL packaging mismatch, not a
kernel feasibility or GPU failure; the supported C++ CUTLASS path remains the
production implementation route.

## Corrected sliced-hidden topology

The first C++ CUTLASS scaffold exposed a hard constraint that changes the
storage topology: the stock SM120 cooperative block-scaled kernel requires an
M tile of at least 128. Retaining the complete 128x1024 hidden tensor would
need 73,728 bytes before mainloop storage and therefore cannot coexist with
the required two TMA stages. The earlier M64/full-hidden 36 KiB plan is
superseded for the tensor-core implementation.

The accepted topology now streams eight K128 FF1 slices. Two ping-pong buffers
hold only the current and next M128xK128 slice: 16,384 packed E2M1 bytes plus
2,048 UE4M3 scale bytes, 18,432 bytes total. Two 256-thread FF2 consumer groups
read each slice and accumulate independent N128 output tiles, producing N256
after all eight slices without a global hidden tensor or FF1 recomputation.

Molab evidence on the RTX PRO 6000 Blackwell Server Edition:

- policy contract: pass;
- 640-thread structural pipeline: two active CTAs per SM;
- 1,024 blocks x 100 iterations: 3.830464 ms, 5,506.840 GiB/s effective traffic;
- 4,096 blocks x 100 iterations: 13.615200 ms, 6,197.118 GiB/s;
- real CUTLASS scaffold: three TMA stages, 86,016 total shared bytes including
  the 18,432-byte reservation;
- full FF1 shape M=51,072, N=1,024, K=256: 0.067971 ms and 393.938 dense-equivalent TFLOP/s.

The exact K128 ReLU/NVFP4-output control used four stages and 86,016 bytes;
three paired samples measured 0.069392/0.069587/0.069699 ms versus reserved
0.067878/0.067933/0.067907 ms. Thus the honest reservation does not regress
the producer and currently improves it by roughly 2.4%. The scaffold still
uses the stock global epilogue; replacing it with the slice-local producer and
adding the two consumer MMA groups remains the next implementation step.

## Concurrent slots and CTA-shape audit

All runs below used the same Molab RTX PRO 6000 Blackwell Server Edition,
CUDA 13.3, CUTLASS `7107b055`, SM120a, exact FF1 shape
M=51,072/N=1,024/K=256, and foreground execution.

| experiment | result |
|---|---:|
| one independent CUDA slot | 385.227 TFLOP/s |
| two independent CUDA slots | 412.121 TFLOP/s |
| three independent CUDA slots | 405.385 TFLOP/s |
| four independent CUDA slots | 397.793 TFLOP/s |
| M128/N256/K128 CTA | 390.500-391.214 TFLOP/s |
| M256/N128/K128 CTA with ReLU/NVFP4 output | 382.892-383.817 TFLOP/s |
| M128/N128/K128 control | 397.529-398.057 TFLOP/s |
| M128/N128/K256 control | 406.753-407.192 TFLOP/s |

The slot sweep shows that the standalone FF1 kernel is already saturated;
launching more independent copies cannot combine into 1 PFLOP/s. Wider M/N
tiles also regress. SM120 compilation rejects cluster shape M2 at the official
CUTLASS builder assertion `no programmatic multicast on this arch`, matching
the official Blackwell GeForce constraint that cluster shape is 1x1.

K256 is the only positive tile result (+2.3% versus K128), so the native fused
kernel should use one complete d_model=256 producer reduction per hidden N128
slice. Reaching the end-to-end target now requires the back-to-back
FF1->ReLU->FF2 kernel and fused residual/LayerNorm/next-NVFP4 epilogue, not more
Stream1 slots or larger standalone tiles.

## Correct Cube4 end-to-end baseline and narrow-shape schedule A/B

A separate Molab build was configured for the actual Cube4 ABI
(`state_logical=96`, `state_physical=112`, `move_count=24`) and loaded the
public Cube4 output-24 ReLU checkpoint.  The earlier Megaminx 120/128 build is
not reused for this baseline.

At `B_micro=384`, two independent inference lanes, eager FP16 Stream1 measured
150.116 TFLOP/s dense-equivalent.  CUDA Graph replay measured 149.720 TFLOP/s,
so host launch overhead is not the dominant gap.  Enabling the three exact
final-CLS optimizations improved the same CUTLASS path to 169.182 TFLOP/s.  A
full sweep through `B_micro=24576` found no hidden large-batch win: throughput
falls to roughly 99.5 TFLOP/s.  The accepted FP16 control for subsequent
kernel A/B is therefore `384 x 2` plus exact final-CLS, not the larger
production accumulator size.

The first eager lane stage profile totals 1.1297 ms.  Across four blocks,
QKV+attention-out consume about 0.260 ms, attention about 0.244 ms, FF1+FF2
about 0.435 ms, and normalization/output about 0.191 ms.  Therefore FFN-only
fusion cannot by itself reach the 1 PFLOP/s end-to-end objective.

Exact Cube4 projection shapes at 21,888 token rows were benchmarked with dense
NVFP4.  The stock cooperative schedule reached 465.638/279.220/466.525/698.737
TFLOP/s for QKV/attention-out/FF1/FF2.  A new SM120 native ping-pong schedule,
paired with the required static persistent scheduler, compiled and ran in the
foreground and improved those rows to 523.243/278.747/559.328/699.780 TFLOP/s.
The gain is material for QKV (+12.4%) and FF1 (+19.9%), while both N256
projections remain unchanged.  This schedule is retained as an experimental
building block; it is not promoted to production before whole-block fusion and
quality gates.
