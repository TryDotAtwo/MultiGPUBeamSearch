# Experimental Rust MLP finite-value check

For a strict native `NativeModel` MLP, a Rust-owned snapshot already checks
source files and binds the retained snapshot to the workers. Python still read
each tensor in 1 MiB chunks, copied each chunk into a `bytearray`, made a
PyTorch tensor and checked its values for finiteness. The new validator borrows
the existing immutable Python `bytes` for the duration of a PyO3 call, checks
the FP16/BF16 exponent bits in Rust, and releases the GIL during the check.
File reads, shape/size checks, SHA-256 identity, snapshot validation and native
loaded-buffer binding still happen. No GPU pointer becomes a Rust reference and
the inference kernels and numerical policy do not change.

The new path is disabled by default. It requires all three session arguments
`engine="rust"`, `experimental_reuse_model_snapshot=True` and
`experimental_rust_mlp_finite=True`, plus an explicit MLP `NativeModel` and
`backend="native"`. A non-little-endian host or an older/missing
extension is rejected when the new flag is explicitly requested. Existing
sessions and old extensions retain the previous default validation path.

## Correctness and timing

- The final Linux release Rust workspace tests and PyO3 build passed; the full
  Python integration suite reported **670 passed, 11 skipped**. All 65,536
  bit patterns of each format were individually compared against PyTorch:
  131,072 comparisons matched. Odd buffers, invalid dtype and mutable Python
  input were rejected. The early tests also recorded failures before each new
  integration or opt-in contract was implemented.
- On one and two RTX 3060s, the existing public snapshot driver exercised first
  calls, repeat calls, unsolved inputs, model changes and width changes for
  Python and Rust owners. The source-bound verifier checked **20 cases**,
  independently replayed **16 returned paths**, and checked width against the
  saved native outcomes. Four additional groups covered native/auto binding
  transitions. A non-finite model source was rejected and the same Rust worker
  group was reused after the source file was restored.
- One two-GPU Rust session used the same untrained Hamming FP16 MLP (512-wide,
  eight residual blocks), input, width 4096 and six-depth limit for five
  alternating full-call pairs. The baseline uses PyTorch for the finite check;
  the candidate uses the Rust checker. Median wall time was **0.6010 s** versus
  **0.5896 s**; the median paired change was **−2.01%** (individual changes
  from −3.60% to −0.77%). Median source-validation time, including reads,
  SHA-256 and shape checks, was 41.53 ms versus 29.20 ms. Median native worker
  time was 302.91 ms versus 303.02 ms. The one cold candidate call with an
  existing compiled cache took 2.737 s; it is not a cold-build measurement.
- The source had **18,369,538 tensor bytes**. The new path avoids copying
  those bytes into Python `bytearray` objects on each full source validation;
  the initial file read and other required representations are still present.
  A single post-run VRAM sample was 339 MiB per GPU, not peak memory. No peak
  host memory, allocation trace, saturated GPU-step timing or CPU-wait trace
  was collected. This result does not meet the 5% full-step promotion rule;
  the new flag remains experimental and off by default. Other models,
  precisions, GPUs, multi-node execution and inference quality were not tested.

## Conditional CUDA graph diagnostic

The previously reported bounded Stream 4 sanitizer findings remain open. A
separate CUDA 12.8 example uses this repository's `Hash128` radix sorter,
CUB `ReduceByKey` with a simple sum, and a distinct shared-memory marker
kernel. It runs the same work in an ordinary graph and inside a conditional
graph. All four direct runs (1 and 1025 entries), both `memcheck` runs per
size, and all `synccheck` runs completed with zero errors. Both ordinary-graph
`racecheck` runs reported zero hazards. In the conditional graph with one
entry, `racecheck` reported **two cross-kernel hazards** between the marker and
CUB's block scan even though the program completed correctly. At 1025 entries,
the run under `racecheck` exited 11 without a usable success result; its direct,
`memcheck` and `synccheck` runs passed. An initial diagnostic compile failed
because the four state-specialization macros were missing; the retry supplied
the production values and both logs are retained.

CUDA documents `racecheck` as a detector of on-chip shared-memory hazards, and
shared memory is scoped to a thread block ([NVIDIA Compute Sanitizer
documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)).
The minimal reproduction is evidence that this tool/conditional-graph
combination can report a similar cross-kernel hazard. It does not prove that
the production Stream 4 graph is race-free, nor does it reproduce the earlier
bounded `synccheck` barrier finding. The bounded paths remain experimental.

## Source and cost

The final run used base commit `678d3d69cac42398f849b7722b776b2b8562ef43`
plus eleven SHA-256-pinned overlays. `rust-mlp-finite/verify_final.py` verifies
the embedded base against Git, overlays against the checkout, exhaustive CPU
results, GPU receipts and paired statistics. `evidence-final.tar.gz` has
SHA-256 `fb2ddaa554fb5cf4fcb52a8362efeff51ba747a852dd0435e3b1de0146cfd764`.
The earlier pre-flag experiment and its independent verifier remain alongside
the final archive. `verify_conditional.py` validates the diagnostic archive
SHA-256 `05d9e8c2919e8ec62dcf41bfd515f1a9bce49d7cfb395d7202067c6d6a26de7f`,
the included source, output logs and exit codes.

The first 2×3060 lease `60736013` stalled while loading the image and was
destroyed without task files. The second `60737291` ran both test revisions and
the diagnostic at $0.16726/hour. Fresh provider inventory confirmed both were
absent, and each task-specific guard was stopped. At query time the provider
reported **$0.008** and **$0.090** respectively, including recorded storage,
GPU and download charges where present; later billing adjustments are possible.
Elapsed GPU/disk estimates were at most $0.0254 and $0.0947 respectively,
excluding network. A first delete request for the second lease failed during
provider instability; a retry succeeded, and absence was verified through a
separate endpoint.
