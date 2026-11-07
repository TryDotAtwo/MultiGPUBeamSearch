# Rust/CUDA runtime consolidation

Status: implementation in progress. GPU acceptance and performance defaults have not changed.
Baseline: published main `0eafb19369262a7d254058a0b4839bea7dfa3334`.

## Accepted requirements

- One active development branch and checkout: main. Preserve source, models, results, indexes and dirty overlays before retiring old worktrees.
- Rust owns configuration, sessions, rank launch, profile selection, history and result services. C++/CUDA owns GPU scheduling, kernels, CUDA Graphs, arenas, CUB and NCCL.
- Preserve existing specialized MLP/Transformer/GNN/blend kernels and numerical policies. Preserve permutation and packed CNOT domains independently.
- Preserve the CayleyPy public API, both Python package names, NativeSession, notebooks, torchrun/Slurm, model export and checkpoint validation.
- Narrow versioned C ABI with opaque owned contexts, coarse jobs and explicit status/cancellation. No exceptions/unwinding across FFI, no raw GPU pointer serialization between processes.
- Borrowed device views carry device, length, stride and lifetime. Static arena ownership and completion events govern reuse. Count real CUB temporary storage. No silent truncation or reallocation in the repeated GPU path.
- Keep Hash128 identity, CandidateMeta size/alignment 32, original Stream3 candidate IDs, persistent zero padding, exact global beam width and final A/B union.

## Implementation sequence

1. Verified Git/source recovery, compare with current main, retain experimental source separately, consolidate main and publish recovery references.
2. Rust workspace: beam-core, beam-native-sys, beam-runtime, beam CLI and PyO3 compatibility module. Lazy CUDA library loading; source/toolchain/shape/backend/ABI-bound build cache.
3. Move host session/configuration/profile/history ownership behind the native boundary while preserving reference behavior and public API.
4. Exact 128-bit radix sort; stationary Stream4 metadata with sorted indices; DeviceScan/BlockScan; stable owner radix partition; bounded bit ranges for score/route keys.
5. Paired real-hardware correctness/performance; enable optimized profiles only after acceptance. Sorted clean/dirty merge, NCCL device transport and separate membership-table experiments follow measured bottlenecks.

## Acceptance

CPU/reference comparisons, ABI/layout checks, tails/empty/overflow inputs, full-key adversarial cases, deterministic ties, exact beam width, path replay, repeated sessions, cancellation and rank failures. GPU sanitizer, actual two-rank collectives and per-architecture tests are separate evidence states.

Performance comparisons use the same model/input/precision/beam/rank count, warmup and at least five paired repetitions. Report cold launch, resident reuse, complete depths, VRAM and transfers. Default promotion requires at least 5% full-depth improvement or measured memory reduction with at most 3% slowdown. Unstable variants remain experimental.

## Resources

Total authorization: USD 50, including associated costs. Initial allocations: USD 10 T4/Ampere, USD 15 Hopper, USD 15 Blackwell, USD 10 reserve. User subsequently requested builds and tests on Vast; local operations are source/archive maintenance only, on drive D. Other projects' existing leases are outside this task.

## Progress and evidence

- Source/index recovery: 26 checkouts, 24,134 recovered files, 8,017 distinct archived objects. Independent Git bundle restore and full fsck passed for all 26 original checkout heads.
- One local main and one registered worktree now match the published baseline. Old directories, including ignored data, were retained under `.local/archived-checkouts/20261107/`.
- Build/cache directories moved under `.local/cache/pre-consolidation/`. Original model files retained at their canonical root paths.
- The first Rust/native integration stage now has CPU, one/two-GPU session and lifecycle evidence on RTX 3060. The specialized inference kernels and default public launch paths are unchanged.
- Native ABI/layout, copied job views, persistent worker groups, cancellation and one-rank failure are covered. Full details and limitations are in `test_results/refactor_20261107/STAGE2.md`.
- Host history (RAM/disk/static hybrid) now runs in Rust through borrowed ready pinned batches; reference GPU comparisons and a memory/leak sanitizer run passed within the documented API-reporting scope. Runtime calibration process ownership and PyO3 bindings are implemented and CPU-tested.
- Rust exact planning, bounded capacity search, full-step process/telemetry ownership and shortlist selection are implemented. Rust now owns independent inference probes and the complete microbatch search/refinement/selection loop, with native/LibTorch/ensemble comparison on one/two RTX3060. Public configuration/cache orchestration remains to migrate; defaults are not switched by the existence of these components.
- Rust now owns typed permutation replay, terminal/collection validation and compatible search sessions, with one request deadline and independent cancellation. Explicit native and LibTorch eager one/two-GPU comparisons passed, including model/beam changes and failures; see STAGE3_SESSION.md.
- Typed public options, artifact-bound inference jobs and validated profile-cache ownership now have Rust implementations and thin Python bindings. Coarse PyO3 calls forward notebook SIGINT to owned process cleanup. Downstream candidate/cache policy and the final public switch remain separate work.
- Native CNOT orchestration, CUDA bulk-operation improvements, architecture-wide acceptance and public launch switching remain pending. Source availability does not establish those acceptance gates.
- Rust downstream jobs now own component candidates, exact admission, source-bound caches and bounded full-step comparisons. Scalar row/parent conversion, startup cancellation, preserved capacity errors and bit-exact timing-cache round trips have regression coverage. Native and actual LibTorch eager one/two-GPU comparisons and the thin Python adapter are recorded in STAGE3_DOWNSTREAM.md. Maximum/working-width orchestration remains separate; the frozen baseline reproduces an overflow in a small-capacity profile despite successful VRAM admission.
- A complete Rust frontier job now connects inference, fixed-width admission and downstream control. Symbolic max records finite memory bounds, exact-width inference refinement and fresh same-profile working-width comparisons. Numeric widths are preserved. Native CUDA Graph and LibTorch eager one/two-RTX3060 fixture comparisons are recorded in STAGE3_FRONTIER.md; this does not switch the public launcher or complete CNOT/CUDA work.
- Full Hash128 CUB radix sorting is available only as an experimental build. Correctness and sanitizer fixtures passed on RTX3060, while the matched full-step fixture was slower; the default remains comparison sort. See STAGE4_RADIX128.md. Stream 4 index sorting and the remaining CUDA stages are still separate work.
