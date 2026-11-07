# Stream1 Transformer Accumulator Decoupling Plan

Baseline: `df534ebb1ef624abd42443211778b5588916d6d2`
Branch: `codex/stream1-transformer-accumulator`

## Global constraints

- Do not change Stream4 algorithm: threshold + compact + CUB/fixed-temp sort/reduce + compact remains.
- Preserve State128, Hash128, CandidateMeta, original candidate-id, and no-shard-top-k contracts from AGENTS.md.
- MLP behavior remains backward compatible and byte/counter equivalent under legacy configuration.
- A Transformer may use small inference `B_MICRO` while accumulating the same macro Stream3 batch as MLP.
- Logical accumulation chunks and physical accumulator/exchange buffers are distinct explicit concepts.
- Physical buffers are reused only after Stream3, ordered Stream5/NCCL exchange, and remote collect complete.
- NCCL collective round order, including zero-send tail rounds, must match on all ranks.
- Every capacity/overflow/tail condition fails closed; never silently truncate.
- TDD: each production behavior begins with a focused failing test.
- No public/Kaggle update until local tests and a private real 2xT4 validation succeed.

## Task 1: Formalize runtime and memory contract

Add explicit, backward-compatible runtime/profile fields for logical accumulation chunks and bounded physical accumulator/exchange buffers. Preserve legacy defaults when fields are absent. Make static memory allocate score/hash/parent/count by physical accumulator buffers and Stream5 send/recv by physical exchange buffers, while Stream3 scratch remains sized by the macro batch. Add fail-first runtime/static-memory tests for Transformer `B_MICRO=384`, macro batch around 1.5-1.8M, 2 accumulator buffers, bounded Stream5 memory, invalid divisibility/zero/budget, and legacy MLP equivalence.

## Task 2: Rebind dispatcher logical rounds to physical buffers

Change graph construction and depth scheduling so logical accumulation rounds map FIFO onto bounded physical accumulator buffers, and ordered Stream5 exchanges map onto bounded exchange buffers. A buffer becomes reusable only after Stream3 + Stream5 + remote collect. Cover full batches, final tail, multi-rank zero-send synchronization, overflow, and no-overwrite behavior with fail-first dispatcher tests. Preserve legacy outputs.

## Task 3: Profiles, notebooks, diagnostics, and local verification

Expose backend-independent macro Stream3 batch plus backend-specific inference `B_MICRO`; select bounded physical buffers automatically. Update public profiles/notebook preflight and debug reporting. Update memory/PROMPTS.md, memory/CHANGELOG.md, and test_results evidence. Run CPU tests, CUDA tests/builds, sanitizer where practical, and exact output comparisons.

## Task 4: Private 2xT4 A/B and publication

Privately compare the existing 2^26 control/winner against the decoupled accumulator configuration at depth 8. Record wall time, normalized throughput, jobs, VRAM, utilization, overflow/fatal evidence. Only after success update profile 26, validate the universal notebook for Transformer and MLP, commit/push, then update the public Kaggle notebook.