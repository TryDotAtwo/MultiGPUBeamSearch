# CNOT distributed cancellation validation

Date: 2026-11-20  
Source commit: `1f331663cc3368b1e7a357abc5a4d5bb7b390301`

## Regression and change

Before the fix, a rank-1 `KeyboardInterrupt` during CNOT root-goal checking
was converted to `RuntimeError`. The CNOT stage now polls Python signals
before control-group votes, includes goal-stage cancellation in the existing
rank-wide diagnostic exchange, and raises `KeyboardInterrupt` on every rank.
The compatibility Python fallback uses the same exception class. Ordinary
distributed errors still raise `RuntimeError`.

Search ordering, scoring, precision, rank ownership, and NCCL payload
collectives were not changed.

## Remote checks

- RED: [Actions run 44800867787](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44800867787)
  showed `RuntimeError: check CNOT root goal failed: rank 1: KeyboardInterrupt:`.
  Artifact 14660528740, SHA-256
  `28d7f9a585c99d7171055b26ccb20e541c119aece309ed8a6523c2ef3e01f10c`.
- GREEN: [Actions run 44802610199](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44802610199)
  passed Rust CNOT checks, the PyO3 extension build, and the two-rank CPU/Gloo
  CNOT suite: 305 passed, 2 skipped, 104 warnings. The regression confirms
  both ranks catch `KeyboardInterrupt` and complete a follow-up Gloo
  all-reduce. Artifact 14660759672, SHA-256
  `bad79e5f3ee7e6a3a476afef2bb7754992137564399ba4568446710aba16a889`.
- Installed wheel: [Actions run 44802610026](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44802610026)
  passed the PEP 517 build, installation outside the source checkout,
  dependency check, public imports/exports, and worker protocol smoke checks.
  Artifact 14662495219, SHA-256
  `ef7d87e8639e80c90c6ec765d599bddb26275c32fd60dedef424a2f974cb81a7`.
- Physical run: [read-only Vast Actions run 44802610013](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44802610013)
  observed instance 61080396 in the exited state. It skipped all remote build
  and execution steps; no start, stop, or rental was attempted. Artifact
  14661622569, SHA-256
  `a499417bc4f690641a8e6d43f1c55299b6ffe25819a463e2ea78061885bcf2fc`.

## Acceptance boundary

This is CPU/Gloo and installed-wheel evidence. Physical CUDA/NCCL cancellation,
GPU checkpoint scoring, and permutation/CNOT search on physical 1 and 2 GPU
remain unverified. The regression injects SIGINT at a CNOT root-goal stage
boundary; it does not test a signal arriving inside a CUDA kernel or an
in-flight NCCL data-plane collective.
