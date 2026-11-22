# CNOT Rust initialization dispatch

## Change

The compatibility Rust stage adapter now owns initialization for typed
`PyCnotSearchInput`: rank 0 starts with the root and `lex=[0]`; other ranks
start with an empty root view and empty `lex`. Existing `cnot_owners`
selects the owner of the root. The existing rank-wide checker negotiation
coordinates owner-operation errors. Python-only and older providers keep the
`CnotStages.initialize` fallback.

Search transitions, scoring, rank ownership rules, exchange order, and the
public API are unchanged.

## Hosted evidence

| Run | Result | Artifact |
|---|---|---|
| [44603384206](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44603384206) | RED as expected at the Python initialization callback; 1 failed, 223 passed, 2 skipped | [14570450432](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44603384206/artifacts/14570450432) |
| [44604046633](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604046633) | rustfmt failed before compilation/tests; 605-byte diagnostic artifact | [14569182918](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604046633/artifacts/14569182918) |
| [44604475999](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604475999) | Rust/PyO3 CPU validation passed; 224 passed, 2 skipped | [14570037387](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604475999/artifacts/14570037387) |
| [44604945533](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604945533) | Corrected owner comparison; Rust checks, PyO3 build and compatibility suite passed; 224 passed, 2 skipped | [14570022898](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44604945533/artifacts/14570022898) |
| [44606108341](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44606108341) | Main after fast-forward; Rust checks, PyO3 build and compatibility suite passed; 224 passed, 2 skipped | [14569653026](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44606108341/artifacts/14569653026) |

Corrected artifact SHA-256:
`7192da508d6bfa3f05dbcf2b4ba05fd2f85c5a4d07ebbf1c7badd22297f30a84`.

The Actions log also passed Python syntax checking for the two-rank NCCL
probe. This does not execute the probe. Physical CUDA/NCCL and multi-GPU
runtime acceptance remain unverified. Main-branch verification passed on `acd3305ce4ccfa9389e24b3b596c76729ef0453d`. Artifact SHA-256: `b4ad12f9dda7dab2583a1a2b819d8a1447413a3b0e27d1d2f95ad3915f021ee5`.
