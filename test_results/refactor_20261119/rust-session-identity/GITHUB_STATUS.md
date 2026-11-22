# Rust session preparation identity migration

Code is on `main` at `9f9f0168b2cbfd03965816c9a0ec475acf30be12`.

## Verification

- Test-first run [44434253940](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44434253940): 76 passed, 3 new expectations failed, 1 skipped. The failures showed the missing Rust-owned preparation-key API and fallback adapter.
- The first implementation run stopped at `rustfmt --check`; it did not reach compile or tests. The source was normalized and the workflow was rerun.
- Green [CNOT CPU compatibility run 44436290093](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44436290093): Rust checks passed, PyO3 extension built, 174 Python tests passed, 2 skipped, and the two-rank NCCL probe passed syntax validation. The focused Rust session test reported 1 passed.
- Green [permutation depth-control run 44436290039](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44436290039).

Full logs are retained remotely in [artifact 14489551078](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44436290093/artifacts/14489551078), SHA-256 `8cd129ed1be1914f21fab080b50282954094d1694ae7dedf362ba51d3e380e8b`.

## Scope and remaining checks

Rust now owns the Rust-engine prepared-session cache key and equality decision. The Python adapter still builds the existing descriptor; old PyO3 extensions and the explicit Python engine keep the Python comparison. Search stages, result format, rank topology, CUDA calls, and NCCL calls did not change.

These hosted checks do not establish physical CUDA/NCCL or multi-GPU search acceptance. The Vast status check [44427129602](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44427129602) confirmed the existing `VAST_API_KEY` secret was available and the instance was stopped; no GPU work was started.

## Corrective recovery, 2026-11-19

The earlier run 44436290093 built and ran the replacement one-test `search_session.rs` file, so it did not exercise the original session regression suite. The recovery commit [f2d9696](https://github.com/TryDotAtwo/MultiGPUBeamSearch/commit/f2d9696cda1f68a56a3bb59686f755409032ee2f) restores the original file and moves the preparation-identity regression to a separate test binary.

Corrective CNOT CPU run [44442228498](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44442228498) passed Rust checks, the original session suite (2 passed), the new identity test (1 passed), PyO3 build, Python compatibility (174 passed, 2 skipped), and NCCL probe syntax. Logs are in artifact [14490574306](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44442228498/artifacts/14490574306), SHA-256 `103d628d22a539f08646ce37e58d89ce9d45ecb4c0d1d1e7598bfbaf8098ef68`.

The full architecture contract was restored from the parent snapshot of the truncating 2026-11-15 commit, with subsequent Rust/CNOT updates retained in a dated addendum. Physical CUDA/NCCL multi-GPU runtime remains unverified; the existing `VAST_API_KEY` secret was already present, so this repair did not overwrite it or start a GPU instance.
