# Rust resident cancellation

Date: 2026-11-20

## Scope

Expose active-search cancellation through Python `NativeSession` to the Rust resident owner. A cancel request is terminal for that session. Idle sessions and Python-engine sessions return `False`; older Rust extensions without the method remain supported.

## Verification

- RED, [Actions run 44719860829](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44719860829): 86 passed, 3 skipped; the new public API test failed because `NativeSession.cancel` was absent.
- GREEN, [Actions run 44721590584](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44721590584), tested commit `327425a2c79c1d717a4a4a0ea0c23191302eb6e3`.
- Rust `search_session`: 3 passed, including a two-rank worker cancellation test that confirmed both ranks received the request, observed the stop marker, exited, and were reaped.
- Python compatibility suite: 280 passed, 2 skipped. Rust checks and PyO3 extension build passed.
- GitHub Actions artifact 14620999670, SHA-256 `3930714e9560a28ec895af213bfafc190a5feed85423371b08f412211cfc8122`.

This is CPU process-level evidence. It does not verify CUDA kernels, NCCL behavior, installed-wheel execution, or physical 1-/2-GPU runs.

## Verification on main

- [Rust public owner CPU validation, run 44722196881](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44722196881): success.
- [CNOT CPU compatibility, run 44722196693](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44722196693), at `014a2ffeaab5493dd0e0363c93a034fe09413381`: success.
- Rust resident session tests: 3 passed. Python compatibility suite: 280 passed, 2 skipped. Rust checks and PyO3 extension build passed.
- Main artifact 14621863125, SHA-256 `c33942672d0c4a24c3845feb5e733236fb6512b731f6fc8ff79fd3729f706840`.

This confirms CPU process-level cancellation and compatibility. Physical CUDA/NCCL, installed-wheel execution, and physical 1-/2-GPU runs remain unverified.
