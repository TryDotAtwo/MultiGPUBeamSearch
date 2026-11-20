# Auto-max beam width session regression — status

Date: 2026-11-20

## Change

Commit `330728d070ff1f96aa3905ebf74fed0d46384b48` keeps the requested `"max"`
form in the fallback session identity and retains the accepted logical width
separately. Reuse checks compare the new terminal width to the accepted width,
for both the Rust-owned and Python compatibility resident paths.

Regression coverage includes ordinary public dispatch, Rust identity fallback
without the optional typed builder, and two consecutive calls in one explicit
session.

## Hosted result

- GitHub Actions run [44707125022](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44707125022) completed successfully.
- The public owner CPU/Python suite reported 86 passed, 3 skipped.
- Artifact [14614783619](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44707125022/artifacts/14614783619), SHA-256 `afbf5615341c6260dddc4254f1d8638c06026682dd6f24ef1dd9ff4b67bf03b6`.

## Acceptance boundary

The check ran on GitHub-hosted CPU with the optional native extension absent.
It validates adapter behavior and mocked Rust identity fallback, not Rust
compilation, CUDA, NCCL, physical multi-GPU reuse or full search equivalence.
