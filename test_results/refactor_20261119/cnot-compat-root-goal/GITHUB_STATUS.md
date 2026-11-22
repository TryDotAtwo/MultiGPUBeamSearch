# CNOT compatibility root identity verification

Code commit: `fcb6972650ccad97f86e69367c2fae2872bee2f2`.

The Rust-owned `CnotSearchInput` and `CnotMatrix` identity methods are used by
the compatibility CNOT stage when available. Older extensions and Python-only
inputs retain the `torch.equal` fallback. The regression covers identity and a
non-identity matrix and forbids `torch.equal` in both cases.

## GitHub Actions

- Test-first run `44429473842`: expected regression, 2 new failures, 165 passed,
  2 skipped. Artifact `14485139945`, SHA-256
  `1f3db547c2cb9fa4e0f7b283dfba55b01680a61af352a936a99a2535953751bd`.
- Green feature-branch run `44430494875`: Rust formatting/checks, PyO3 build,
  167 Python tests passed, 2 skipped, and NCCL probe syntax passed. Artifact
  `14485584309`, SHA-256
  `dd95918e93181f866c56ad5f0df833f2e705782bf81407f3f06184232cd73648`.
- Published `main` run `44431181310`: same checks passed. Artifact
  `14485953394`, SHA-256
  `8bf2df6d39b97417a6bc6b52cae3cd89d9941dc329242d9a9c2bd28aca884b13`.

Raw logs remain in GitHub Actions artifacts; no local test logs or build
outputs were created.

## Vast status

Read-only run `44427129602` authenticated with the existing repository
`VAST_API_KEY` and reported instance `61080396` as exited. The two-rank GPU
job was skipped. No start, stop, rental, source transfer, or GPU execution was
attempted. The existing Actions secret is already functional, so it was not
overwritten.

Physical CUDA, NCCL runtime, and multi-GPU CNOT acceptance remain unverified.
