# Typed CPU CNOT first-unseen

## Scope

Move typed Rust compatibility-path CPU selection of first unseen packed CNOT matrices into beam-core. Preserve full-state equality, row-tail masking, generation-order winner, int64 result shape, CUDA kernels, Python-only fallback, and multi-GPU protocol.

## Evidence

- RED: [GitHub Actions run 44671005907](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671005907) reached the typed CPU path and failed exactly because the Python cnot_first_unseen helper was called; 267 passed, 1 failed, 2 skipped. Artifact [14600765521](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671005907/artifacts/14600765521), SHA-256 09684db96085d6ee9bef6b6e3b7802c023c5c061ebfc54090f04387f8b939409.
- Hosted attempt [44671574865](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671574865) stopped at rustfmt on one line-wrapping difference in the PyO3 adapter; no Rust tests or build ran. Formatting is corrected. Artifact [14599897789](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671574865/artifacts/14599897789), SHA-256 4b81553d8e059c868d9b7fd7b86db5cdc52b5031c908eeb2acb968aa1bd5216f.
- Hosted attempt [44671760449](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671760449) stopped at rustfmt on a blank line and public helper signature; no compile/tests ran. Both formatting differences were corrected. Artifact [14600586872](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671760449/artifacts/14600586872), SHA-256 9ad89211797baeafa74bc08201f7cfa2675aa3811da92cfde7ab23819322d482.
- GREEN hosted run [44671946779](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671946779) passed formatting, Rust tests/checks, PyO3 build, 268 Python CPU compatibility tests (2 skipped), and NCCL probe syntax. Artifact [14600462371](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671946779/artifacts/14600462371), SHA-256 92eaa60b94e5a1ca1e96f90f38e12385186db6a801ee28a83e759a67b1675db6.
- MAIN GREEN at commit 7a2e679f66d903b6bd309e00006078f1f2617c1c: [Actions run 44672860884](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44672860884) passed Rust CNOT checks, PyO3 extension build, 268 Python CPU compatibility tests (2 skipped, 98 warnings), and NCCL probe syntax. Artifact [14599974834](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44672860884/artifacts/14599974834), SHA-256 a7eb4010f90efde25bdc3e057e76f73ca3447fa9f345b5688c4330d4191610ca.
- Full logs remain in GitHub Actions. Physical CUDA/NCCL multi-GPU execution was not run; paid compute was not used.
