# Typed CPU CNOT first-unseen

## Scope

Move typed Rust compatibility-path CPU selection of first unseen packed CNOT matrices into beam-core. Preserve full-state equality, row-tail masking, generation-order winner, int64 result shape, CUDA kernels, Python-only fallback, and multi-GPU protocol.

## Evidence

- RED: [GitHub Actions run 44671005907](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671005907) reached the typed CPU path and failed exactly because the Python cnot_first_unseen helper was called; 267 passed, 1 failed, 2 skipped. Artifact [14600765521](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44671005907/artifacts/14600765521), SHA-256 09684db96085d6ee9bef6b6e3b7802c023c5c061ebfc54090f04387f8b939409.
- GREEN: pending hosted validation.
- Physical CUDA/NCCL multi-GPU execution: not run. Paid compute: not used.
