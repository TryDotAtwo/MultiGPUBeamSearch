# Rust-owned permutation search input serialization

- Final source: main commit f9dd14a0c3f7b4d045bd37e2810a85340b787993.
- RED: [GitHub Actions run 44637299070](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44637299070) reached Rust compilation and failed because the new beam_runtime::search_inputs module was not implemented yet.
- GREEN: [GitHub Actions run 44640784456](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640784456) passed Rust formatting/checks, the PyO3 build, 240 Python tests (2 skipped), and NCCL probe syntax. Artifact [14586960403](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640784456/artifacts/14586960403), SHA-256 81c2e270527d8e2180c8a49f8dfcc3ef627a9bd2b62cbe7fb71ee64fd61a211b.
- Companion [permutation depth-control run 44640371591](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44640371591) passed.
- The compatibility path writes only the legacy permutation task files. No search loop, rank ordering, collective, inference arithmetic, CUDA kernel, or CNOT behavior changed.
- These are hosted CPU/PyO3 checks plus NCCL syntax only. No physical CUDA/NCCL multi-GPU runtime was run.
