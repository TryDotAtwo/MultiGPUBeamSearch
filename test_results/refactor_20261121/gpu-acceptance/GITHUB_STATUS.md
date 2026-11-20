# GPU acceptance source validation

- Source revision: `ae9c6e81d32c2c3a7fb1402cdd14cc391345f7c3`.
- Hosted Actions run: [44829728978](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44829728978).
- Python compile, remote Bash syntax, embedded Python syntax, source-archive guards, forced wheel installation, venv import provenance, and remote result-artifact collection guards: passed.
- Source-validation artifact: [14674899394](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44829728978/artifacts/14674899394), SHA-256 `06f2ae40ae37b6224378dce805c1e527bab3d65533fca4f6f3efc09d73249b1a`.
- The physical GPU job was skipped on the branch-push event. This run does not establish CUDA, NCCL, installed-wheel GPU search, or checkpoint-scored CNOT execution.
- Physical acceptance remains gated by a user-supplied spend cap and an already-running 2x RTX 3060 instance. The workflow does not start or stop the instance.
