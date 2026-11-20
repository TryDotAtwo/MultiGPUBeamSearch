# GPU acceptance source validation

- Source revision: `3b9e37b6c70dd4068ec46599802295011cd29b49`.
- Hosted Actions run: [44828781264](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44828781264).
- Python compile, remote Bash syntax, embedded Python syntax, archive-content guard, wheel force-install and venv provenance guards: passed.
- Source-validation artifact: [14675486620](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44828781264/artifacts/14675486620), SHA-256 `5e0653276e4cda72e9f7c5d98f01b711d56935572a690dce1f64ba63ab78c653`.
- The physical GPU job was skipped on the branch-push event. No CUDA, NCCL, installed-wheel GPU search, or checkpoint-scored CNOT execution is established by this run.
- Physical acceptance remains gated by a user-supplied spend cap and an already-running 2x RTX 3060 instance. The workflow does not start or stop the instance.
