# GPU acceptance source validation

- Source revision: `30c1e204a2e048c5e56ccee17d3daa4006d653ce`.
- Hosted Actions run: [44831036050](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44831036050).
- Python compile, fake-SSH retrieval after key revocation (successful and failed remote cases), remote Bash syntax, embedded Python syntax, archive-content guards, wheel installation provenance, and result-artifact guards: passed.
- Source-validation artifact: [14675264978](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44831036050/artifacts/14675264978), SHA-256 `66fb835caa5cc9e1a897bb97777c18cccdffd9cb3092c3ec2e7908ae9181bfad`.
- The physical GPU job was skipped on the branch-push event. This run does not establish CUDA, NCCL, installed-wheel GPU search, or checkpoint-scored CNOT execution.
- Physical acceptance remains gated by a user-supplied spend cap and an already-running 2x RTX 3060 instance. The workflow does not start or stop the instance.
