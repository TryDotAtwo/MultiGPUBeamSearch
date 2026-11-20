# Installed wheel validation

- Source commit: `238acd87d22ba7a08c91108effb331578bebe6ff`
- Workflow: [run 44833720083](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44833720083)
- Result: success on GitHub-hosted `ubuntu-latest`, Python 3.11.18. The PEP 517 wheel built, installed from outside the checkout, and passed `pip check`.
- Installed provenance: package and compatibility package resolved under `site-packages`; the compiled `_beam_runtime.abi3.so` and executable `beam-runner` were also loaded from the installed package. Rust ABI 1 and resident protocol `resident-search-v1` were reported.
- Rust session smoke: the compiled PyO3 `SearchSession` ran twice with a deterministic CPU fixture worker. Both returned path `[0]`; the second reported `reused=true` and `request_count=2`. The smoke also checked preparation identity serialization, including the `beam_width="max"` request. This directly exercises the installed extension, but does not call the public permutation `beam_search` entry point.
- CNOT smoke: `tools/verify_cnot.py --case wide --backend gloo --engine rust` completed through the installed wheel. It used CPU/Gloo, world size 1, `WireFixture` scorer, n=144, beam width 1, and returned a two-gate path with `bitwise_replay=true`.
- Artifact: [installed-python-wheel-validation.zip](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44833720083/artifacts/14677568145), artifact ID `14677568145`, SHA-256 `6bbc4646c42aa298518bad861b74807c89bb64489b4c3e15d252d1f13a5ea565`, 8,488 bytes.

This is CPU-only evidence. It does not validate physical CUDA/NCCL, multi-GPU execution, the trained TQP checkpoint scorer, or a public permutation search call.
