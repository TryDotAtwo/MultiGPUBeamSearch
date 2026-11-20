# Session identity and CNOT exchange verification

Tested source: `d1ed3e163e1d8a0e3d7e9da214fc6889be35df18` on `main`.

## AutoMax session identity

- Rust public-owner CPU run [44734461453](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44734461453): 90 passed, 4 skipped. Artifact [14627089168](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44734461453/artifacts/14627089168), SHA-256 `afdd2192ec1d2bae5359eba447f3b11565bfd3a22f3d75e64a5df5c5e7c3a915`.
- CNOT CPU/PyO3 run [44734461316](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44734461316): 283 passed, 2 skipped. Artifact [14627533143](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44734461316/artifacts/14627533143), SHA-256 `55cf8d87717abb7c2f82ffdad34038818b0cdd021a9a9586a94f4a5436728b51`.
- Installed wheel run [44734461301](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44734461301): passed, including the two-call PyO3 session smoke. Artifact [14627961945](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44734461301/artifacts/14627961945), SHA-256 `64235dd58a940d97cc761c3c6ed76072b1cb851f433549bb355de62753dffd5b`.

## CNOT engine agreement and receive allocation

- Two-rank Gloo engine-agreement run [44736922496](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44736922496): 284 passed, 2 skipped. It verifies mixed `auto`/`python` requests fail collectively before search. Artifact [14629311464](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44736922496/artifacts/14629311464), SHA-256 `47a2d00a8de967c8c95460afc3b3acd7afbc9dbd8189a1891e6aef2c083766e4`.
- Final CNOT CPU run [44738874159](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874159): 294 passed, 2 skipped, 104 warnings; Rust `cnot_exchange` passed 6 tests, including local and peer receive-allocation failure before payload. Artifact [14628794155](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874159/artifacts/14628794155), SHA-256 `a16b41766901ffd35e1a32c9d2ca565ed6db5d7fe544244402707837cb3dd9c1`.
- Final installed wheel run [44738874151](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874151): passed. Artifact [14629797227](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738874151/artifacts/80f6cc4467937e301bf42ad1ebcd51fa38ee7bc535217d8cb3b09e779df4cc36).

## Physical GPU status

Vast status run [44738618500](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738618500) checked instance 61080396 and reported `actual_status=exited`. GPU checks were skipped; the workflow made no start, stop, or rent request. Artifact [14629413208](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44738618500/artifacts/14629413208), SHA-256 `c890e78be962175d3479e0c36242b985f8af941150db9c291f5720c5a8d0c622`.

No physical CUDA/NCCL run or one-rank device-allocation fault injection has passed. The new allocation-failure evidence is Rust controller plus two-rank CPU/Gloo coordination.
