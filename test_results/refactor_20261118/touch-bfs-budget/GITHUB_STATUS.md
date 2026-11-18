# Rust touch-BFS entry budget

Latest source commit: `922b1f7eb588441d8b2978b2d0a28cacda96d54c`.

- Hosted workflow [44218335727](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44218335727) passed Rust formatting, 3 `beam-core` CNOT tests, 5 `beam-core` search tests, 4 `beam-runtime` CNOT tests, 4 `beam-native-sys` scorer tests, `cargo check -p beam-python`, the PyO3 extension build, 44 Python tests, and NCCL probe syntax compilation.
- Full log artifact [14388638855](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44218335727/artifacts/14388638855), SHA-256 `c2e9cc7ba1cf5d4922bae8eb9c127679800f2ba2846d353ab30d643db557c8fc`.
- The tests-first run [44217173408](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44217173408) passed the existing checks and 39 Python tests; the two new tests failed because the wrapper did not delegate and the extension lacked the new method. Run 44218056948 found only rustfmt differences; these were corrected in commit `922b1f7e` before the successful run.
- This validates CPU control arithmetic, Python compatibility and extension integration only. No CUDA kernel, NCCL communication, physical multi-GPU search, or performance test ran.
