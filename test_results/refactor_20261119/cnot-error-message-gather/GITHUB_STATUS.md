# CNOT fixed-tensor error diagnostics

Source: `8f4bfa93e338a1be0c968708f1c672ad590f2a18` on `main`.

## Change

The Rust CNOT controller now uses optional `run_cnot_native_checked_v2` for failure diagnostics. After the existing rank-wide error vote, ranks gather UTF-8 byte lengths and then zero-padded byte tensors in rank order through the existing fixed-size collective. The prior `run_cnot_native_checked` ABI remains intact for compatibility.

The Python `World.checked` path performs one cached all-rank capability vote before selecting v2. If any rank lacks v2, all ranks use the legacy Rust checker or the Python object-gather fallback. Search stages, scorer calls, matrix handling, winner selection and normal search ordering did not change. The first multi-rank checked call now includes the capability vote.

## Hosted verification

- RED run [44560022663](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44560022663): 217 passed, 2 skipped; the new test failed at the existing `torch.distributed.all_gather_object` call as intended. Artifact 14546707888, SHA-256 `f972e186dac67cfb2e4e4f7568a82c7e211281c3f85c6575d56ecca694ba4e33`.
- The first implementation run [44562279152](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44562279152) stopped at the Rust formatting gate before compilation. Follow-up commit `8f4bfa93e338a1be0c968708f1c672ad590f2a18` corrected the import formatting.
- GREEN run [44562629629](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44562629629): Rust formatting, selected Rust tests, `cargo check -p beam-python`, PyO3 build, 219 Python tests (2 skipped), and NCCL probe syntax passed. Artifact 14549203505, SHA-256 `53d387fea8c2694bf8b2c5d22aeae5771bf11394a6855ed18dfbf4b6576bc374`.
- Existing-instance status run [44562629392](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44562629392) found Vast instance 61080396 `exited`; GPU work was skipped, and the workflow made no start, stop, or rent request. Status artifact 14549093331, SHA-256 `d3409abd3c09bb18aac4b6f96a56bf9168b51dc1a93dbb37317c5c004f00295e`.

Full logs remain in GitHub Actions artifacts. No result archive or build output was downloaded to Windows.

## Limits

These checks validate Rust/PyO3 control flow and CPU collectives only. Physical CUDA/NCCL error propagation and multi-GPU recovery remain unverified.
