# CNOT descriptor recovery workflow wiring — status

Date: 2026-11-20

## Change

The existing-Vast CNOT workflow now builds the standalone owned-domain ABI
and runs both the two-rank Stream 5 control test and the corrected
`cnot_device_unseen_distributed.py` recovery test. It requires two success
markers from each rank before reporting a GPU pass. The workflow does not start,
stop, or rent the instance.

## Hosted workflow result

- Source commit: `96503061d83e0de088a38546a711b09da99f3a42`.
- Actions run [44705007455](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44705007455): workflow completed successfully.
- Vast status query authenticated with the configured Actions secret. The instance was stopped; source checkout, build, transfer, and GPU execution were skipped.
- Sanitized artifact [14614466680](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44705007455/artifacts/14614466680), SHA-256 `84ec4991f053a6914ae815526ab09af29c07f15723cec5135cf7db3e0131d512`.

## Acceptance

This validates workflow parsing, secret-backed read-only status access, and
the stopped-instance gate only. It does not validate CMake compilation, CUDA,
NCCL, descriptor-error propagation, or same-process-group recovery. The
physical two-rank check remains pending.
