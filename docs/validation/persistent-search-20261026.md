# Persistent native search

The opt-in NativeSession retains one local native process per GPU, uploaded
models, static device storage, streams and CUDA Graphs across compatible calls.
The resident production loop uses the existing per-task reset and recreates
search history. Start state changes reuse workers. Changed graph/model/binary,
selected devices, beam/depth or options invalidate the group before calibration.

CPU checks cover real subprocess mailboxes for two ranks, per-call log isolation,
worker failure and timeout cleanup, context ownership, model invalidation and
independent replay rejection. Existing backend/dispatch/preparation tests pass.

RTX 3060 12 GiB, real Cube555 Transformer + MLP FP16 blend (0.8/0.2), LibTorch:
- Two identical starts used PID 1244; both replayed [16,8,1].
- A changed start reused PID 1244 and replayed [11,0].
- Changing beam 4096 to 8192 rebuilt the group (PID 1264).
- All workers terminated on context exit.
- With autotuning enabled, the second call succeeded with both calibration
  functions replaced by functions that raise if called.
- A fresh session reused both validated inference and pipeline disk profiles.

One-GPU evidence archive SHA256:
0683a7ffac15c6367b85ebc53c514199302c775c90109e1b68769ee0ab2650b3
All 477 manifest members were hash-verified. No long solve or performance sweep
was run. Multi-GPU and CUTLASS acceptance are still pending at this checkpoint.

This change does not certify the unrelated maximum-working-frontier candidate,
128-GPU operation, or a persistent service across different physical machines.

Two RTX 3060 host: both resident and cold control failed before search at
CUB MergeSort during CUDA Graph capture (invalid device ordinal), on driver
535.161.07 and CUDA 12.8. This does not establish multi-GPU acceptance.
All 357 archived evidence members verified; archive SHA256
9c88db88241508ccaf83baae6329aa6aaf5733e33c399788946fcf868b316aff.
The owned test instance was deleted and its absence verified.
