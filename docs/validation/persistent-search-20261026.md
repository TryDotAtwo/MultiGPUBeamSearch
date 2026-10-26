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
was run. Multi-GPU LibTorch acceptance is recorded below. A separate CUTLASS resident
GPU acceptance was not completed.

This change does not certify the unrelated maximum-working-frontier candidate,
128-GPU operation, or a persistent service across different physical machines.

Two RTX 3060 host: both resident and cold control failed before search at
CUB MergeSort during CUDA Graph capture (invalid device ordinal), on driver
535.161.07 and CUDA 12.8. This does not establish multi-GPU acceptance.
All 357 archived evidence members verified; archive SHA256
9c88db88241508ccaf83baae6329aa6aaf5733e33c399788946fcf868b316aff.
The owned test instance was deleted and its absence verified.

Kaggle 2xT4 found a controller-result contract bug in the first implementation:
all ranks completed, but only the reconstruction controller prints a solved
terminal record. Fixed in 7c38e2983f16a1c9a941b565e48365cafa3f6648.
The transport still requires every rank completion receipt. Solved results
require one controller record; unsolved results require one record per rank.
Targeted one-/two-rank controller contract checks: 2 passed.

## Completed Kaggle 2xT4 acceptance

Source 7c38e2983f16a1c9a941b565e48365cafa3f6648, private validation v4:
- Simple public-notebook flow completed from a clean GPU environment. Two calls
  retained PIDs908/909, returned independently replay-validated paths, and both
  processes were gone after explicit close.
- Real Cube555 Transformer+MLP FP16 blend0.8/0.2, LibTorch, beam4096/depth4:
  five calls retained PIDs756/757. Paths were [16,8,1], [16,8,1], [11,0],
  not-found, [16,8,1]. Every found path independently replayed to the center.
- Increasing beam to8192 created PIDs811/812 and returned [16,8,1]. Both new
  processes were gone on context exit. No stale stop flag/history after not-found.
- Solved calls had one controller terminal record; not-found had two rank records.
  Every request required matching completion receipts from both ranks.
- GPU tests were short correctness/lifetime checks, not a performance campaign.

Both Kaggle validation kernels reached COMPLETE. The first two setup attempts
used a CPU Docker pin and failed preflight; the GPU image pin is now corrected.
The previous full Cube555 notebook160 solve remains separate evidence; it was
not repeated for this change. Compatible profile disk reuse and forbidden warm
recalibration were separately verified on RTX3060 as described above.
