# CNOT collective preflight — verification status

Source commit: `74f71ccef2895f96b716282e7ee29d9800c564dc`.

## Change

Rank-local preparation in CNOT selection, selected-record gather, row gather,
and Stream 5 fixed gather is coordinated over the existing CPU/Gloo control
group before the next search-data collective. The Python compatibility path
uses the same stage ordering. Search order, tie-breaks, scores, precision,
candidate IDs, ownership and NCCL payload/collective order are unchanged.

This protects against local preparation failures while Gloo remains usable.
It does not recover process loss or a failed/hung NCCL communicator.

## Verification

- RED: [run 44804192845](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44804192845)
  reproduced a rank-local histogram preparation exception escaping before the
  peer could safely advance.
- GREEN CNOT CPU/Gloo:
  [run 44812797417](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797417),
  308 passed, 2 skipped, 104 warnings. Artifact
  [14667597426](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797417/artifacts/14667597426),
  SHA-256 `057e12b7adc26b11fc4022acedffd17c0bbf786d0111222570398250ef3662be`.
- GREEN installed wheel:
  [run 44812797344](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797344)
  built and installed a PEP 517 wheel outside the source checkout; `pip check`,
  public exports and `resident-search-v1` passed. Artifact
  [14667767054](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797344/artifacts/14667767054),
  SHA-256 `cadfb46d6191211dbaf420ae3092ae7822ee16b5f4f42120f05dcd5288a7be4f`.
- GREEN Rust/permutation-depth CPU contract:
  [run 44812797266](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797266).
- The read-only
  [Vast run 44812797614](https://github.com/TryDotAtwo/MultiGPUBeamSearch/actions/runs/44812797614)
  found instance 61080396 exited and skipped checkout, transfer, build and GPU
  execution. It did not change resource state.

## Acceptance boundary

CPU/Gloo, PyO3 wheel installation, and CPU permutation-depth contracts passed.
No physical GPU was used. Physical single-GPU and two-GPU execution, the штатный
checkpoint scorer on GPU, and CUDA/NCCL failure recovery remain unverified.
No local build artifacts or result archives were created; no paid execution was
used.
