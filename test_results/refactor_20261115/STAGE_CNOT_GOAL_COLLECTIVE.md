# Collective CNOT goal errors

## Change

Previously, a rank could raise while looking for a one-gate completion and leave
its peers waiting in the following goal exchange. Goal evaluation now catches a
local exception and carries a failure flag in the existing fixed-size exchange.
When any flag is set, ranks collect and report the rank-specific messages. A
single-rank search retains the original exception. The successful path keeps the
same collective and host read; its exchanged payload grows from two to three
64-bit values per rank.

This addresses recoverable local exceptions when the device and process group
can still exchange the flag and error messages. It cannot recover a fatal CUDA
context or NCCL failure, an inability to allocate the status payload, or a
failure of the collective itself. Other stages have their own error boundaries;
this change does not claim that every distributed error is recoverable.

## Source and results

The baseline is commit `7c25b3c6ecd0becdcbd8d819a71bb575eaad3912`.
The archive SHA-256 is
`b55adefe2b3aff473e77851cfb64257a340896f93a43a170aab118bc67212135`.
`cnot-goal-collective/verify_results.py` checks 541 baseline files against that
commit, three overlays against the current source, the original goal function
used by the paired benchmark, and the before/after source receipts.

- On the original implementation, the new Linux regression reported **4 failed,
  1 passed**: a local failure skipped the exchange and the two-process Gloo
  cases failed for rank 0, rank 1 and simultaneous failures. The fixed Linux
  integration suite reported **675 passed, 17 skipped**, including those cases
  and successful reuse of the process group. A focused local CPU run reported
  18 passed with 4 deselected; no local GPU load was required.
- On two RTX 3060 12 GiB GPUs, CUDA 12.8, driver 595.91.07 and PyTorch
  2.9.0+cu128, fault injection inside an actual search separately failed rank 0
  and rank 1. Both ranks received the same error, then each pair ran a new
  search in the same default NCCL process group. The verifier independently
  replays the four rank receipts for the recovered one-gate solution.
- Normal one- and two-GPU searches agreed exactly with the baseline result on
  every timed call. The two-GPU synthetic case expanded parents on both ranks.
  It used `WireFixture(9)`, not a trained checkpoint, and reached the depth
  limit; the separate recovery case supplied the replayed solution.

## Paired whole-search timings

The small GL(4,2) fixture used width 5, shortlist 9, depth limit 4 and three
completed layers. Each configuration had two warmups and five alternating
pairs; the two-GPU time is the slower rank. The final column is the median of
`(old-new)/old` across the five pairs, so a negative number means a slowdown.

| GPUs | Old median | New median | Median paired change |
| --- | ---: | ---: | ---: |
| 1 × RTX 3060 | 21.502 ms | 21.533 ms | -0.016% |
| 2 × RTX 3060 | 533.887 ms | 534.495 ms | -0.133% |

This is a bounded regression measurement, not evidence of throughput for a
saturated trained-model workload or other GPU architectures. It does not move
the CNOT search loop into Rust.

## Cost and cleanup

The task lease `60754090` was created at $0.1348/hour including the offered
storage rate, destroyed after evidence collection, and independently confirmed
absent. At the recorded invoice query, its compute, storage and download charges
totaled $0.015. The query found $4.083 for 27 known refactor lease IDs; this is
a filtered current subtotal, not an audited overall project budget or final bill.
