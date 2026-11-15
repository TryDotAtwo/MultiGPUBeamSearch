# Experimental native CNOT scorer packet

## Change and contract

The optional `mgbs_tqp_score_many` entry point submits a complete packet to the
existing scorer and runs its ordinary bounded inference sub-packets in native
code. Python enables it explicitly with `experimental_native_batch=True`; the
ordinary per-sub-packet ABI remains the default. Requesting the new path against
an older library fails before opening a checkpoint. There is no change to the
model, coverage checks, precision, or complete CNOT search orchestration.

The ABI checks the caller-declared byte extents and pointer-offset overflow
before the first sub-packet. A short extent reports the named buffer and its
required/available bytes. The caller must provide live, disjoint allocations on
the correct CUDA device with truthful lengths and retain them until their work
finishes. The ABI cannot prove those allocation properties. Python allocates
separate score and coverage tensors and retains the existing contiguous-input
conversion, which can copy a strided input. The native packet interface adds no
packet copy; existing scorer feature and output operations still execute.

## Source and environment

The evidence archive is SHA-256
`ed1a73419a5e23e35ce1cf771b53aefaace846f6a8ba408b99bc7e3a296166b5`.
Its source manifest names base commit
`2efd9a9bde065aca88e5f06394dd31d8097dbfc0` and five overlays. The public
verifier checks 539 baseline files against that Git commit, checks each overlay
against the current source, and checks the remote before/after source receipts.

The scorer donor was the pinned TheQuantumProject commit
`c5b3e3f93b45aa7e533fd2c06209c28ce516ffeb`. Before transfer, 621 donor files were
compared with the private historical source manifest. Two JSON files had CRLF
differences, with the original working bytes matching the manifest and their
normalized bytes matching Git. The test host checked the overlaid bridge again.
The checkpoint was the original one-update GL(4,2) fixture with SHA-256
`92f3ec8e6a6ef831916a9d3ba9bbdae9307c9850d61334916a2b3e690c0020c4`.
The donor code, checkpoint, compiled library and private fixture are not in the
published evidence archive.

The Vast host had two RTX 3060 12 GiB GPUs, driver 595.104.02, CUDA 12.8,
PyTorch 2.9.0+cu128, and Rust 1.100.0. The private scorer was built for SM86.
The evidence does not establish behavior on other GPU architectures.

## Verification

- The Linux Python integration suite reported **670 passed, 17 skipped**;
  its ordinary run did not load the private checkpoint.
- Each physical GPU separately reported **3 passed, 1 skipped** for the
  checkpoint scorer. Qhat, proposal and V outputs of the two paths matched
  within `rtol=atol=2e-5` for empty, partial, full and multiple sub-packets,
  including non-contiguous input. A coverage failure retained its row/feature.
  Raw ABI tests exercised zero rows, invalid arguments and each short buffer.
  The separate comparison with upstream canonical CLI outputs was skipped.
- One- and two-GPU checkpoint-backed searches matched their baseline results
  exactly. The verifier independently decodes and replays the construction path
  `[5, 7, 0]` on all three rank receipts. These are bounded correctness fixtures,
  not coverage of every dimension, checkpoint, topology or failure mode.
- `cnot-scorer-batch/verify_results.py` verifies the archive and these receipts;
  `cnot-scorer-batch/verification.log` records its output. The optional ABI is
  also tested to fail closed when an older scorer lacks the symbol.

## Complete-search timing and decision

The same checkpoint, matrix, `n=4`, width 2, shortlist 3, inference sub-packet
size 2 and depth limit 3 were used for both variants. Each configuration had
two warmups followed by five alternating pairs. Whole-search wall time includes
the existing search and communication; the two-rank value is the slower rank.
The table reports the median of each set of five times and separately the median
of the five per-pair relative reductions `(old-new)/old`.

| GPUs | Old median | New median | Median paired reduction |
| --- | ---: | ---: | ---: |
| 1 × RTX 3060 | 223.020 ms | 223.502 ms | -0.254% |
| 2 × RTX 3060 | 606.283 ms | 599.633 ms | +0.031% |

This tiny fixture is not a saturated production profile, and no VRAM,
allocation, host-wait, or cold-start comparison was collected. Neither result
meets the 5% improvement gate; the option remains experimental and disabled by
default. No inference or CUDA search kernel was replaced in this change.

## Cost and cleanup

The task lease was `60751435`, two RTX 3060 at $0.1369/hour including the
offered storage rate. It was destroyed, and a separate provider inventory
confirmed absence. The release receipt is in `cnot-scorer-batch/release.json`.
At the recorded invoice query, its GPU, storage and download charges totaled
$0.033. The query found $4.068 for 26 known refactor lease IDs; that filtered
subtotal can lag and is not an audited overall project budget or final invoice.
