# Owned CNOT input snapshot

Previously CNOT admission validated a host copy of the caller's tensor, but the
identity shortcut and search still read that caller-owned tensor. If it changed
after validation, the rank could search a different matrix. In particular, a
one-gate input changed to identity returned an incorrect empty solution without
replaying it. Changing an identity input into a one-gate input instead failed
the later reconstruction check. Both Python and Rust input validators shared
this discrepancy.

After validating shape and search parameters, each rank clones its input into
an owned contiguous tensor on the same device. The host validation bytes, rank
fingerprint, identity shortcut and search all derive from that tensor. The
copy happens once per search and occupies at most 8192 bytes for 256 wires; the
host validation snapshot existed before this change. There is no new per-child
copy. The caller must synchronize external writes during the initial capture:
the clone is not an atomic transaction across independent GPU streams or host
threads. The Python search loop and checkpoint scorer remain in place.

## Verification

- A local regression deliberately mutated the caller tensor immediately after
  validation. Before the fix, all four original cases failed, including the
  incorrect empty solution. After the fix, eight cases passed across Python and
  a Rust-validator stub, identity/one-gate inputs and contiguous/strided views.
  Windows PyTorch's `pyreadline3` emitted an ignored console-destructor error at
  interpreter shutdown after the successful run; the full logs are retained.
- On the remote Linux image, the release PyO3 module was rebuilt from source and
  its independent CNOT validator/replay checks passed. The full integration
  suite reported **651 passed, 11 skipped**. Public searches with the actual
  Rust validator matched the Python validator on one CPU rank, two CPU/Gloo
  ranks, one RTX 3060 and two RTX 3060/NCCL ranks. A separate receipt verifier
  independently replayed 12 returned paths and checked 24 strided-input
  mutation cases across those ranks. Validator mismatch and rank-1 singular
  input were rejected collectively in the two-rank runs.
- These are synthetic CNOT fixtures. A trained checkpoint scorer, other GPU
  families, multi-node execution, Compute Sanitizer and speed were not tested
  in this stage. The result establishes neither a speedup nor completion of
  the Rust/CUDA search migration.

## Provenance and cost

The base is `3592a50f10db30ac64cb1832ba95459209937972` plus the three
source/test overlays recorded in the evidence manifest. The archive SHA-256 is
`10496e4c7f7dcf4daf88886a4f6a4d34cf25818a6bacbfbf4c67ce3140be5b0b`.
`cnot-snapshot/verify_results.py` checks the archive, 533 base files against
Git, all three overlays, test results, rank receipts and path replay. Raw red
and green local outputs are retained beside it.

Vast lease `60725966` used two RTX 3060s at $0.134815/hour. The evidence was
downloaded and verified before deletion; a fresh provider inventory confirmed
the instance absent, and its task-specific timeout guard was then stopped.
Estimated GPU/disk cost is **$0.0157–$0.0164**, excluding network charges and
invoice adjustments.
