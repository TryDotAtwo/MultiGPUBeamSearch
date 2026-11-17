# Standalone CNOT borrowed-device expansion

The optional device ABI now provides a thread-bound device-only context and
passes borrowed GPU views to the existing CNOT child-generation kernel. The
context records capacities and owns a stream without allocating the base ABI's
private parent/child/descriptor/CUB buffers. Host-buffer methods reject this
context; full contexts retain their prior behavior and may also use the borrowed
operations. The caller guarantees allocation bounds, context accessibility,
lifetimes, stream ordering, nonoverlap and valid descriptor values. Both int64
and uint16 physical moves are supported. Invalid descriptors retain the
existing CUDA kernel's device-error behavior.

This is standalone and not connected to the public CNOT search. The public
Torch stages, checkpoint scorer and multi-GPU collective order are unchanged.
This result does not establish that the complete beam search has been ported.

The source-pinned run used base `3d029d303b683e5882a1e4163d496a45248f96d7`
and six overlays. Only the plan document's status was edited after execution;
the five code/test overlays are checked against the archived manifest. The
archives under `cnot-borrowed-expansion/` retain these stages:

- The first source-pinned run built the old and new libraries and completed
  the domain, borrowed-owner, expansion, CUDA sanitizer, Rust workspace and
  new Rust GPU checks. Its old-owner compatibility test failed because the
  test was given the previous library, which already supported device owners;
  this was a fixture error and the script exited with status 101.
- The continuation verified the same source manifest and library hashes,
  built the actually older `4846dd9d222cfb09a0df4d1acd556ff81dda7ac2`
  version, and passed the old-owner compatibility check. It then passed Rust
  memcheck, one/two-GPU search and checkpoint scorer comparisons, the Python
  suite and cross-compilation. It exited with status 0 and reverified both
  source manifests after the run.

On two RTX 3060 12 GiB GPUs, CUDA 12.8, driver 595.91.07 and Rust 1.100.0:

- Six expansion tests passed for n=2, 4, 64, 144 and 256, both move formats,
  default and non-default streams, input preservation, both devices, empty
  and exact-capacity views, short views, overlaps, alignment, host pointers,
  wrong-device stream, device restoration and wrong-thread rejection. A very
  large logical parent capacity also created a device-only context without a
  corresponding parent allocation. Full-context host results were retained.
- Compute Sanitizer memcheck reported zero errors and zero leaked bytes for
  expansion, borrowed-owner regression and the Rust expansion GPU test.
  Racecheck displayed zero hazards in the expansion workload.
- The Rust workspace and old/new Rust ABI checks passed. Existing one/two-GPU
  CNOT searches and native checkpoint scorer results matched their references.
  The full Python suite reported 835 passed and 10 skipped.
- SM75, SM80, SM90 and SM120 builds succeeded. They establish compilation
  only; physical T4, Hopper, Blackwell, multi-node and Slurm runs were not
  performed in this stage.

The first Vast host failed during provider image setup and was destroyed; the
replacement completed the tests and was also destroyed. Provider absence was
verified for instances `61001341` and `61002149`. Their last durations and
rates imply approximately $0.054 in instance charges combined, excluding
possible fees and traffic. `verify_results.py` checks the source binding,
results, compatibility-fixture correction and release receipts.
